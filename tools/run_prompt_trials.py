#!/usr/bin/env python3
"""Run prompted LLM_Commander trials and record plottable CSV rows.

This is an external orchestration script. It does not import robot runtime
modules; it only launches LLM_Commander.py with trial-specific environment
variables and records operator grading.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINAL_RESULT_CHOICES = {"success", "partial", "fail", "aborted"}
FAILURE_STAGE_CHOICES = {
    "none",
    "startup",
    "observe",
    "classify",
    "grasp",
    "carry",
    "place",
    "verify",
    "correction",
    "recovery",
    "stop_run",
    "shutdown",
    "unknown",
}
FAILURE_SOURCE_CHOICES = {
    "none",
    "perception_detection",
    "perception_depth",
    "tracking_identity",
    "color_classification",
    "manipulation_grasp",
    "manipulation_place",
    "verification_uncertain",
    "state_estimation",
    "planner_action_invalid",
    "planner_state_wrong",
    "hardware_current_overload",
    "hardware_arm_comm",
    "operator_abort",
    "runtime_crash",
    "uncertain_mixed",
    "other",
}
PLANNER_STATE_ALIGNMENT_CHOICES = {
    "none",
    "planner_action_invalid",
    "planner_action_valid_state_wrong",
    "planner_state_uncertain",
    "hardware_perception_failure",
    "uncertain",
}
RECOVERY_SUCCESSFUL_CHOICES = {"not_attempted", "yes", "partial", "no", "uncertain"}
OPERATOR_FAIL_MODE_CHOICES = {
    "cube_fallen",
    "stack_fall",
    "bad_orientation",
    "overcurrent",
    "other",
}
OPERATOR_FAIL_MODE_ALIASES = {
    "1": "cube_fallen",
    "2": "stack_fall",
    "3": "bad_orientation",
    "4": "overcurrent",
    "5": "other",
}

SUMMARY_FIELDS = [
    "session",
    "prompt_id",
    "runs",
    "success",
    "partial",
    "fail",
    "aborted",
    "autonomous_success",
    "assisted_completion",
    "partial_success",
    "safe_handled_failure",
    "unhandled_failure",
    "mean_completion_ratio_auto",
    "mean_num_recovery",
    "mean_total_policy_moves",
    "mean_num_place",
    "mean_num_pick_misplaced",
    "mean_place_verify_xy_error_m",
    "top_failure_source_auto",
]

CSV_FIELDS = [
    "session",
    "trial_id",
    "prompt_id",
    "prompt_index",
    "repeat_index",
    "trial_dir",
    "started_at",
    "ended_at",
    "duration_s",
    "return_code",
    "crashed",
    "prompt_text",
    "prompt_initial_condition",
    "initial_condition",
    "final_result",
    "autonomous",
    "autonomy_level",
    "outcome_alias",
    "notes",
    "video_file",
    "steps_completed_auto",
    "steps_required_auto",
    "completion_ratio_auto",
    "failure_stage_auto",
    "failure_source_auto",
    "failure_confidence",
    "steps_completed",
    "steps_required",
    "completion_ratio",
    "stack_correct_count",
    "stack_desired_count",
    "stack_correct_fraction",
    "operator_fail_mode",
    "operator_fail_mode_other",
    "failure_stage",
    "failure_source",
    "planner_state_alignment",
    "recovery_successful",
    "runtime_picked_count",
    "runtime_placed_count",
    "runtime_returned_count",
    "total_policy_moves",
    "num_observe",
    "num_classify",
    "num_grasp",
    "num_place",
    "num_pick_other",
    "num_pick_misplaced",
    "num_recovery",
    "num_verify",
    "recovery_attempted",
    "auto_recovery_observes",
    "invalid_recoveries",
    "policy_invalid_count",
    "overcurrent_event",
    "current_guard_freeze_events",
    "current_guard_recoveries",
    "current_guard_unrecoverable_events",
    "verify_place_confirmed",
    "verify_place_uncertain",
    "place_verify_xy_error_count",
    "mean_place_verify_xy_error_m",
    "track_missing_id_frames",
    "stop_reason",
    "stop_run_llm_reason",
    "final_left_level",
    "final_left_stack",
    "final_right_level",
    "final_right_stack",
    "final_stack_source",
    "console_log",
    "policy_trace_dir",
]

SUMMARY_RE = re.compile(
    r"Picked=(?P<picked>\d+)\s*\|\s*Placed=(?P<placed>\d+)\s*\|\s*"
    r"stop_reason=(?P<stop>.*?)\s*\|\s*freeze_events=(?P<freeze>\d+)\s*\|\s*"
    r"recoveries=(?P<recoveries>\d+)\s*\|\s*unrecoverable_events=(?P<unrecoverable>\d+)"
)
PROMPTED_STATS_RE = re.compile(
    r"\[PromptedStats\]\s*steps=(?P<steps>\d+)\s*\|\s*reobserve=(?P<reobserve>\d+).*?"
    r"returned=(?P<returned>\d+)\s*\|\s*auto_recovery_observes=(?P<auto>\d+)\s*\|\s*"
    r"invalid_recoveries=(?P<invalid>\d+).*?\|\s*policy_invalid=(?P<policy_invalid>\d+)"
)
VERIFY_STATS_RE = re.compile(
    r"\[VerifyStats\]\s*pick_unstable=(?P<pick_unstable>\d+)\s*\|\s*"
    r"place_confirmed=(?P<confirmed>\d+)\s*\|\s*place_uncertain=(?P<uncertain>\d+)"
)
TRACK_STATS_RE = re.compile(
    r"\[TrackStats\]\s*missing_id_rows_total=(?P<rows>\d+)\s*\|\s*"
    r"missing_id_frames=(?P<frames>\d+)"
)
POLICY_COMMAND_RE = re.compile(r"\[Policy\]\s*step=\d+\s+phase=\S+\s+command=(?P<command>\S+)")
POLICY_DECISION_RE = re.compile(
    r"\[Policy\]\s*step=\d+\s+phase=\S+\s+command=(?P<command>\S+).*?\sreason=(?P<reason>.*)$"
)
STOP_RE = re.compile(r"\[Stop\]\s*(?P<reason>.*?)(?:\.)?$")
STACK_STATE_RE = re.compile(
    r"\[(?P<tag>StackState|PlannerStack)\]\s*source=(?P<source>\S+)\s+"
    r"left\(level=(?P<left_level>\d+),base=(?P<left_base>[^,)]*),middle=(?P<left_middle>[^,)]*),top=(?P<left_top>[^,)]*)"
    r"(?:,[^)]*)?\)\s+"
    r"right\(level=(?P<right_level>\d+),base=(?P<right_base>[^,)]*),middle=(?P<right_middle>[^,)]*),top=(?P<right_top>[^,)]*)"
    r"(?:,[^)]*)?\)"
)
PLACE_VERIFY_V2_XY_RE = re.compile(
    r"\[PlaceVerifyV2\].*?\berr_xy=(?P<err_xy>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|inf)"
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def session_stamp() -> str:
    return "prompt_trials_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except FileNotFoundError as exc:
        raise SystemExit(f"ERROR: prompt file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: invalid JSON in {path}: {exc}") from exc


def load_prompt_file(path: Path) -> tuple[list[dict[str, Any]], int]:
    payload = read_json(path)
    if isinstance(payload, list):
        prompts = payload
        default_repeats = 10
    elif isinstance(payload, dict):
        prompts = payload.get("prompts")
        default_repeats = int(payload.get("default_repeats", 10))
    else:
        raise SystemExit("ERROR: prompts JSON must be an object or list")

    if not isinstance(prompts, list) or not prompts:
        raise SystemExit("ERROR: prompts JSON must contain a non-empty prompts list")

    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(prompts, start=1):
        if not isinstance(row, dict):
            raise SystemExit(f"ERROR: prompt #{idx} must be an object")
        prompt_id = str(row.get("id", "")).strip()
        prompt_text = str(row.get("text", "")).strip()
        initial_condition = str(row.get("initial_condition", "")).strip()
        if not prompt_id:
            raise SystemExit(f"ERROR: prompt #{idx} is missing id")
        if prompt_id in seen:
            raise SystemExit(f"ERROR: duplicate prompt id: {prompt_id}")
        if not prompt_text:
            raise SystemExit(f"ERROR: prompt {prompt_id} is missing text")
        steps_raw = row.get("steps_required", row.get("steps_required_default", ""))
        steps_required: int | str
        if str(steps_raw).strip() == "":
            steps_required = ""
        else:
            try:
                steps_required = int(steps_raw)
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"ERROR: prompt {prompt_id} has invalid steps_required") from exc
            if steps_required <= 0:
                raise SystemExit(f"ERROR: prompt {prompt_id} must have steps_required > 0 when provided")
        seen.add(prompt_id)
        normalized.append(
            {
                "id": prompt_id,
                "text": prompt_text,
                "initial_condition": initial_condition,
                "steps_required_default": steps_required,
                "prompt_index": idx,
            }
        )
    return normalized, max(1, int(default_repeats))


def build_trial_plan(
    prompts: list[dict[str, Any]],
    repeats: int,
    session_dir: Path,
) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    trial_n = 1
    for prompt in prompts:
        for repeat_index in range(1, int(repeats) + 1):
            trial_id = f"trial_{trial_n:04d}"
            trial_dir = session_dir / trial_id
            trials.append(
                {
                    "trial_id": trial_id,
                    "trial_sequence": int(trial_n),
                    "prompt_id": prompt["id"],
                    "prompt_index": int(prompt["prompt_index"]),
                    "repeat_index": int(repeat_index),
                    "prompt_text": prompt["text"],
                    "prompt_initial_condition": prompt.get("initial_condition", ""),
                    "steps_required_default": prompt["steps_required_default"],
                    "trial_dir": trial_dir,
                }
            )
            trial_n += 1
    return trials


def filter_prompts(prompts: list[dict[str, Any]], prompt_filter: str | None) -> list[dict[str, Any]]:
    if not prompt_filter:
        return prompts
    requested = str(prompt_filter).strip().lower()
    selected: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_id = str(prompt["id"])
        index_aliases = {
            str(prompt["prompt_index"]).lower(),
            f"p{prompt['prompt_index']}".lower(),
            f"prompt_{prompt['prompt_index']}".lower(),
        }
        if requested == prompt_id.lower() or requested in index_aliases:
            selected.append(prompt)
    if not selected:
        valid = ", ".join([str(p["id"]) for p in prompts])
        raise SystemExit(f"ERROR: --prompt-id {prompt_filter!r} did not match any prompt. Valid ids: {valid}")
    return selected


def choose_prompt_scope_interactive(prompts: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
    print("")
    print("=" * 72)
    print("Experiment plan:")
    print(f"  Enter/all: all prompts grouped ({repeats} x each prompt)")
    for prompt in prompts:
        print(f"  P{prompt['prompt_index']}: {prompt['id']} ({repeats} repeats)")
    choice = input("Choose experiment type, or press Enter for all: ").strip()
    if not choice or choice.lower() == "all":
        return prompts
    return filter_prompts(prompts, choice)


def read_completed_trial_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        return {str(row.get("trial_id", "")).strip() for row in reader if row.get("trial_id")}


def read_trial_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        return [dict(row) for row in reader]


def trial_key_from_values(prompt_id: object, repeat_index: object) -> str:
    return f"{str(prompt_id).strip()}#{int(repeat_index)}"


def trial_key(trial: dict[str, Any]) -> str:
    return trial_key_from_values(trial["prompt_id"], trial["repeat_index"])


def read_completed_trial_keys(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    keys: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            prompt_id = str(row.get("prompt_id", "")).strip()
            repeat_raw = str(row.get("repeat_index", "")).strip()
            if not prompt_id or not repeat_raw:
                continue
            try:
                keys.add(trial_key_from_values(prompt_id, repeat_raw))
            except ValueError:
                continue
    return keys


def last_initial_condition_by_prompt(csv_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in read_trial_rows(csv_path):
        prompt_id = str(row.get("prompt_id", "")).strip()
        initial_condition = str(row.get("initial_condition", "")).strip()
        if prompt_id and initial_condition:
            out[prompt_id] = initial_condition
    return out


def completed_counts_by_prompt(trials: list[dict[str, Any]], completed_trial_keys: set[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for trial in trials:
        prompt_id = str(trial["prompt_id"])
        row = counts.setdefault(prompt_id, {"completed": 0, "total": 0})
        row["total"] += 1
        if trial_key(trial) in completed_trial_keys:
            row["completed"] += 1
    return counts


def next_available_trial_number(session_dir: Path) -> int:
    max_seen = 0
    if session_dir.exists():
        for child in session_dir.iterdir():
            if not child.is_dir():
                continue
            match = re.fullmatch(r"trial_(\d{4,})", child.name)
            if match:
                max_seen = max(max_seen, int(match.group(1)))
    return max_seen + 1


def assign_available_trial_folder(session_dir: Path, trial: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(trial)
    planned_dir = Path(candidate["trial_dir"])
    if not planned_dir.exists():
        return candidate
    next_n = next_available_trial_number(session_dir)
    candidate["trial_id"] = f"trial_{next_n:04d}"
    candidate["trial_dir"] = session_dir / str(candidate["trial_id"])
    return candidate


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reconcile_csv_schema(path)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def reconcile_csv_schema(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        existing_fields = list(reader.fieldnames or [])
        if existing_fields == CSV_FIELDS:
            return
        rows = [dict(row) for row in reader]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def parse_trial_delete_spec(spec: str) -> set[str]:
    requested: set[str] = set()
    for raw_part in str(spec).replace(";", ",").split(","):
        part = raw_part.strip().lower()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = [item.strip() for item in part.split("-", 1)]
            start_i = int(re.sub(r"^trial_", "", start_text))
            end_i = int(re.sub(r"^trial_", "", end_text))
            if end_i < start_i:
                start_i, end_i = end_i, start_i
            for trial_i in range(start_i, end_i + 1):
                requested.add(f"trial_{trial_i:04d}")
            continue
        trial_i = int(re.sub(r"^trial_", "", part))
        requested.add(f"trial_{trial_i:04d}")
    return requested


def wipe_trials_from_session(session_dir: Path, spec: str) -> set[str]:
    try:
        requested = parse_trial_delete_spec(spec)
    except ValueError as exc:
        print(f"[TrialWipe] invalid_spec={spec!r} error={exc}")
        return set()
    if not requested:
        print("No trials requested for deletion.")
        return set()
    trials_csv = session_dir / "trials.csv"
    rows = read_trial_rows(trials_csv)
    kept_rows = [row for row in rows if str(row.get("trial_id", "")).strip() not in requested]
    removed_from_csv = {str(row.get("trial_id", "")).strip() for row in rows if str(row.get("trial_id", "")).strip() in requested}
    if rows or trials_csv.exists():
        write_csv_rows(trials_csv, kept_rows)
    failure_rows = [
        row
        for row in kept_rows
        if str(row.get("final_result", "")).strip().lower() in {"partial", "fail", "aborted"}
    ]
    write_csv_rows(session_dir / "failures.csv", failure_rows)
    deleted_dirs: set[str] = set()
    session_resolved = session_dir.resolve()
    for trial_id in sorted(requested):
        trial_dir = (session_dir / trial_id).resolve()
        if trial_dir != session_resolved and session_resolved in trial_dir.parents and trial_dir.exists():
            shutil.rmtree(trial_dir)
            deleted_dirs.add(trial_id)
    write_summary_csv(session_dir)
    removed = set(removed_from_csv) | set(deleted_dirs)
    missing = sorted(requested - removed)
    print(
        f"[TrialWipe] requested={sorted(requested)} "
        f"removed_rows={sorted(removed_from_csv)} deleted_dirs={sorted(deleted_dirs)}"
    )
    if missing:
        print(f"[TrialWipe] not_found={missing}")
    return removed


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
        fp.write("\n")


def update_progress(
    session_dir: Path,
    *,
    session: str,
    prompts_path: Path,
    repeats: int,
    total_trials: int,
    completed_trial_ids: set[str],
    completed_trial_keys: set[str],
    trials: list[dict[str, Any]],
) -> None:
    per_prompt = completed_counts_by_prompt(trials, completed_trial_keys)
    write_json(
        session_dir / "progress.json",
        {
            "session": session,
            "prompts_file": str(prompts_path),
            "repeats": int(repeats),
            "total_trials": int(total_trials),
            "completed_trials": len(completed_trial_keys),
            "completed_trial_ids": sorted(completed_trial_ids),
            "completed_trial_keys": sorted(completed_trial_keys),
            "per_prompt": per_prompt,
            "last_updated_at": now_iso(),
        },
    )


def blank_metrics() -> dict[str, Any]:
    return {
        "runtime_picked_count": "",
        "runtime_placed_count": "",
        "runtime_returned_count": "",
        "total_policy_moves": 0,
        "num_observe": 0,
        "num_classify": 0,
        "num_grasp": 0,
        "num_place": 0,
        "num_pick_other": 0,
        "num_pick_misplaced": 0,
        "num_recovery": 0,
        "num_verify": 0,
        "recovery_attempted": "false",
        "auto_recovery_observes": 0,
        "invalid_recoveries": 0,
        "policy_invalid_count": 0,
        "overcurrent_event": "false",
        "current_guard_freeze_events": 0,
        "current_guard_recoveries": 0,
        "current_guard_unrecoverable_events": 0,
        "verify_place_confirmed": 0,
        "verify_place_uncertain": 0,
        "place_verify_xy_error_count": 0,
        "mean_place_verify_xy_error_m": "",
        "track_missing_id_frames": 0,
        "stop_reason": "",
        "stop_run_llm_reason": "",
        "final_left_level": "",
        "final_left_stack": "",
        "final_right_level": "",
        "final_right_stack": "",
        "final_stack_source": "",
        "steps_completed_auto": "manual_required",
        "steps_required_auto": "manual_required",
        "completion_ratio_auto": "manual_required",
        "failure_stage_auto": "unknown",
        "failure_source_auto": "uncertain_mixed",
        "failure_confidence": "manual_required",
    }


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_place_verify_xy_errors(text: str) -> list[float]:
    """XY placement error (m) from [PlaceVerifyV2] err_xy= (measured vs expected, XY only)."""
    values: list[float] = []
    for line in str(text).splitlines():
        if "[PlaceVerifyV2]" not in line:
            continue
        match = PLACE_VERIFY_V2_XY_RE.search(line)
        if match is None:
            continue
        raw = str(match.group("err_xy")).strip().lower()
        if raw in {"inf", "+inf", "-inf", "nan"}:
            continue
        try:
            err_xy = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(err_xy):
            continue
        values.append(float(err_xy))
    return values


def parse_console_log(log_path: Path, trace_dir: Path | None = None) -> dict[str, Any]:
    metrics = blank_metrics()
    decisions: list[dict[str, str]] = []
    stop_lines: list[str] = []
    text = ""
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            decision_match = POLICY_DECISION_RE.search(line)
            if decision_match:
                command = decision_match.group("command").strip()
                reason = decision_match.group("reason").strip()
                decisions.append({"command": command, "reason": reason})
                if command == "stop_run":
                    metrics["stop_run_llm_reason"] = reason
            else:
                command_match = POLICY_COMMAND_RE.search(line)
                if command_match:
                    decisions.append({"command": command_match.group("command").strip(), "reason": ""})
            summary_match = SUMMARY_RE.search(line)
            if summary_match:
                metrics["runtime_picked_count"] = parse_int(summary_match.group("picked"))
                metrics["runtime_placed_count"] = parse_int(summary_match.group("placed"))
                metrics["stop_reason"] = summary_match.group("stop").strip()
                metrics["current_guard_freeze_events"] = parse_int(summary_match.group("freeze"))
                metrics["current_guard_recoveries"] = parse_int(summary_match.group("recoveries"))
                metrics["current_guard_unrecoverable_events"] = parse_int(summary_match.group("unrecoverable"))
            prompted_match = PROMPTED_STATS_RE.search(line)
            if prompted_match:
                metrics["total_policy_moves"] = parse_int(prompted_match.group("steps"))
                metrics["runtime_returned_count"] = parse_int(prompted_match.group("returned"))
                metrics["auto_recovery_observes"] = parse_int(prompted_match.group("auto"))
                metrics["invalid_recoveries"] = parse_int(prompted_match.group("invalid"))
                metrics["policy_invalid_count"] = parse_int(prompted_match.group("policy_invalid"))
                metrics["num_recovery"] = metrics["invalid_recoveries"]
            verify_match = VERIFY_STATS_RE.search(line)
            if verify_match:
                metrics["verify_place_confirmed"] = parse_int(verify_match.group("confirmed"))
                metrics["verify_place_uncertain"] = parse_int(verify_match.group("uncertain"))
            track_match = TRACK_STATS_RE.search(line)
            if track_match:
                metrics["track_missing_id_frames"] = parse_int(track_match.group("frames"))
            stop_match = STOP_RE.search(line)
            if stop_match:
                stop_lines.append(stop_match.group("reason").strip())
            stack_match = STACK_STATE_RE.search(line)
            if stack_match:
                left_parts = [
                    stack_match.group("left_base"),
                    stack_match.group("left_middle"),
                    stack_match.group("left_top"),
                ]
                right_parts = [
                    stack_match.group("right_base"),
                    stack_match.group("right_middle"),
                    stack_match.group("right_top"),
                ]
                metrics["final_left_level"] = parse_int(stack_match.group("left_level"))
                metrics["final_left_stack"] = compact_stack_colors(left_parts)
                metrics["final_right_level"] = parse_int(stack_match.group("right_level"))
                metrics["final_right_stack"] = compact_stack_colors(right_parts)
                metrics["final_stack_source"] = f"{stack_match.group('tag')}:{stack_match.group('source')}"

    if trace_dir is not None:
        trace_decisions = policy_decisions_from_raw_trace(trace_dir)
        if not decisions:
            decisions = trace_decisions
        if not metrics["stop_run_llm_reason"]:
            for decision in trace_decisions:
                if decision.get("command") == "stop_run" and decision.get("reason"):
                    metrics["stop_run_llm_reason"] = str(decision["reason"]).strip()

    if decisions:
        commands = [str(decision.get("command", "")).strip() for decision in decisions]
        metrics["total_policy_moves"] = len([cmd for cmd in commands if cmd])
        metrics["num_observe"] = sum(1 for cmd in commands if cmd == "observe_scene")
        metrics["num_classify"] = sum(1 for cmd in commands if cmd == "classify_cube")
        metrics["num_grasp"] = sum(1 for cmd in commands if cmd == "grasp_cube")
        metrics["num_place"] = sum(1 for cmd in commands if cmd.startswith("place_"))
        metrics["num_pick_other"] = sum(1 for cmd in commands if cmd == "pick_other")
        metrics["num_pick_misplaced"] = sum(
            1
            for cmd in commands
            if cmd in {"pick_placed_left", "pick_placed_right", "pick_misplaced_left", "pick_misplaced_right"}
        )
        metrics["num_verify"] = sum(1 for cmd in commands if cmd == "verify_last_place")

    if metrics["num_recovery"] == 0:
        metrics["num_recovery"] = max(
            parse_int(str(metrics["invalid_recoveries"])),
            text.count("source=auto_recovery") + text.count("run_auto_recovery_observe"),
        )
    if metrics["num_verify"] == 0:
        metrics["num_verify"] = parse_int(str(metrics["verify_place_confirmed"])) + parse_int(
            str(metrics["verify_place_uncertain"])
        )
    metrics["recovery_attempted"] = "true" if parse_int(str(metrics["num_recovery"])) > 0 else "false"

    lowered = text.lower()
    guard_count_event = any(
        parse_int(str(metrics[field])) > 0
        for field in (
            "current_guard_freeze_events",
            "current_guard_recoveries",
            "current_guard_unrecoverable_events",
        )
    )
    if guard_count_event or "overcurrent" in lowered or "[currentguard] event=" in lowered:
        metrics["overcurrent_event"] = "true"
    if not metrics["stop_reason"] and stop_lines:
        metrics["stop_reason"] = stop_lines[-1]

    xy_errors = parse_place_verify_xy_errors(text)
    metrics["place_verify_xy_error_count"] = int(len(xy_errors))
    metrics["mean_place_verify_xy_error_m"] = mean_string(xy_errors) if xy_errors else ""

    return metrics


def compact_stack_colors(parts: list[str]) -> str:
    out: list[str] = []
    for part in parts:
        color = str(part).strip().lower()
        if color in {"empty", "unknown", "none", ""}:
            continue
        if color.startswith("blue"):
            out.append("B")
        elif color.startswith("orange"):
            out.append("O")
        else:
            out.append(color[:1].upper())
    return "".join(out)


def policy_decisions_from_raw_trace(trace_dir: Path) -> list[dict[str, str]]:
    decisions: list[dict[str, str]] = []
    for trace_path in sorted(trace_dir.glob("policy_raw_*.jsonl")):
        try:
            with trace_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    raw_output = str(row.get("raw_output", ""))
                    decision = policy_decision_from_raw_output(raw_output)
                    if decision.get("command"):
                        decisions.append(decision)
        except (OSError, json.JSONDecodeError):
            continue
    return decisions


def policy_decision_from_raw_output(raw_output: str) -> dict[str, str]:
    text = str(raw_output or "")
    block_match = re.search(r"FINAL_JSON_START\s*(?P<payload>\{.*?\})\s*FINAL_JSON_END", text, re.DOTALL)
    if block_match:
        try:
            payload = json.loads(block_match.group("payload"))
            return {
                "command": str(payload.get("command", "")).strip(),
                "reason": str(payload.get("reason", "")).strip(),
            }
        except json.JSONDecodeError:
            pass
    command_match = re.search(r'"command"\s*:\s*"(?P<command>[^"]+)"', text)
    reason_match = re.search(r'"reason"\s*:\s*"(?P<reason>[^"]*)"', text)
    return {
        "command": command_match.group("command").strip() if command_match else "",
        "reason": reason_match.group("reason").strip() if reason_match else "",
    }


def completion_ratio(steps_completed: int, steps_required: int) -> str:
    if steps_required <= 0:
        return ""
    return f"{max(0.0, min(1.0, float(steps_completed) / float(steps_required))):.3f}"


def is_positive_int(value: object) -> bool:
    try:
        return int(str(value).strip()) > 0
    except (TypeError, ValueError):
        return False


def canonical_steps_required_from_history(
    session_dir: Path,
    *,
    prompt_id: str,
    initial_condition: str,
) -> str:
    for row in reversed(read_trial_rows(session_dir / "trials.csv")):
        if str(row.get("prompt_id", "")).strip() != str(prompt_id).strip():
            continue
        if str(row.get("initial_condition", "")).strip() != str(initial_condition).strip():
            continue
        if str(row.get("final_result", "")).strip().lower() != "success":
            continue
        steps_completed = str(row.get("steps_completed_auto", row.get("steps_completed", ""))).strip()
        if is_positive_int(steps_completed):
            return str(int(steps_completed))
    return "manual_required"


def resolve_steps_required_auto(
    session_dir: Path,
    *,
    trial: dict[str, Any],
    initial_condition: str,
) -> str:
    prompt_steps = str(trial.get("steps_required_default", "")).strip()
    if is_positive_int(prompt_steps):
        return str(int(prompt_steps))
    return canonical_steps_required_from_history(
        session_dir,
        prompt_id=str(trial["prompt_id"]),
        initial_condition=str(initial_condition),
    )


def infer_steps_completed_auto(metrics: dict[str, Any], steps_required: str = "") -> str:
    action_score = parse_int(str(metrics.get("num_place", "0"))) + parse_int(str(metrics.get("num_pick_misplaced", "0")))
    if is_positive_int(steps_required) and action_score > 0:
        return str(min(action_score, int(steps_required)))
    placed_count = str(metrics.get("runtime_placed_count", "")).strip()
    if placed_count != "" and placed_count.lower() != "manual_required":
        try:
            return str(max(0, int(placed_count)))
        except ValueError:
            pass
    num_place = str(metrics.get("num_place", "")).strip()
    if num_place != "" and num_place.lower() != "manual_required":
        try:
            return str(max(0, int(num_place)))
        except ValueError:
            pass
    return "manual_required"


def infer_failure_auto(metrics: dict[str, Any], *, crashed: bool) -> tuple[str, str, str]:
    stop_reason = str(metrics.get("stop_reason", "")).strip().lower()
    if crashed:
        return "unknown", "runtime_crash", "high"
    if str(metrics.get("overcurrent_event", "")).strip().lower() == "true":
        return "carry", "hardware_current_overload", "high"
    if str(metrics.get("policy_invalid_count", "0")).strip() not in {"", "0"} or stop_reason.startswith("policy_invalid:"):
        return "stop_run", "planner_action_invalid", "high"
    if "place_failed" in stop_reason or "place_fail" in stop_reason:
        return "place", "manipulation_place", "high"
    if "grasp_stop" in stop_reason or "grasp_retry" in stop_reason or "grasp" in stop_reason:
        return "grasp", "manipulation_grasp", "medium"
    if parse_int(str(metrics.get("verify_place_uncertain", "0"))) > 0:
        return "verify", "verification_uncertain", "medium"
    if parse_int(str(metrics.get("track_missing_id_frames", "0"))) > 0:
        return "observe", "tracking_identity", "low"
    if "depth_invalid" in stop_reason:
        return "observe", "perception_depth", "medium"
    if "no placement progress" in stop_reason:
        return "recovery", "state_estimation", "medium"
    if "policy requested stop_run" in stop_reason:
        return "none", "none", "none"
    if stop_reason:
        return "unknown", "uncertain_mixed", "low"
    return "unknown", "uncertain_mixed", "manual_required"


def build_auto_fields(
    *,
    session_dir: Path,
    trial: dict[str, Any],
    initial_condition: str,
    metrics: dict[str, Any],
    crashed: bool,
) -> dict[str, Any]:
    steps_required = resolve_steps_required_auto(
        session_dir,
        trial=trial,
        initial_condition=initial_condition,
    )
    steps_completed = infer_steps_completed_auto(metrics, steps_required)
    ratio = "manual_required"
    if steps_completed.isdigit() and steps_required.isdigit() and int(steps_required) > 0:
        ratio = completion_ratio(int(steps_completed), int(steps_required))
    failure_stage, failure_source, failure_confidence = infer_failure_auto(metrics, crashed=crashed)
    return {
        "steps_completed_auto": steps_completed,
        "steps_required_auto": steps_required,
        "completion_ratio_auto": ratio,
        "completion_ratio": ratio,
        "failure_stage_auto": failure_stage,
        "failure_source_auto": failure_source,
        "failure_confidence": failure_confidence,
        # Legacy aliases retained for existing analysis notebooks.
        "steps_completed": steps_completed,
        "steps_required": steps_required,
        "failure_stage": failure_stage,
        "failure_source": failure_source,
    }


def adjust_auto_fields_for_manual_result(auto_fields: dict[str, Any], final_result: str, metrics: dict[str, Any]) -> dict[str, Any]:
    adjusted = dict(auto_fields)
    result = str(final_result).strip().lower()
    if result == "success":
        adjusted["failure_stage_auto"] = "none"
        adjusted["failure_source_auto"] = "none"
        adjusted["failure_confidence"] = "none"
        adjusted["failure_stage"] = "none"
        adjusted["failure_source"] = "none"
        return adjusted
    if result in {"partial", "fail", "aborted"} and str(adjusted.get("failure_source_auto")) == "none":
        stop_reason = str(metrics.get("stop_reason", "")).strip().lower()
        if "policy requested stop_run" in stop_reason:
            adjusted["failure_stage_auto"] = "stop_run"
            adjusted["failure_source_auto"] = "planner_state_wrong"
            adjusted["failure_confidence"] = "low"
            adjusted["failure_stage"] = adjusted["failure_stage_auto"]
            adjusted["failure_source"] = adjusted["failure_source_auto"]
    return adjusted


def normalize_choice(value: str, choices: set[str], *, field: str, default: str | None = None) -> str:
    value = str(value).strip().lower()
    if not value and default is not None:
        return default
    if value in choices:
        return value
    raise ValueError(f"{field} must be one of: {', '.join(sorted(choices))}")


def prompt_text(field: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{field}{suffix}: ").strip()
    return value if value else default


def prompt_choice(field: str, choices: set[str], default: str, aliases: dict[str, str] | None = None) -> str:
    aliases = aliases or {}
    while True:
        print(f"{field} choices: {', '.join(sorted(choices))}")
        raw = input(f"{field} [{default}]: ").strip().lower()
        if raw in aliases:
            raw = aliases[raw]
        try:
            return normalize_choice(raw, choices, field=field, default=default)
        except ValueError as exc:
            print(exc)


def prompt_yes_no(field: str, default: str = "yes") -> str:
    aliases = {
        "": default,
        "y": "yes",
        "yes": "yes",
        "n": "no",
        "no": "no",
    }
    while True:
        raw = input(f"{field} (y/n) [{default}]: ").strip().lower()
        if raw in aliases:
            return aliases[raw]
        print(f"{field} must be y or n")


def prompt_int_range(field: str, *, min_value: int, max_value: int, default: int | None = None) -> int:
    default_text = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{field} ({min_value}-{max_value}){default_text}: ").strip()
        if not raw and default is not None:
            return default
        try:
            value = int(raw)
        except ValueError:
            print(f"{field} must be an integer from {min_value} to {max_value}")
            continue
        if min_value <= value <= max_value:
            return value
        print(f"{field} must be from {min_value} to {max_value}")


def prompt_positive_int(field: str, default: int | None = None) -> int:
    default_text = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{field}{default_text}: ").strip()
        if not raw and default is not None:
            return default
        try:
            value = int(raw)
        except ValueError:
            print(f"{field} must be a positive integer")
            continue
        if value > 0:
            return value
        print(f"{field} must be greater than 0")


def positive_int_or_none(value: object) -> int | None:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if parsed > 0:
        return parsed
    return None


def desired_stack_count_from_context(
    *,
    trial: dict[str, Any] | None,
    auto_fields: dict[str, Any] | None,
) -> int | None:
    auto_fields = auto_fields or {}
    trial = trial or {}
    return (
        positive_int_or_none(auto_fields.get("steps_required_auto"))
        or positive_int_or_none(auto_fields.get("steps_required"))
        or positive_int_or_none(trial.get("steps_required_default"))
    )


def collect_grading(
    default_final_result: str = "fail",
    *,
    trial: dict[str, Any] | None = None,
    auto_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    print("Outcome shortcuts: s=success, p=partial, f=fail, a=aborted")
    final_result = prompt_choice(
        "final_result",
        FINAL_RESULT_CHOICES,
        default_final_result,
        aliases={"s": "success", "p": "partial", "f": "fail", "a": "aborted", "abort": "aborted"},
    )
    autonomous = prompt_yes_no(
        "autonomous_no_human_correction "
        "(n if you reoriented a cube, fixed a dropped stack, moved a cube, or otherwise intervened)",
        "yes",
    )
    desired_count = desired_stack_count_from_context(trial=trial, auto_fields=auto_fields)
    stack_correct_count: str | int = ""
    stack_desired_count: str | int = desired_count if desired_count is not None else ""
    stack_correct_fraction = ""
    manual_steps: dict[str, Any] = {}
    if final_result == "partial":
        if desired_count is None:
            print("Desired stack count was not available from the prompt metadata.")
            desired_count = prompt_positive_int("stack_desired_count")
            stack_desired_count = desired_count
        else:
            print(f"stack_desired_count: {desired_count}")
        correct_count = prompt_int_range(
            "stack_correct_count",
            min_value=0,
            max_value=desired_count,
        )
        stack_correct_count = correct_count
        stack_correct_fraction = completion_ratio(correct_count, desired_count)
        manual_steps = {
            "steps_completed": str(correct_count),
            "steps_required": str(desired_count),
            "completion_ratio": stack_correct_fraction,
        }
    elif final_result == "success" and desired_count is not None:
        stack_correct_count = desired_count
        stack_correct_fraction = "1.000"

    operator_fail_mode = ""
    operator_fail_mode_other = ""
    if final_result in {"partial", "fail", "aborted"}:
        print("Failure mode shortcuts: 1=cube_fallen, 2=stack_fall, 3=bad_orientation, 4=overcurrent, 5=other")
        operator_fail_mode = prompt_choice(
            "operator_fail_mode",
            OPERATOR_FAIL_MODE_CHOICES,
            "other",
            aliases=OPERATOR_FAIL_MODE_ALIASES,
        )
        if operator_fail_mode == "other":
            operator_fail_mode_other = prompt_text("operator_fail_mode_other")

    grading = {
        "final_result": final_result,
        "autonomous": autonomous,
        "notes": "",
        "video_file": "",
        "stack_correct_count": stack_correct_count,
        "stack_desired_count": stack_desired_count,
        "stack_correct_fraction": stack_correct_fraction,
        "operator_fail_mode": operator_fail_mode,
        "operator_fail_mode_other": operator_fail_mode_other,
        "planner_state_alignment": "not_asked",
        "recovery_successful": "not_asked",
    }
    grading.update(manual_steps)
    return grading


def truthy_csv(value: object) -> bool:
    return str(value).strip().lower() in {"true", "yes", "1"}


def positive_csv(value: object) -> bool:
    return parse_int(str(value), 0) > 0


def derive_outcome_fields(row: dict[str, Any]) -> dict[str, str]:
    final_result = str(row.get("final_result", "")).strip().lower()
    autonomous = str(row.get("autonomous", "")).strip().lower()
    if autonomous not in {"yes", "no"}:
        autonomous = "unknown"

    if autonomous == "yes":
        autonomy_level = "autonomous"
    elif autonomous == "no" and final_result in {"success", "partial"}:
        autonomy_level = "assisted_reset"
    elif autonomous == "no":
        autonomy_level = "manual_stop_or_unhandled"
    else:
        autonomy_level = "unknown"

    safe_handled = (
        truthy_csv(row.get("recovery_attempted"))
        or truthy_csv(row.get("overcurrent_event"))
        or positive_csv(row.get("num_recovery"))
        or positive_csv(row.get("current_guard_freeze_events"))
        or positive_csv(row.get("current_guard_recoveries"))
        or positive_csv(row.get("current_guard_unrecoverable_events"))
    )
    if final_result == "success" and autonomous == "yes":
        outcome_alias = "autonomous_success"
    elif final_result == "success":
        outcome_alias = "assisted_completion"
    elif final_result == "partial":
        outcome_alias = "partial_success"
    elif final_result in {"fail", "aborted"} and safe_handled:
        outcome_alias = "safe_handled_failure"
    elif final_result in {"fail", "aborted"}:
        outcome_alias = "unhandled_failure"
    else:
        outcome_alias = "unknown"
    return {
        "autonomous": autonomous,
        "autonomy_level": autonomy_level,
        "outcome_alias": outcome_alias,
    }


def make_row(
    *,
    session: str,
    trial: dict[str, Any],
    started_at: str,
    ended_at: str,
    duration_s: float,
    return_code: int | str,
    crashed: bool,
    initial_condition: str,
    grading: dict[str, Any],
    metrics: dict[str, Any],
    auto_fields: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "session": session,
        "trial_id": trial["trial_id"],
        "prompt_id": trial["prompt_id"],
        "prompt_index": int(trial["prompt_index"]),
        "repeat_index": int(trial["repeat_index"]),
        "trial_dir": str(Path(trial["trial_dir"]).resolve()),
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_s": f"{float(duration_s):.3f}",
        "return_code": return_code,
        "crashed": "true" if crashed else "false",
        "prompt_text": trial["prompt_text"],
        "prompt_initial_condition": str(trial.get("prompt_initial_condition", "")),
        "initial_condition": initial_condition,
        "console_log": str((Path(trial["trial_dir"]) / "console_log.txt").resolve()),
        "policy_trace_dir": str(Path(trial["trial_dir"]).resolve()),
    }
    row.update(blank_metrics())
    row.update(metrics)
    row.update(auto_fields)
    row.update(grading)
    row.update(derive_outcome_fields(row))
    return row


def append_trial_outputs(
    *,
    session_dir: Path,
    row: dict[str, Any],
    completed_trial_ids: set[str],
    completed_trial_keys: set[str],
) -> None:
    trials_csv = session_dir / "trials.csv"
    failures_csv = session_dir / "failures.csv"
    append_csv_row(trials_csv, row)
    if str(row.get("final_result", "")).strip().lower() in {"partial", "fail", "aborted"}:
        append_csv_row(failures_csv, row)
    completed_trial_ids.add(str(row["trial_id"]))
    completed_trial_keys.add(trial_key_from_values(row["prompt_id"], row["repeat_index"]))
    write_summary_csv(session_dir)


def parse_float_or_none(value: object) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def mean_string(values: list[float]) -> str:
    if not values:
        return ""
    return f"{(sum(values) / len(values)):.3f}"


def top_value(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        norm = str(value).strip()
        if not norm or norm == "none":
            continue
        counts[norm] = counts.get(norm, 0) + 1
    if not counts:
        return ""
    key, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return f"{key}:{count}"


def write_summary_csv(session_dir: Path) -> None:
    rows = read_trial_rows(session_dir / "trials.csv")
    by_prompt: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        prompt_id = str(row.get("prompt_id", "")).strip()
        if prompt_id:
            by_prompt.setdefault(prompt_id, []).append(row)
    summary_path = session_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for prompt_id in sorted(by_prompt):
            prompt_rows = by_prompt[prompt_id]
            result_counts = {key: 0 for key in ["success", "partial", "fail", "aborted"]}
            outcome_counts = {
                key: 0
                for key in [
                    "autonomous_success",
                    "assisted_completion",
                    "partial_success",
                    "safe_handled_failure",
                    "unhandled_failure",
                ]
            }
            for row in prompt_rows:
                result = str(row.get("final_result", "")).strip().lower()
                if result in result_counts:
                    result_counts[result] += 1
                outcome = str(row.get("outcome_alias", "")).strip().lower()
                if outcome in outcome_counts:
                    outcome_counts[outcome] += 1
            completion_values = [
                value
                for value in (parse_float_or_none(row.get("completion_ratio_auto")) for row in prompt_rows)
                if value is not None
            ]
            recovery_values = [
                value
                for value in (parse_float_or_none(row.get("num_recovery")) for row in prompt_rows)
                if value is not None
            ]
            policy_move_values = [
                value
                for value in (parse_float_or_none(row.get("total_policy_moves")) for row in prompt_rows)
                if value is not None
            ]
            place_values = [
                value
                for value in (parse_float_or_none(row.get("num_place")) for row in prompt_rows)
                if value is not None
            ]
            correction_values = [
                value
                for value in (parse_float_or_none(row.get("num_pick_misplaced")) for row in prompt_rows)
                if value is not None
            ]
            place_xy_values = [
                value
                for value in (parse_float_or_none(row.get("mean_place_verify_xy_error_m")) for row in prompt_rows)
                if value is not None
            ]
            writer.writerow(
                {
                    "session": str(prompt_rows[0].get("session", "")),
                    "prompt_id": prompt_id,
                    "runs": len(prompt_rows),
                    "success": result_counts["success"],
                    "partial": result_counts["partial"],
                    "fail": result_counts["fail"],
                    "aborted": result_counts["aborted"],
                    "autonomous_success": outcome_counts["autonomous_success"],
                    "assisted_completion": outcome_counts["assisted_completion"],
                    "partial_success": outcome_counts["partial_success"],
                    "safe_handled_failure": outcome_counts["safe_handled_failure"],
                    "unhandled_failure": outcome_counts["unhandled_failure"],
                    "mean_completion_ratio_auto": mean_string(completion_values),
                    "mean_num_recovery": mean_string(recovery_values),
                    "mean_total_policy_moves": mean_string(policy_move_values),
                    "mean_num_place": mean_string(place_values),
                    "mean_num_pick_misplaced": mean_string(correction_values),
                    "mean_place_verify_xy_error_m": mean_string(place_xy_values),
                    "top_failure_source_auto": top_value(
                        [str(row.get("failure_source_auto", "")) for row in prompt_rows]
                    ),
                }
            )


def launch_trial(
    *,
    python_exe: str,
    root: Path,
    trial: dict[str, Any],
    initial_condition: str,
) -> tuple[int | str, bool, str, str, float]:
    trial_dir = Path(trial["trial_dir"])
    trial_dir.mkdir(parents=True, exist_ok=True)
    console_log = trial_dir / "console_log.txt"
    started_at = now_iso()
    start_time = time.monotonic()

    env = os.environ.copy()
    env["QARM_MISSION_PROMPT"] = str(trial["prompt_text"])
    env["QARM_POLICY_TRACE_DIR"] = str(trial_dir.resolve())
    env["PYTHONUNBUFFERED"] = "1"

    write_json(
        trial_dir / "trial_meta.json",
        {
            "status": "running",
            "trial_id": trial["trial_id"],
            "prompt_id": trial["prompt_id"],
            "prompt_index": int(trial["prompt_index"]),
            "repeat_index": int(trial["repeat_index"]),
            "prompt_text": trial["prompt_text"],
            "steps_required_default": trial["steps_required_default"],
            "prompt_initial_condition": str(trial.get("prompt_initial_condition", "")),
            "initial_condition": str(initial_condition),
            "started_at": started_at,
            "command": [python_exe, "LLM_Commander.py"],
            "env": {
                "QARM_MISSION_PROMPT": str(trial["prompt_text"]),
                "QARM_POLICY_TRACE_DIR": str(trial_dir.resolve()),
                "PYTHONUNBUFFERED": "1",
            },
        },
    )

    crashed = False
    return_code: int | str = ""
    proc: subprocess.Popen[str] | None = None
    try:
        with console_log.open("a", encoding="utf-8", errors="replace") as log_fp:
            log_fp.write(f"[TrialRunner] started_at={started_at}\n")
            log_fp.write(f"[TrialRunner] command={python_exe} LLM_Commander.py\n")
            log_fp.flush()
            proc = subprocess.Popen(
                [python_exe, "LLM_Commander.py"],
                cwd=str(root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log_fp.write(line)
                log_fp.flush()
            return_code = proc.wait()
    except KeyboardInterrupt:
        crashed = True
        return_code = "keyboard_interrupt"
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=8)
        with console_log.open("a", encoding="utf-8", errors="replace") as log_fp:
            log_fp.write("[TrialRunnerAbort] keyboard_interrupt received; subprocess terminated.\n")
        print(
            f"\nRecorded interrupt for {trial['trial_id']}; "
            "the runner will mark this trial aborted and pause before the next one.",
            file=sys.stderr,
        )
    except Exception as exc:
        crashed = True
        return_code = "launch_error"
        with console_log.open("a", encoding="utf-8", errors="replace") as log_fp:
            log_fp.write(f"[TrialRunnerError] {type(exc).__name__}: {exc}\n")
        print(f"ERROR: failed to run trial {trial['trial_id']}: {exc}", file=sys.stderr)

    ended_at = now_iso()
    duration_s = time.monotonic() - start_time
    if isinstance(return_code, int) and return_code != 0:
        crashed = True

    meta_path = trial_dir / "trial_meta.json"
    meta_payload = read_json(meta_path) if meta_path.exists() else {}
    if not isinstance(meta_payload, dict):
        meta_payload = {}
    meta_payload.update(
        {
            "status": "finished",
            "ended_at": ended_at,
            "duration_s": float(duration_s),
            "return_code": return_code,
            "crashed": bool(crashed),
        }
    )
    write_json(meta_path, meta_payload)
    return return_code, crashed, started_at, ended_at, duration_s


def auto_abort_grading(
    *,
    notes: str,
    recovery_successful: str = "uncertain",
    autonomous: str = "unknown",
) -> dict[str, Any]:
    return {
        "final_result": "aborted",
        "autonomous": autonomous,
        "planner_state_alignment": "uncertain",
        "recovery_successful": recovery_successful,
        "notes": notes,
        "video_file": "",
    }


def color_name(char: str) -> str:
    norm = str(char).strip().upper()
    if norm == "B":
        return "BLUE"
    if norm == "O":
        return "ORANGE"
    return norm or "unknown"


def describe_stack_sequence(seq: str) -> str:
    seq = str(seq).strip().upper()
    if not seq:
        return "empty"
    return " -> ".join(color_name(ch) for ch in seq)


def describe_pick_inventory(text: str) -> str:
    text = str(text).strip().upper()
    if not text:
        return "empty"
    parts = re.findall(r"([BO])\s*(\d*)", text)
    if not parts:
        return text
    labels: list[str] = []
    for color, count_raw in parts:
        count = int(count_raw) if count_raw else 1
        labels.append(f"{count} {color_name(color)}")
    return ", ".join(labels)


def parse_initial_condition(initial_condition: str) -> dict[str, str]:
    parsed = {"L": "", "R": "", "P": ""}
    for chunk in str(initial_condition).split(";"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip().upper()
        if key in parsed:
            parsed[key] = value.strip().upper()
    return parsed


def describe_initial_condition(initial_condition: str) -> list[str]:
    parsed = parse_initial_condition(initial_condition)
    return [
        f"Left stack (bottom->top): {describe_stack_sequence(parsed['L'])}",
        f"Right stack (bottom->top): {describe_stack_sequence(parsed['R'])}",
        f"Pick-space inventory: {describe_pick_inventory(parsed['P'])}",
    ]


def abort_trial_before_run(
    *,
    session: str,
    session_dir: Path,
    trial: dict[str, Any],
    initial_condition: str,
    completed_trial_ids: set[str],
    completed_trial_keys: set[str],
) -> None:
    trial_dir = Path(trial["trial_dir"])
    trial_dir.mkdir(parents=True, exist_ok=True)
    started_at = now_iso()
    ended_at = started_at
    write_json(
        trial_dir / "trial_meta.json",
        {
            "status": "operator_aborted_before_run",
            "trial_id": trial["trial_id"],
            "prompt_id": trial["prompt_id"],
            "prompt_index": int(trial["prompt_index"]),
            "repeat_index": int(trial["repeat_index"]),
            "prompt_text": trial["prompt_text"],
            "prompt_initial_condition": str(trial.get("prompt_initial_condition", "")),
            "started_at": started_at,
            "ended_at": ended_at,
        },
    )
    metrics = blank_metrics()
    metrics["runtime_placed_count"] = 0
    auto_fields = build_auto_fields(
        session_dir=session_dir,
        trial=trial,
        initial_condition=initial_condition,
        metrics=metrics,
        crashed=False,
    )
    auto_fields.update(
        {
            "steps_completed_auto": "0",
            "steps_completed": "0",
            "failure_stage_auto": "unknown",
            "failure_stage": "unknown",
            "failure_source_auto": "operator_abort",
            "failure_source": "operator_abort",
            "failure_confidence": "high",
        }
    )
    row = make_row(
        session=session,
        trial=trial,
        started_at=started_at,
        ended_at=ended_at,
        duration_s=0.0,
        return_code="operator_abort",
        crashed=False,
        initial_condition=initial_condition,
        grading=auto_abort_grading(
            notes="Operator aborted before launching LLM_Commander.py.",
            recovery_successful="not_attempted",
            autonomous="no",
        ),
        metrics=metrics,
        auto_fields=auto_fields,
    )
    append_trial_outputs(
        session_dir=session_dir,
        row=row,
        completed_trial_ids=completed_trial_ids,
        completed_trial_keys=completed_trial_keys,
    )


def auto_abort_crashed_trial(
    *,
    session: str,
    session_dir: Path,
    trial: dict[str, Any],
    started_at: str,
    ended_at: str,
    duration_s: float,
    return_code: int | str,
    initial_condition: str,
    notes: str,
    completed_trial_ids: set[str],
    completed_trial_keys: set[str],
) -> None:
    log_path = Path(trial["trial_dir"]) / "console_log.txt"
    metrics = parse_console_log(log_path, Path(trial["trial_dir"]))
    auto_fields = build_auto_fields(
        session_dir=session_dir,
        trial=trial,
        initial_condition=initial_condition,
        metrics=metrics,
        crashed=True,
    )
    row = make_row(
        session=session,
        trial=trial,
        started_at=started_at,
        ended_at=ended_at,
        duration_s=duration_s,
        return_code=return_code,
        crashed=True,
        initial_condition=initial_condition,
        grading=auto_abort_grading(
            notes=notes,
        ),
        metrics=metrics,
        auto_fields=auto_fields,
    )
    append_trial_outputs(
        session_dir=session_dir,
        row=row,
        completed_trial_ids=completed_trial_ids,
        completed_trial_keys=completed_trial_keys,
    )


def finalize_orphaned_running_trials(
    *,
    session: str,
    session_dir: Path,
    completed_trial_ids: set[str],
    completed_trial_keys: set[str],
) -> None:
    for meta_path in sorted(session_dir.glob("trial_*/trial_meta.json")):
        try:
            meta = read_json(meta_path)
        except SystemExit:
            continue
        if not isinstance(meta, dict):
            continue
        trial_id = str(meta.get("trial_id", meta_path.parent.name)).strip()
        orphan_key = ""
        try:
            orphan_key = trial_key_from_values(meta.get("prompt_id", ""), meta.get("repeat_index", ""))
        except (TypeError, ValueError):
            orphan_key = ""
        if not trial_id or trial_id in completed_trial_ids or (orphan_key and orphan_key in completed_trial_keys):
            continue
        if str(meta.get("status", "")).strip().lower() != "running":
            continue
        trial = {
            "trial_id": trial_id,
            "prompt_id": str(meta.get("prompt_id", "")),
            "prompt_index": int(meta.get("prompt_index", 0) or 0),
            "repeat_index": int(meta.get("repeat_index", 0) or 0),
            "prompt_text": str(meta.get("prompt_text", "")),
            "prompt_initial_condition": str(meta.get("prompt_initial_condition", "")),
            "steps_required_default": meta.get("steps_required_default", ""),
            "trial_dir": meta_path.parent,
        }
        started_at = str(meta.get("started_at", "") or "")
        ended_at = now_iso()
        write_json(
            meta_path,
            {
                **meta,
                "status": "auto_aborted_on_resume",
                "ended_at": ended_at,
                "return_code": "orphaned_running_trial",
                "crashed": True,
            },
        )
        auto_abort_crashed_trial(
            session=session,
            session_dir=session_dir,
            trial=trial,
            started_at=started_at,
            ended_at=ended_at,
            duration_s=0.0,
            return_code="orphaned_running_trial",
            initial_condition=str(meta.get("initial_condition", "")),
            notes="Runner resumed and found this trial marked running without a CSV row.",
            completed_trial_ids=completed_trial_ids,
            completed_trial_keys=completed_trial_keys,
        )


def pre_run_prompt(
    *,
    session: str,
    trial: dict[str, Any],
    total_trials: int,
    previous_initial_condition: str,
) -> tuple[str, str]:
    print("")
    print("=" * 72)
    print(f"session: {session}")
    print(f"Next trial: {trial['trial_id']} ({int(trial.get('trial_sequence', 0))}/{total_trials})")
    print(f"prompt_id: {trial['prompt_id']}")
    print(f"repeat_index: {trial['repeat_index']}")
    print("prompt_text:")
    print(str(trial["prompt_text"]))
    prompt_ic = str(trial.get("prompt_initial_condition", "")).strip()
    default_initial_condition = prompt_ic or previous_initial_condition
    print(f"setup_initial_condition: {default_initial_condition or '<none>'}")
    if default_initial_condition:
        print("setup checklist:")
        for line in describe_initial_condition(default_initial_condition):
            print(f"  - {line}")
    print(f"previous_initial_condition: {previous_initial_condition or '<none>'}")
    print(f"planned_trial_folder: {Path(trial['trial_dir']).resolve()}")
    print("")
    action_raw = input(
        "Set scene/start video, then press Enter to launch "
        "(or type override L=...;R=...;P=..., pause/quit/abort): "
    ).strip()
    initial_condition = default_initial_condition
    action = action_raw.lower()
    if action.startswith("override "):
        initial_condition = action_raw.split(" ", 1)[1].strip()
        action = ""
    return initial_condition, action


def print_dry_run(trials: list[dict[str, Any]], session_dir: Path, repeats: int) -> None:
    print("DRY RUN: no directories will be created and LLM_Commander.py will not be launched.")
    print(f"session_dir: {session_dir}")
    print(f"repeats: {repeats}")
    print(f"planned_trials: {len(trials)}")
    for trial in trials:
        print(
            f"{trial['trial_id']}: prompt_id={trial['prompt_id']} "
            f"repeat={trial['repeat_index']} "
            f"initial_condition={trial.get('prompt_initial_condition', '')} "
            f"trial_dir={trial['trial_dir']}"
        )


def print_resume_summary(
    *,
    session: str,
    session_dir: Path,
    trials: list[dict[str, Any]],
    completed_trial_keys: set[str],
    last_initial_by_prompt: dict[str, str],
) -> None:
    counts = completed_counts_by_prompt(trials, completed_trial_keys)
    next_trial = next((trial for trial in trials if trial_key(trial) not in completed_trial_keys), None)
    print("")
    print("=" * 72)
    print(f"session: {session}")
    print(f"session_folder: {session_dir}")
    print(f"trials_csv: {session_dir / 'trials.csv'}")
    print(f"failures_csv: {session_dir / 'failures.csv'}")
    print(f"summary_csv: {session_dir / 'summary.csv'}")
    print("completed_count_per_prompt:")
    for prompt_id, row in counts.items():
        print(f"  {prompt_id}: {row['completed']}/{row['total']}")
    print("last_initial_condition_per_prompt:")
    if last_initial_by_prompt:
        for prompt_id, initial_condition in sorted(last_initial_by_prompt.items()):
            print(f"  {prompt_id}: {initial_condition}")
    else:
        print("  <none>")
    print("prompt_initial_condition_defaults:")
    seen_prompt_defaults: set[str] = set()
    for trial in trials:
        prompt_id = str(trial["prompt_id"])
        if prompt_id in seen_prompt_defaults:
            continue
        seen_prompt_defaults.add(prompt_id)
        print(f"  {prompt_id}: {str(trial.get('prompt_initial_condition', '')).strip() or '<none>'}")
    if next_trial is None:
        print("next_planned_trial: <none>")
    else:
        planned = assign_available_trial_folder(session_dir, next_trial)
        print(
            "next_planned_trial: "
            f"{planned['trial_id']} prompt_id={planned['prompt_id']} "
            f"repeat={planned['repeat_index']} folder={Path(planned['trial_dir']).resolve()}"
        )
    print("=" * 72)


def session_progress_label(session_dir: Path) -> str:
    progress_path = session_dir / "progress.json"
    if progress_path.exists():
        try:
            progress = read_json(progress_path)
            completed = progress.get("completed_trials", "?") if isinstance(progress, dict) else "?"
            total = progress.get("total_trials", "?") if isinstance(progress, dict) else "?"
            updated = progress.get("last_updated_at", "") if isinstance(progress, dict) else ""
            return f"{completed}/{total} completed" + (f", updated {updated}" if updated else "")
        except SystemExit:
            pass
    trials_path = session_dir / "trials.csv"
    if trials_path.exists():
        return f"{len(read_trial_rows(trials_path))} rows in trials.csv"
    return "no progress file"


def existing_session_dirs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    sessions = [
        path
        for path in results_root.iterdir()
        if path.is_dir() and ((path / "progress.json").exists() or (path / "trials.csv").exists())
    ]
    return sorted(sessions, key=lambda path: path.stat().st_mtime, reverse=True)


def delete_session_with_confirmation(results_root: Path, session_dir: Path) -> bool:
    root_resolved = results_root.resolve()
    target_resolved = session_dir.resolve()
    if target_resolved != root_resolved and root_resolved not in target_resolved.parents:
        print(f"Refusing to delete outside results root: {target_resolved}")
        return False
    print("")
    print(f"Delete session folder: {target_resolved}")
    print("This removes trial folders, console logs, CSVs, and progress for that session.")
    confirm = input("Press Enter again to confirm delete, or type anything to cancel: ")
    if confirm.strip():
        print("Delete cancelled.")
        return False
    shutil.rmtree(target_resolved)
    print(f"Deleted session: {target_resolved}")
    return True


def choose_session_interactive(results_root: Path, default_session: str) -> tuple[str, Path]:
    while True:
        sessions = existing_session_dirs(results_root)
        if not sessions:
            return default_session, results_root / default_session
        print("")
        print("=" * 72)
        print("Existing experiment sessions:")
        for idx, session_dir in enumerate(sessions[:10], start=1):
            print(f"  {idx}. {session_dir.name} ({session_progress_label(session_dir)})")
        print("")
        choice = input(
            f"Enter session number to resume, Enter for new '{default_session}', "
            "type d<number> to delete, or type a new session name: "
        ).strip()
        if not choice:
            return default_session, results_root / default_session
        choice_lower = choice.lower()
        delete_match = re.fullmatch(r"(?:d|delete)\s*(\d+)", choice_lower)
        if delete_match:
            idx = int(delete_match.group(1))
            if 1 <= idx <= min(10, len(sessions)):
                delete_session_with_confirmation(results_root, sessions[idx - 1])
                continue
            print(f"Invalid delete number {idx}.")
            continue
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= min(10, len(sessions)):
                chosen = sessions[idx - 1]
                return chosen.name, chosen
            print(f"Invalid session number {choice}; starting new '{default_session}'.")
            return default_session, results_root / default_session
        session_name = choice
        return session_name, results_root / session_name


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default="experiments/prompts.json", help="Prompt JSON file.")
    parser.add_argument("--prompt-id", default=None, help="Run one prompt by id, index, or alias like P1.")
    parser.add_argument("--repeats", type=int, default=None, help="Repeats per prompt. Defaults to prompt file.")
    parser.add_argument("--session", default=None, help="Session name under results/.")
    parser.add_argument("--resume", default=None, help="Existing results/<session> directory to resume.")
    parser.add_argument("--results-root", default="results", help="Root directory for session output.")
    parser.add_argument("--python", dest="python_exe", default=sys.executable, help="Python executable for runtime.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned trials without launching runtime.")
    parser.add_argument("--delete-trials", default="", help="Delete trial rows/folders before resuming, e.g. 11,12,13-15.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = project_root()
    prompts_path = (root / args.prompts).resolve() if not Path(args.prompts).is_absolute() else Path(args.prompts)
    prompts, default_repeats = load_prompt_file(prompts_path)
    repeats = int(args.repeats if args.repeats is not None else default_repeats)
    if repeats <= 0:
        raise SystemExit("ERROR: --repeats must be > 0")
    if args.prompt_id:
        prompts = filter_prompts(prompts, args.prompt_id)
    elif not args.dry_run:
        prompts = choose_prompt_scope_interactive(prompts, repeats)

    if args.resume:
        session_dir = (root / args.resume).resolve() if not Path(args.resume).is_absolute() else Path(args.resume)
        session = session_dir.name
    else:
        results_root = (root / args.results_root).resolve() if not Path(args.results_root).is_absolute() else Path(args.results_root)
        if args.session or args.dry_run:
            session = str(args.session or session_stamp()).strip()
            if not session:
                raise SystemExit("ERROR: session name cannot be empty")
            session_dir = results_root / session
        else:
            session, session_dir = choose_session_interactive(results_root, session_stamp())

    trials = build_trial_plan(prompts, repeats, session_dir)
    if args.dry_run:
        print_dry_run(trials, session_dir, repeats)
        return 0

    session_dir.mkdir(parents=True, exist_ok=True)
    completed_trial_ids = read_completed_trial_ids(session_dir / "trials.csv")
    completed_trial_keys = read_completed_trial_keys(session_dir / "trials.csv")
    last_initial_by_prompt = last_initial_condition_by_prompt(session_dir / "trials.csv")
    if str(args.delete_trials).strip():
        wipe_trials_from_session(session_dir, str(args.delete_trials))
        completed_trial_ids = read_completed_trial_ids(session_dir / "trials.csv")
        completed_trial_keys = read_completed_trial_keys(session_dir / "trials.csv")
        last_initial_by_prompt = last_initial_condition_by_prompt(session_dir / "trials.csv")
    finalize_orphaned_running_trials(
        session=session,
        session_dir=session_dir,
        completed_trial_ids=completed_trial_ids,
        completed_trial_keys=completed_trial_keys,
    )
    update_progress(
        session_dir,
        session=session,
        prompts_path=prompts_path,
        repeats=repeats,
        total_trials=len(trials),
        completed_trial_ids=completed_trial_ids,
        completed_trial_keys=completed_trial_keys,
        trials=trials,
    )
    last_initial_by_prompt = last_initial_condition_by_prompt(session_dir / "trials.csv")
    print_resume_summary(
        session=session,
        session_dir=session_dir,
        trials=trials,
        completed_trial_keys=completed_trial_keys,
        last_initial_by_prompt=last_initial_by_prompt,
    )

    while True:
        planned_trial = next((trial for trial in trials if trial_key(trial) not in completed_trial_keys), None)
        if planned_trial is None:
            break
        while True:
            trial = assign_available_trial_folder(session_dir, planned_trial)
            initial_condition, action = pre_run_prompt(
                session=session,
                trial=trial,
                total_trials=len(trials),
                previous_initial_condition=last_initial_by_prompt.get(str(trial["prompt_id"]), ""),
            )
            if not action or action in {"pause", "quit", "abort"} or action.startswith(("dset", "delete_trials", "delete-trials")):
                break
            print(f"Unknown pre-run command '{action}'. Use Enter, pause, quit, abort, or dset 11,12,13-15.")
        if action in {"pause", "quit"}:
            update_progress(
                session_dir,
                session=session,
                prompts_path=prompts_path,
                repeats=repeats,
                total_trials=len(trials),
                completed_trial_ids=completed_trial_ids,
                completed_trial_keys=completed_trial_keys,
                trials=trials,
            )
            print(f"Paused. Resume with: python tools/run_prompt_trials.py --resume {session_dir}")
            return 0
        if action.startswith(("dset", "delete_trials", "delete-trials")):
            delete_spec = re.sub(r"^(?:dset|delete_trials|delete-trials)\s*[:,]?\s*", "", action).strip()
            if not delete_spec:
                print("No trial range supplied. Example: dset 11,12,13-15")
                continue
            wipe_trials_from_session(session_dir, delete_spec)
            completed_trial_ids = read_completed_trial_ids(session_dir / "trials.csv")
            completed_trial_keys = read_completed_trial_keys(session_dir / "trials.csv")
            last_initial_by_prompt = last_initial_condition_by_prompt(session_dir / "trials.csv")
            update_progress(
                session_dir,
                session=session,
                prompts_path=prompts_path,
                repeats=repeats,
                total_trials=len(trials),
                completed_trial_ids=completed_trial_ids,
                completed_trial_keys=completed_trial_keys,
                trials=trials,
            )
            print_resume_summary(
                session=session,
                session_dir=session_dir,
                trials=trials,
                completed_trial_keys=completed_trial_keys,
                last_initial_by_prompt=last_initial_by_prompt,
            )
            continue
        if action == "abort":
            abort_trial_before_run(
                session=session,
                session_dir=session_dir,
                trial=trial,
                initial_condition=initial_condition,
                completed_trial_ids=completed_trial_ids,
                completed_trial_keys=completed_trial_keys,
            )
            update_progress(
                session_dir,
                session=session,
                prompts_path=prompts_path,
                repeats=repeats,
                total_trials=len(trials),
                completed_trial_ids=completed_trial_ids,
                completed_trial_keys=completed_trial_keys,
                trials=trials,
            )
            print(f"Recorded {trial['trial_id']} as aborted before launch.")
            last_initial_by_prompt[str(trial["prompt_id"])] = str(initial_condition)
            continue

        return_code, crashed, started_at, ended_at, duration_s = launch_trial(
            python_exe=str(args.python_exe),
            root=root,
            trial=trial,
            initial_condition=initial_condition,
        )
        log_path = Path(trial["trial_dir"]) / "console_log.txt"
        metrics = parse_console_log(log_path, Path(trial["trial_dir"]))

        if crashed:
            auto_abort_crashed_trial(
                session=session,
                session_dir=session_dir,
                trial=trial,
                started_at=started_at,
                ended_at=ended_at,
                duration_s=duration_s,
                return_code=return_code,
                initial_condition=initial_condition,
                notes=f"Runtime exited unexpectedly with return_code={return_code}.",
                completed_trial_ids=completed_trial_ids,
                completed_trial_keys=completed_trial_keys,
            )
            print(
                f"Recorded {trial['trial_id']} as aborted after crash/nonzero exit. "
                "Power-cycle/reset as needed before the next trial."
            )
            last_initial_by_prompt[str(trial["prompt_id"])] = str(initial_condition)
        else:
            print("")
            print(f"Trial {trial['trial_id']} finished. Enter manual grading.")
            auto_fields = build_auto_fields(
                session_dir=session_dir,
                trial=trial,
                initial_condition=initial_condition,
                metrics=metrics,
                crashed=False,
            )
            print("auto_parse:")
            for field in (
                "total_policy_moves",
                "num_observe",
                "num_classify",
                "num_grasp",
                "num_place",
                "num_pick_other",
                "num_pick_misplaced",
                "num_recovery",
                "num_verify",
                "overcurrent_event",
                "stop_reason",
                "stop_run_llm_reason",
            ):
                print(f"  {field}: {metrics.get(field, '')}")
            for field in (
                "steps_completed_auto",
                "steps_required_auto",
                "completion_ratio_auto",
                "failure_stage_auto",
                "failure_source_auto",
                "failure_confidence",
            ):
                print(f"  {field}: {auto_fields.get(field, '')}")
            grading = collect_grading(default_final_result="success", trial=trial, auto_fields=auto_fields)
            auto_fields = adjust_auto_fields_for_manual_result(auto_fields, str(grading.get("final_result", "")), metrics)
            row = make_row(
                session=session,
                trial=trial,
                started_at=started_at,
                ended_at=ended_at,
                duration_s=duration_s,
                return_code=return_code,
                crashed=False,
                initial_condition=initial_condition,
                grading=grading,
                metrics=metrics,
                auto_fields=auto_fields,
            )
            append_trial_outputs(
                session_dir=session_dir,
                row=row,
                completed_trial_ids=completed_trial_ids,
                completed_trial_keys=completed_trial_keys,
            )
            last_initial_by_prompt[str(trial["prompt_id"])] = str(initial_condition)

        update_progress(
            session_dir,
            session=session,
            prompts_path=prompts_path,
            repeats=repeats,
            total_trials=len(trials),
            completed_trial_ids=completed_trial_ids,
            completed_trial_keys=completed_trial_keys,
            trials=trials,
        )

    print(f"All planned trials complete. Results: {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
