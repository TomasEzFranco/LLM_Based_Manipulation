#!/usr/bin/env python3
"""Planner I/O and state shaping owner module (extract-only)."""

from __future__ import annotations

import json
import math
import time

import cv2
import numpy as np

_CORE_BIND_READY = False


def _bind_core_globals() -> None:
    global _CORE_BIND_READY
    import runtime_core as core
    protected = {
        '_bind_core_globals', '_CORE_BIND_READY',
        'build_live_policy_brain', 'build_prompted_allowed_commands',
        'choose_candidate_near_uv', 'choose_track_candidate_near_uv',
        'collect_track_measurement_verify_style',
        'can_pick_misplaced_cube_now', 'can_pick_misplaced_on_side_now',
        '_get_confirmed_entry_color_for_planner',
        'get_section_confirmed_color_sequence_bottom_to_top',
        '_planner_section_row_unified', '_slots_from_level_and_sequence',
        '_remove_nearest_placed_target', 'build_prompted_step_allowed_commands',
        'maybe_append_policy_raw_row', 'build_prompted_planner_state',
    }
    for name, value in core.__dict__.items():
        if name.startswith('__') or name in protected:
            continue
        globals()[name] = value
    _CORE_BIND_READY = True

def build_live_policy_brain():
    if not _CORE_BIND_READY:
        _bind_core_globals()
    if LivePolicyBrain is None or LivePolicyConfig is None:
        detail = f" ({LIVE_POLICY_IMPORT_ERROR})" if LIVE_POLICY_IMPORT_ERROR else ""
        print(f"[Policy] llm_commander planner backend unavailable; prompted mode requires live LLM policy.{detail}")
        return None
    try:
        cfg = LivePolicyConfig(
            backend=LLM_POLICY_BACKEND,
            model_id=LLM_POLICY_MODEL,
            endpoint=LLM_POLICY_ENDPOINT,
            timeout_s=LLM_POLICY_TIMEOUT_S,
            think=LLM_POLICY_THINK,
            temperature=0.0,
            top_p=1.0,
            prompt_path=LLM_POLICY_PROMPT_PATH,
            max_reprompt=LLM_POLICY_MAX_REPROMPT,
            num_predict=int(LLM_POLICY_NUM_PREDICT),
        )
        return LivePolicyBrain(cfg)
    except Exception as exc:
        print(f"[Policy] Failed to initialize LLM policy brain: {exc}")
        return None

# ============================= Tracking / cube identity =============================
def choose_candidate_near_uv(candidates: list[dict], u: int, v: int, min_conf: float = 0.0) -> dict | None:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    best = None
    best_key = None
    for c in candidates:
        conf = float(c.get("conf", 0.0))
        if conf < min_conf:
            continue
        cu = int(c.get("u", 0))
        cv = int(c.get("v", 0))
        d2 = (cu - int(u)) ** 2 + (cv - int(v)) ** 2
        key = (d2, -conf)
        if best is None or key < best_key:
            best = c
            best_key = key
    return best


def choose_track_candidate_near_uv(
    candidates: list[dict],
    track_id: int | None,
    u: int,
    v: int,
    min_conf: float = 0.0,
) -> dict | None:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    if track_id is None:
        return None
    tid_i = int(track_id)
    tid_rows: list[dict] = []
    for c in list(candidates):
        c_tid = _candidate_track_id_or_none(c)
        if c_tid is None or int(c_tid) != int(tid_i):
            continue
        tid_rows.append(c)
    if not tid_rows:
        return None
    return choose_candidate_near_uv(tid_rows, int(u), int(v), min_conf=float(min_conf))


def collect_track_measurement_verify_style(
    *,
    state: CycleState,
    arm: Arm,
    per: Perception | None,
    det: YOLODetector | None,
    track_id: int | None,
    lock_uv: tuple[int, int] | None,
    sample_count: int | None = None,
    timeout_s: float | None = None,
    min_conf: float | None = None,
    show_window: bool = False,
    status_prefix: str = "startup_measure",
) -> dict:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    if det is None or per is None:
        return {
            "samples": int(max(1, int(sample_count if sample_count is not None else STARTUP_STACK_MEASURE_SAMPLES))),
            "hits": 0,
            "median_xyz": None,
            "resolved_track_id": (None if track_id is None else int(track_id)),
        }
    samples = int(max(1, int(sample_count if sample_count is not None else STARTUP_STACK_MEASURE_SAMPLES)))
    timeout_used = float(max(0.4, float(timeout_s if timeout_s is not None else STARTUP_STACK_MEASURE_TIMEOUT_S)))
    conf_min = float(max(0.0, float(min_conf if min_conf is not None else PLACE_VERIFY_MIN_CONF)))
    xyz_rows: list[np.ndarray] = []
    resolved_track_id = (None if track_id is None else int(track_id))
    t0 = time.time()
    while len(xyz_rows) < samples and (time.time() - t0) < timeout_used:
        obs = observe_scene_frame(
            det=det,
            arm=arm,
            per=per,
            draw=False,
            projected_min_conf=float(conf_min),
            state=state,
            update_tracks=True,
        )
        if obs is None:
            break
        candidate = None
        if resolved_track_id is not None:
            for c in list(obs.candidates):
                try:
                    c_tid = _candidate_track_id_or_none(c)
                except Exception:
                    c_tid = None
                if c_tid is None or int(c_tid) != int(resolved_track_id):
                    continue
                if float(c.get("conf", 0.0)) < float(conf_min):
                    continue
                candidate = c
                break
        if candidate is None and isinstance(lock_uv, (list, tuple)) and len(lock_uv) >= 2:
            candidate = choose_candidate_near_uv(
                obs.candidates,
                int(lock_uv[0]),
                int(lock_uv[1]),
                min_conf=float(conf_min),
            )
            if candidate is not None and resolved_track_id is None:
                c_tid = _candidate_track_id_or_none(candidate)
                if c_tid is not None:
                    resolved_track_id = int(c_tid)
        if candidate is None:
            time.sleep(max(0.0, float(arm.sample_time)))
            continue
        u = int(candidate.get("u", 0))
        v = int(candidate.get("v", 0))
        proj = _match_projected_row_by_uv(obs.projected_rows, u=u, v=v)
        xyz = None
        if proj is not None:
            arr = np.array(proj.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
            if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                xyz = [float(arr[0]), float(arr[1]), float(arr[2])]
        if xyz is None:
            fast = estimate_base_xyz_from_uv_fast(
                arm=arm,
                per=per,
                depth_frame=obs.depth_frame,
                u=u,
                v=v,
            )
            fast = np.array(fast, dtype=float).reshape(-1)
            if fast.size >= 3 and np.all(np.isfinite(fast[:3])):
                xyz = [float(fast[0]), float(fast[1]), float(fast[2])]
        if xyz is not None:
            xyz_rows.append(np.array([float(xyz[0]), float(xyz[1]), float(xyz[2])], dtype=float))
        if bool(show_window and SHOW_WINDOW):
            disp = obs.image_display
            cx_m, cy_m = obs.image_center_uv
            cv2.circle(disp, (u, v), 10, (0, 255, 0), 2)
            cv2.line(disp, (int(cx_m), int(cy_m)), (u, v), (255, 0, 255), 2)
            cv2.putText(
                disp,
                f"{status_prefix} id={resolved_track_id if resolved_track_id is not None else 'n/a'} hits={len(xyz_rows)}/{samples}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            if _show_center_frame(True, disp):
                break
        time.sleep(max(0.0, float(arm.sample_time)))
    median_xyz = None
    if xyz_rows:
        med = np.median(np.array(xyz_rows, dtype=float), axis=0)
        median_xyz = [float(med[0]), float(med[1]), float(med[2])]
    return {
        "samples": int(samples),
        "hits": int(len(xyz_rows)),
        "median_xyz": median_xyz,
        "resolved_track_id": (None if resolved_track_id is None else int(resolved_track_id)),
    }


def build_prompted_allowed_commands(
    section_groups: dict[str, list[int]],
    placed_targets: list[np.ndarray],
    blocked_slots: set[int],
    stack_levels: dict[str, int],
) -> list[str]:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    allowed: list[str] = ["stop_run"]
    left_slot = next_slot_in_section(SECTION_LEFT_NAME, section_groups, placed_targets, blocked_slots)
    right_slot = next_slot_in_section(SECTION_RIGHT_NAME, section_groups, placed_targets, blocked_slots)
    stack_allowed = (not PROMPTED_COLUMN_ONLY) and ENABLE_STACK_ACTIONS
    left_stack_started = int(stack_levels.get(SECTION_LEFT_NAME, 0)) > 0
    right_stack_started = int(stack_levels.get(SECTION_RIGHT_NAME, 0)) > 0
    if left_slot is not None and not (stack_allowed and left_stack_started):
        allowed.append("place_left")
    if right_slot is not None and not (stack_allowed and right_stack_started):
        allowed.append("place_right")
    if stack_allowed and section_groups.get(SECTION_LEFT_NAME):
        if stack_levels.get(SECTION_LEFT_NAME, 0) < max(1, int(MAX_STACK_LEVELS_PER_SECTION)):
            allowed.append("place_left_stack")
    if stack_allowed and section_groups.get(SECTION_RIGHT_NAME):
        if stack_levels.get(SECTION_RIGHT_NAME, 0) < max(1, int(MAX_STACK_LEVELS_PER_SECTION)):
            allowed.append("place_right_stack")
    return allowed

# ============================= Stack model / scene reconstruction =============================
def can_pick_misplaced_cube_now(state: CycleState) -> bool:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    return bool(can_pick_misplaced_on_side_now(state, SECTION_LEFT_NAME) or can_pick_misplaced_on_side_now(state, SECTION_RIGHT_NAME))

def can_pick_misplaced_on_side_now(
    state: CycleState,
    section_name: str,
    stack_levels: dict[str, int] | None = None,
) -> bool:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    side = str(section_name).strip().lower()
    if side not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        return False
    if bool(state.holding_object):
        return False
    if side == SECTION_LEFT_NAME and int(getattr(state, "pick_placed_empty_cooldown_left", 0)) > 0:
        return False
    if side == SECTION_RIGHT_NAME and int(getattr(state, "pick_placed_empty_cooldown_right", 0)) > 0:
        return False
    level_hint = None
    if isinstance(stack_levels, dict):
        try:
            level_hint = int(max(0, stack_levels.get(side, 0)))
        except Exception:
            level_hint = 0
    side_row = _planner_section_row_unified(state, side, stack_level_hint=level_hint)
    try:
        side_level = int(side_row.get("stack_level", 0) or 0)
    except Exception:
        side_level = 0
    if side_level <= 0:
        return False
    return True

def _get_confirmed_entry_color_for_planner(entry: dict | None) -> str | None:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    if not isinstance(entry, dict):
        return None
    verify = entry.get("verify_result", None)
    if isinstance(verify, dict):
        verify_color = str(verify.get("measured_color", "unknown")).strip().lower()
        try:
            verify_conf = float(verify.get("measured_color_conf", 0.0))
        except Exception:
            verify_conf = 0.0
        try:
            verify_hits = int(verify.get("measured_color_hits", 0))
        except Exception:
            verify_hits = 0
        if (
            verify_color in {"orange", "blue"}
            and verify_hits >= max(1, int(PLACE_VERIFY_V2_COLOR_MIN_HITS))
            and verify_conf >= float(PLACE_VERIFY_V2_COLOR_COMMIT_CONF)
        ):
            return str(verify_color)
    color_name = str(entry.get("cube_color", "")).strip().lower()
    if color_name in {"orange", "blue"}:
        return color_name
    return None

def get_section_confirmed_color_sequence_bottom_to_top(state: CycleState, section_name: str) -> list[str]:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    seq: list[str] = []
    section_norm = str(section_name).strip().lower()
    for entry in list(state.placed_ledger):
        if not _is_confirmed_active_placement(entry):
            continue
        if str(entry.get("section", "")).strip().lower() != section_norm:
            continue
        color_name = _get_confirmed_entry_color_for_planner(entry)
        if color_name in {"orange", "blue"}:
            seq.append(color_name)
    return seq

def _planner_section_row_unified(
    state: CycleState,
    section_name: str,
    *,
    stack_level_hint: int | None = None,
) -> dict:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    _ = stack_level_hint
    section_norm = str(section_name).strip().lower()
    hydrated = get_startup_hydrated_section_row(state, section_norm)
    seq_use_raw = [
        str(c).strip().lower()
        for c in list(hydrated.get("color_sequence_bottom_to_top", []))
        if str(c).strip().lower() in {"orange", "blue", "unknown"}
    ]
    try:
        hydrated_level = int(max(0, hydrated.get("stack_level", 0) or 0))
    except Exception:
        hydrated_level = int(len(seq_use_raw))
    if len(seq_use_raw) < int(hydrated_level):
        seq_use_raw.extend(["unknown"] * int(hydrated_level - len(seq_use_raw)))
    seq_use_raw = list(seq_use_raw[: max(0, int(hydrated_level))])
    stack_level = int(max(0, hydrated_level))

    stack_level, slots, seq_use = _slots_from_level_and_sequence(stack_level, seq_use_raw)
    out = {
        "stack_level": int(stack_level),
        "slots": dict(slots),
    }
    if bool(PLANNER_INCLUDE_COLOR_SEQUENCE):
        out["color_sequence_bottom_to_top"] = list(seq_use)
    return out

def _slots_from_level_and_sequence(
    stack_level_raw: object,
    seq_raw: object,
) -> tuple[int, dict[str, str], list[str]]:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    try:
        level = int(stack_level_raw)
    except Exception:
        level = 0
    level = int(max(0, min(3, level)))
    seq_norm: list[str] = []
    for c in list(seq_raw if isinstance(seq_raw, list) else []):
        color = str(c).strip().lower()
        if color in {"orange", "blue", "unknown"}:
            seq_norm.append(color)
    seq_use = list(seq_norm[:level])
    if len(seq_use) < level:
        seq_use.extend(["unknown"] * int(level - len(seq_use)))
    slots = {"base": "empty", "middle": "empty", "top": "empty"}
    if level >= 1:
        slots["base"] = str(seq_use[0])
    if level >= 2:
        slots["middle"] = str(seq_use[1])
    if level >= 3:
        slots["top"] = str(seq_use[2])
    return int(level), dict(slots), list(seq_use)


def _normalize_planner_color_name(
    color: str,
    color_conf: float = 0.0,
) -> tuple[str, float]:
    """Planner-facing color: blue, orange, or unknown (with conf 0 when unknown)."""
    color_norm = str(color).strip().lower()
    if color_norm in {"orange", "blue"}:
        try:
            conf_val = float(color_conf)
        except (TypeError, ValueError):
            conf_val = 0.0
        return str(color_norm), float(max(0.0, min(1.0, conf_val)))
    return "unknown", 0.0

def _remove_nearest_placed_target(placed_targets: list[np.ndarray], target_xyz: list[float] | None) -> bool:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    if not placed_targets or not isinstance(target_xyz, (list, tuple)) or len(target_xyz) < 3:
        return False
    arr_t = np.array([float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])], dtype=float).reshape(-1)
    if arr_t.size < 3 or not np.all(np.isfinite(arr_t[:3])):
        return False
    best_idx = None
    best_score = float("inf")
    for idx, row in enumerate(list(placed_targets)):
        try:
            arr_r = np.array(row, dtype=float).reshape(-1)
        except Exception:
            continue
        if arr_r.size < 3 or not np.all(np.isfinite(arr_r[:3])):
            continue
        d_xy = float(math.hypot(float(arr_r[0]) - float(arr_t[0]), float(arr_r[1]) - float(arr_t[1])))
        d_z = float(abs(float(arr_r[2]) - float(arr_t[2])))
        score = float(d_xy + 0.25 * d_z)
        if score < best_score:
            best_score = score
            best_idx = int(idx)
    if best_idx is None:
        return False
    try:
        placed_targets.pop(int(best_idx))
    except Exception:
        return False
    return True


def build_prompted_step_allowed_commands(
    state: CycleState,
    section_groups: dict[str, list[int]],
    stack_levels: dict[str, int],
    centered_pos: tuple[int, int] | None,
    cube_color: str,
    color_conf: float,
) -> list[str]:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    _ = (color_conf,)
    place_allowed = build_prompted_allowed_commands(
        section_groups=section_groups,
        placed_targets=state.placed_targets,
        blocked_slots=state.blocked_slots,
        stack_levels=stack_levels,
    )
    place_only = [cmd for cmd in place_allowed if cmd.startswith("place_")]
    core = ["stop_run"]
    holding = bool(state.holding_object)
    centered = bool(centered_pos is not None)
    eff_color, _eff_conf = resolve_effective_centered_color(state, cube_color, color_conf)
    color_known = str(eff_color).strip().lower() in {"orange", "blue"}
    can_return_placed_cube = False
    can_pick_misplaced_left = bool(can_pick_misplaced_on_side_now(state, SECTION_LEFT_NAME, stack_levels))
    can_pick_misplaced_right = bool(can_pick_misplaced_on_side_now(state, SECTION_RIGHT_NAME, stack_levels))
    pick_other_source = str(getattr(state, "pick_other_block_source", "none")).strip().lower()
    has_pick_other_block = (
        state.pick_other_block_track_id is not None
        or state.pick_other_block_xyz is not None
        or state.pick_other_block_uv is not None
    )
    can_pick_other = (
        (not holding)
        and bool(has_pick_other_block)
        and pick_other_source in {"classify", "return"}
    )
    # Allow placed-cube correction anytime we're not holding (including grasp phase),
    # so planner can correct stack mistakes before committing a new pick.
    allow_pick_misplaced_phase = bool(not holding)

    # Phase-tight gating to reduce invalid/loop-prone choices.
    if holding:
        if state.last_pick_return_xyz is not None:
            core.append("return_cube")
        core.extend(place_only)
    else:
        if not centered:
            core.append("observe_scene")
        elif not color_known:
            core.append("classify_cube")
        else:
            core.append("grasp_cube")
            if can_pick_other:
                core.append("pick_other")
        if centered:
            core.append("push_cube")
        if allow_pick_misplaced_phase and can_pick_misplaced_left:
            core.append("pick_placed_left")
        if allow_pick_misplaced_phase and can_pick_misplaced_right:
            core.append("pick_placed_right")
    blocked_place_sections: set[str] = set()
    if (not holding) and state.placed_ledger:
        latest_verify = state.placed_ledger[-1].get("verify_result", None)
        if latest_verify is None or not bool(latest_verify.get("confirmed", False)):
            core.append("verify_last_place")
            if STACK_VERIFY_CORRECTION_ENABLED and STACK_VERIFY_BLOCK_STACK_ON_UNCONFIRMED:
                latest_section = str(state.placed_ledger[-1].get("section", "")).strip().lower()
                if latest_section in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
                    blocked_place_sections.add(str(latest_section))
    if SECTION_LEFT_NAME in blocked_place_sections:
        core = [cmd for cmd in core if cmd not in {"place_left", "place_left_stack"}]
    if SECTION_RIGHT_NAME in blocked_place_sections:
        core = [cmd for cmd in core if cmd not in {"place_right", "place_right_stack"}]
    # Keep order stable but dedupe in case multiple gates add same command.
    deduped: list[str] = []
    for cmd in core:
        if cmd not in deduped:
            deduped.append(cmd)
    return deduped


# ============================= Runtime utilities =============================

def maybe_append_policy_raw_row(
    state: CycleState,
    *,
    cycle: int,
    step_index: int,
    phase: str,
    llm_input: dict,
    raw_output: str,
    prompt_path: str,
    prompt_text: str,
    normalization_reason: str = "",
    normalized_from: str = "",
):
    if not _CORE_BIND_READY:
        _bind_core_globals()
    if not POLICY_TRACE_SAVE_RAW:
        return
    try:
        llm_input_frozen = json.loads(json.dumps(llm_input, ensure_ascii=True))
    except Exception:
        llm_input_frozen = {"_error": "failed_to_serialize_llm_input"}
    row = {
        "timestamp_ms": int(time.time() * 1000),
        "cycle": int(cycle),
        "step_index": int(step_index),
        "phase": str(phase),
        "trace_schema": "policy_io_v1",
        "llm_config": {
            "backend": str(LLM_POLICY_BACKEND),
            "model": str(LLM_POLICY_MODEL),
            "endpoint": str(LLM_POLICY_ENDPOINT),
            "timeout_s": float(LLM_POLICY_TIMEOUT_S),
            "think": bool(LLM_POLICY_THINK),
        },
        "prompt_path": str(prompt_path),
        "prompt_text": str(prompt_text or ""),
        "llm_input": llm_input_frozen,
        "raw_output": str(raw_output or ""),
    }
    norm_reason = str(normalization_reason or "").strip()
    norm_from = str(normalized_from or "").strip()
    if norm_reason:
        row["normalization"] = {
            "reason_code": norm_reason,
            "from": norm_from,
            "to": "observe_scene",
        }
    state.raw_policy_rows.append(row)


def resolve_effective_centered_color(
    state: CycleState,
    cube_color: str,
    color_conf: float,
) -> tuple[str, float]:
    """Cube color for planner/gating: live classify first, else pregrasp pick lock."""
    color_norm, conf_norm = _normalize_planner_color_name(cube_color, color_conf)
    if color_norm in {"orange", "blue"}:
        return str(color_norm), float(conf_norm)
    lock_color = str(getattr(state, "pregrasp_pick_lock_color", "unknown")).strip().lower()
    if lock_color in {"orange", "blue"}:
        try:
            lock_conf = float(getattr(state, "pregrasp_pick_lock_color_conf", 0.0))
        except (TypeError, ValueError):
            lock_conf = 0.0
        return str(lock_color), float(max(0.0, min(1.0, lock_conf)))
    return "unknown", 0.0


def _resolve_held_cube_color_for_planner(
    state: CycleState,
    cube_color: str,
    color_conf: float,
) -> tuple[str, float]:
    return resolve_effective_centered_color(state, cube_color, color_conf)


def build_prompted_planner_state(
    *,
    state: CycleState,
    phase_name: str,
    holding_object: bool,
    cube_color: str,
    color_conf: float,
    centered_pos: tuple[int, int] | None,
    stack_levels: dict[str, int],
    picked_count: int,
    placed_count: int,
    scene_empty_confirmed: bool,
    last_feedback: dict | None,
) -> dict:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    def _section_status_for_planner(section_name: str) -> dict:
        try:
            level_hint = int(max(0, stack_levels.get(str(section_name).strip().lower(), 0)))
        except Exception:
            level_hint = 0
        internal_row = _planner_section_row_unified(state, section_name, stack_level_hint=level_hint)
        try:
            cube_count = int(max(0, internal_row.get("stack_level", 0) or 0))
        except Exception:
            cube_count = 0
        out = {
            "cube_count": int(cube_count),
            "slots": dict(internal_row.get("slots", {})),
        }
        if "color_sequence_bottom_to_top" in internal_row:
            out["color_sequence_bottom_to_top"] = list(internal_row.get("color_sequence_bottom_to_top", []))
        return out

    _ = (picked_count, placed_count, last_feedback)  # runtime-only tracking, not planner payload
    holding = bool(holding_object)
    if holding:
        held_color, held_conf = _resolve_held_cube_color_for_planner(state, cube_color, color_conf)
        held_cube = {"color": str(held_color), "conf": float(held_conf)}
        pick_target = None
    else:
        held_cube = None
        pick_color, pick_conf = resolve_effective_centered_color(state, cube_color, color_conf)
        pick_target = {
            "color": str(pick_color),
            "conf": float(pick_conf),
            "is_centered": bool(centered_pos is not None),
        }
    return {
        "mission_prompt": MISSION_PROMPT,
        "phase": str(phase_name),
        "holding_object": holding,
        "held_cube": held_cube,
        "pick_target": pick_target,
        "section_status": {
            SECTION_LEFT_NAME: _section_status_for_planner(SECTION_LEFT_NAME),
            SECTION_RIGHT_NAME: _section_status_for_planner(SECTION_RIGHT_NAME),
        },
        "scene_empty_confirmed": bool(scene_empty_confirmed),
    }



__all__ = [
    "build_live_policy_brain",
    "build_prompted_allowed_commands",
    "build_prompted_step_allowed_commands",
    "build_prompted_planner_state",
    "resolve_effective_centered_color",
    "maybe_append_policy_raw_row",
]
