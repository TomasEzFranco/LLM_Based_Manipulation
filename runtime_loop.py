"""Runtime loop orchestration (M1 extraction).

This module intentionally keeps behavior aligned with `runtime_core` while moving
the prompted control-loop definition into a dedicated file for readability.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from runtime_core import (
    Arm,
    CENTER_EY_I_CLAMP,
    CENTER_EY_I_DECAY,
    CENTER_EY_I_ENABLE_ABS_PX,
    CENTER_EY_I_RANGE_GATE_ENABLED,
    CENTER_EY_KI,
    CENTER_TIMEOUT_ACTIVE_S,
    CENTER_TIMEOUT_NO_DETECTION_S,
    COMMIT_CONF,
    CycleState,
    DEFAULT_RESULTS_ROOT,
    DETECT_CONF,
    EMPTY_SCENE_CONFIRM_PASSES,
    HOME,
    KELBOW,
    KSHOULDER,
    KYAW,
    LLM_POLICY_ENDPOINT,
    LLM_POLICY_PROMPT_PATH,
    MAX_CYCLES_WITHOUT_PLACE_PROGRESS,
    MAX_STACK_LEVELS_PER_SECTION,
    PICK_CORRECTION_FAIL_HYDRATE_REFRESH_ENABLED,
    PICK_OTHER_HARD_TIMEOUT_S,
    PICK_OTHER_MAX_REJECTS,
    PICK_OTHER_VALIDATE_SAMPLES,
    PICK_PLACED_EMPTY_COOLDOWN_STEPS,
    PLACE_FAIL_CONTINUE_REASONS,
    PLACE_RELEASE_OPEN_GRIP,
    PLACE_VERIFY_V2_MIN_HITS,
    POLICY_LOG_ALLOWED_COMMANDS,
    POLICY_PRINT_RAW,
    POLICY_PRINT_RAW_BLOCK,
    POLICY_TRACE_SAVE_RAW,
    PROMPTED_SAFE_PICK_REACH_M,
    PUSH_MIN_PROGRESS_M,
    PX_TOL,
    Perception,
    REALSENSE_COLOR_HEIGHT,
    REALSENSE_COLOR_WIDTH,
    REALSENSE_FPS,
    RUNTIME_MODE_DEFAULT,
    SECTION_LEFT_NAME,
    SECTION_RIGHT_NAME,
    STACK_VERIFY_ALLOW_DOWNWARD_CORRECTION,
    STACK_VERIFY_CORRECTION_ENABLED,
    STACK_VERIFY_DOWNWARD_REQUIRE_STABLE_REMEASURE,
    STACK_VERIFY_REQUIRE_CONFIRMED_FOR_ADVANCE,
    STARTUP_VERIFY_SHOW_WINDOW,
    TARGET_CLASSES,
    TRACK_ENABLE,
    TUNE_MAX_DELTA_FROM_BASELINE_M,
    TUNE_RUNS_ROOT,
    YOLODetector,
    YOLO_CONF,
    YOLO_MODEL_PATH,
    YOLO_TRACK_PERSIST,
    choose_track_candidate_near_uv,
    clamp_grip_cmd,
    classify_cube_color_patch,
    estimate_base_xyz_from_uv_fast,
    execute_push_cube_action,
    finalize_run_home,
    log_cycle_header,
    log_startup_config,
    log_stop_reason,
    log_summary,
    nearest_visible_track_by_uv,
    shutdown_runtime,
)
import runtime_core as core
import pick_actions
import misplaced_actions
import place_actions
import stack_scene
import projection_geometry
import planner_io
import runtime_loop_startup
import runtime_loop_policy
import runtime_loop_observe
import runtime_loop_dispatch
import runtime_loop_cycle
import runtime_loop_actions_router

RUNTIME_VERSION_STAMP = "2026-05-05.prompted-stability-v1"
OBSERVE_FAIL_STOP_AFTER = max(1, int(os.getenv("QARM_OBSERVE_FAIL_STOP_AFTER", "2")))


def main_prompted():
    runtime_loop_file = str(Path(__file__).resolve())
    runtime_core_file = str(Path(getattr(core, "__file__", "runtime_core.py")).resolve())
    entrypoint_file = str((Path(runtime_loop_file).parent / "LLM_Commander.py").resolve())
    print(
        "[RuntimeSignature] runtime_loop.main_prompted startup_bootstrap_path=v2 "
        f"version={RUNTIME_VERSION_STAMP}"
    )
    print(
        f"[RuntimePaths] entry={entrypoint_file} | runtime_loop={runtime_loop_file} | "
        f"runtime_core={runtime_core_file}"
    )
    if Path(runtime_loop_file).parent != Path(runtime_core_file).parent:
        print(
            "[RuntimeSignatureWarning] runtime_loop and runtime_core are loaded from different "
            "directories. This can indicate a stale copy or mixed deployment."
        )
    core.maybe_apply_calibration_profile_from_env(log_prefix="[CalibrationProfile]")
    print("=" * 60)
    print("QArm Prompted Operator Prototype")
    print("=" * 60)
    print(f"[GripTune] prompted_safe_pick_reach={PROMPTED_SAFE_PICK_REACH_M:.3f} m")
    print(
        f"[VisionTune] center_timeout(no_det/active)="
        f"{CENTER_TIMEOUT_NO_DETECTION_S:.1f}s/{CENTER_TIMEOUT_ACTIVE_S:.1f}s | "
        f"detect_conf={DETECT_CONF:.2f} | commit_conf={COMMIT_CONF:.2f}"
    )
    print(
        f"[CenterCtrl] px_tol={PX_TOL} | "
        f"gains(yaw/sh/el)=({KYAW:.5f}/{KSHOULDER:.5f}/{KELBOW:.5f}) | "
        f"ey_ki={CENTER_EY_KI:.6f} i_clamp={CENTER_EY_I_CLAMP:.1f} i_decay={CENTER_EY_I_DECAY:.2f} "
        f"i_range_gate={bool(CENTER_EY_I_RANGE_GATE_ENABLED)} "
        f"(<= {CENTER_EY_I_ENABLE_ABS_PX:.0f}px when on)"
    )
    arm = Arm()
    per = Perception(
        width=int(REALSENSE_COLOR_WIDTH),
        height=int(REALSENSE_COLOR_HEIGHT),
        fps=int(REALSENSE_FPS),
    )
    det = YOLODetector(
        YOLO_MODEL_PATH,
        target_classes=TARGET_CLASSES,
        conf=YOLO_CONF,
        track_persist=YOLO_TRACK_PERSIST,
    )
    policy_brain = planner_io.build_live_policy_brain()
    policy_prompt_text = "" if policy_brain is None else str(getattr(policy_brain, "prompt_text", ""))
    state = CycleState()
    trace_root = Path(os.getenv("QARM_POLICY_TRACE_DIR", str(DEFAULT_RESULTS_ROOT / "live_policy_runs")))
    trace_root.mkdir(parents=True, exist_ok=True)
    run_ts = int(time.time())
    raw_trace_path = trace_root / f"policy_raw_{run_ts}.jsonl"

    def run_startup_stack_bootstrap_verify(*, mode: str = "full") -> dict:
        return runtime_loop_startup.run_startup_stack_bootstrap_verify(
            state=state,
            arm=arm,
            per=per,
            det=det,
            mode=mode,
        )
    try:
        slots = stack_scene.get_place_slots()
        section_groups = stack_scene.section_slot_groups(slots)
        stack_levels = {SECTION_LEFT_NAME: 0, SECTION_RIGHT_NAME: 0}

        def sync_stack_levels_from_startup_bootstrap(startup_boot_row: dict | None) -> None:
            runtime_loop_startup.sync_stack_levels_from_startup_bootstrap(
                state=state,
                stack_levels=stack_levels,
                startup_boot_row=startup_boot_row,
            )

        def sync_stack_levels_from_authoritative_state() -> None:
            runtime_loop_startup.sync_stack_levels_from_authoritative_state(
                state=state,
                stack_levels=stack_levels,
            )

        log_startup_config(mode="prompted", safe_slots=slots, section_groups=section_groups)
        prompt_path_obj = Path(LLM_POLICY_PROMPT_PATH).resolve()
        print(
            f"[StartupConfig] version={RUNTIME_VERSION_STAMP} "
            f"prompt_file={prompt_path_obj}"
        )
        print(f"[StartupConfig] startup_verify_show_window={bool(STARTUP_VERIFY_SHOW_WINDOW)}")
        if not prompt_path_obj.exists():
            print(f"[StartupConfigWarning] prompt_file_missing={prompt_path_obj}")
        print("[StartupPhase] 1/3 -> HOME")
        arm.goto_task_space(HOME, duration=1.5, label="startup_home_prompted")
        print("[StartupPhase] 2/3 -> startup bootstrap verify")
        startup_boot = run_startup_stack_bootstrap_verify()

        def _startup_expected_shortfall_sides(startup_row: dict | None) -> list[str]:
            if not isinstance(startup_row, dict):
                return []
            return [
                str(side).strip().lower()
                for side in list(startup_row.get("hydration_expected_shortfall_sides", []))
                if str(side).strip().lower() in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}
            ]

        def _startup_hydrated_level(startup_row: dict | None, side_name: str) -> int:
            if not isinstance(startup_row, dict):
                return 0
            side_norm = str(side_name).strip().lower()
            hydrated = startup_row.get("hydrated_stacks", {})
            if not isinstance(hydrated, dict):
                hydrated = {}
            levels = hydrated.get("observed_stack_levels", {})
            if isinstance(levels, dict):
                try:
                    return int(max(0, levels.get(side_norm, 0) or 0))
                except Exception:
                    pass
            sections = hydrated.get("sections", {})
            if isinstance(sections, dict):
                row = sections.get(side_norm, {})
                if isinstance(row, dict):
                    try:
                        return int(max(0, row.get("stack_level", 0) or 0))
                    except Exception:
                        pass
            return 0

        def _startup_shortfall_pick_placed_ready_sides(startup_row: dict | None) -> list[str]:
            return [
                str(side)
                for side in _startup_expected_shortfall_sides(startup_row)
                if _startup_hydrated_level(startup_row, str(side)) > 0
            ]

        def _startup_blocking_shortfall_sides(startup_row: dict | None) -> list[str]:
            correction_ready = set(_startup_shortfall_pick_placed_ready_sides(startup_row))
            return [
                str(side)
                for side in _startup_expected_shortfall_sides(startup_row)
                if str(side) not in correction_ready
            ]

        startup_pass = 1
        while (
            _startup_blocking_shortfall_sides(startup_boot)
            and int(startup_pass) < int(core.STARTUP_STACK_BOOTSTRAP_MAX_PASSES)
        ):
            print(
                f"[StartupHydrateRetry] pass={int(startup_pass) + 1}/"
                f"{int(core.STARTUP_STACK_BOOTSTRAP_MAX_PASSES)} "
                f"reason=expected_level_shortfall "
                f"sides={_startup_blocking_shortfall_sides(startup_boot)}"
            )
            startup_boot = run_startup_stack_bootstrap_verify()
            startup_pass += 1

        sync_stack_levels_from_startup_bootstrap(startup_boot)
        startup_pick_placed_ready_shortfall = _startup_shortfall_pick_placed_ready_sides(startup_boot)
        if startup_pick_placed_ready_shortfall:
            levels_ready = {
                str(side): int(_startup_hydrated_level(startup_boot, str(side)))
                for side in startup_pick_placed_ready_shortfall
            }
            print(
                "[StartupHydrateContinue] expected_layer_shortfall "
                f"sides={startup_pick_placed_ready_shortfall} hydrated_levels={levels_ready} "
                "reason=pick_placed_can_correct"
            )
        print(
            "[StartupPhase] 3/3 -> policy ready "
            f"(status={startup_boot.get('status')}, fresh={bool(startup_boot.get('fresh', False))}, "
            f"ready_for_stacking={bool(startup_boot.get('ready_for_stacking', False))})"
        )
        startup_blocking_shortfall = _startup_blocking_shortfall_sides(startup_boot)
        if startup_blocking_shortfall:
            state.stop_reason = (
                "startup_hydrate_expected_layers_incomplete:"
                + ",".join(startup_blocking_shortfall)
            )
            log_stop_reason(state.stop_reason)
        while True:
            if state.stop_reason != "completed":
                break
            state.cycle_count += 1
            if state.cycles_without_place_progress >= MAX_CYCLES_WITHOUT_PLACE_PROGRESS:
                state.stop_reason = (
                    f"no placement progress for {state.cycles_without_place_progress} cycles "
                    f"(limit {MAX_CYCLES_WITHOUT_PLACE_PROGRESS})"
                )
                log_stop_reason(state.stop_reason)
                break
            log_cycle_header(state.cycle_count, "Prompted Cycle")
            centered_pos: tuple[int, int] | None = None
            cube_color = "unknown"
            color_conf = 0.0
            hold_grip = 0.0
            carry_supervisor: MotionGripSupervisor | None = None
            step_index = 0
            last_feedback: dict | None = None
            reobserve_streak = 0
            pick_other_streak = 0
            observe_fail_streak = 0

            def record_policy_step(
                command: str,
                result: str,
                progress: bool,
                feedback_context: dict | None = None,
            ):
                nonlocal reobserve_streak, last_feedback
                reobserve_streak, last_feedback = runtime_loop_policy.record_policy_step(
                    state=state,
                    reobserve_streak=int(reobserve_streak),
                    last_feedback=last_feedback,
                    command=command,
                    result=result,
                    progress=bool(progress),
                    feedback_context=feedback_context,
                )

            def capture_pick_lock_snapshot(source: str) -> None:
                runtime_loop_observe.capture_pick_lock_snapshot(
                    state=state,
                    centered_pos=centered_pos,
                    cube_color=str(cube_color),
                    color_conf=float(color_conf),
                    arm=arm,
                    source=str(source),
                )

            def run_post_lift_place_space_refresh(source_tag: str = "post_lift_place_space_refresh") -> dict | None:
                return runtime_loop_observe.run_post_lift_place_space_refresh(
                    source_tag=str(source_tag),
                    state=state,
                    arm=arm,
                    per=per,
                    det=det,
                )

            def log_ledger_stack_snapshot(source: str) -> None:
                runtime_loop_policy.log_ledger_stack_snapshot(
                    state=state,
                    source=str(source),
                    section_left_name=SECTION_LEFT_NAME,
                    section_right_name=SECTION_RIGHT_NAME,
                )

            def log_planner_stack_view(source: str, planner_state_row: dict | None) -> None:
                runtime_loop_policy.log_planner_stack_view(
                    source=str(source),
                    planner_state_row=planner_state_row,
                    section_left_name=SECTION_LEFT_NAME,
                    section_right_name=SECTION_RIGHT_NAME,
                )

            def run_observe_action(command_for_history: str, clear_first: bool = False, source: str = "policy_observe"):
                nonlocal centered_pos, cube_color, color_conf, observe_fail_streak
                centered_pos, cube_color, color_conf, observe_fail_streak = runtime_loop_observe.run_observe_action(
                    command_for_history=str(command_for_history),
                    clear_first=bool(clear_first),
                    source=str(source),
                    state=state,
                    arm=arm,
                    per=per,
                    det=det,
                    section_groups=section_groups,
                    cycle_count=int(state.cycle_count),
                    centered_pos=centered_pos,
                    cube_color=str(cube_color),
                    color_conf=float(color_conf),
                    observe_fail_streak=int(observe_fail_streak),
                    observe_fail_stop_after=int(OBSERVE_FAIL_STOP_AFTER),
                    record_policy_step=record_policy_step,
                    capture_pick_lock_snapshot_fn=capture_pick_lock_snapshot,
                )

            def run_pick_other_action(command_for_history: str):
                nonlocal centered_pos, cube_color, color_conf
                centered_pos, cube_color, color_conf = runtime_loop_observe.run_pick_other_action(
                    command_for_history=str(command_for_history),
                    state=state,
                    arm=arm,
                    per=per,
                    det=det,
                    centered_pos=centered_pos,
                    cube_color=str(cube_color),
                    color_conf=float(color_conf),
                    place_verify_v2_min_hits=max(1, int(PLACE_VERIFY_V2_MIN_HITS)),
                    pick_other_validate_samples=int(PICK_OTHER_VALIDATE_SAMPLES),
                    record_policy_step=record_policy_step,
                    capture_pick_lock_snapshot_fn=capture_pick_lock_snapshot,
                )

            while True:
                step_index += 1
                cycle_row = runtime_loop_cycle.compute_phase_and_allowed_commands(
                    state=state,
                    step_index=int(step_index),
                    section_groups=section_groups,
                    stack_levels=stack_levels,
                    centered_pos=centered_pos,
                    cube_color=cube_color,
                    color_conf=color_conf,
                    observe_fail_streak=int(observe_fail_streak),
                    observe_fail_stop_after=int(OBSERVE_FAIL_STOP_AFTER),
                    empty_scene_confirm_passes=int(EMPTY_SCENE_CONFIRM_PASSES),
                    max_stack_levels_per_section=int(MAX_STACK_LEVELS_PER_SECTION),
                    section_left_name=SECTION_LEFT_NAME,
                    section_right_name=SECTION_RIGHT_NAME,
                    planner_io_module=planner_io,
                    policy_log_allowed_commands=bool(POLICY_LOG_ALLOWED_COMMANDS),
                )
                phase_name = str(cycle_row.get("phase_name", "observe"))
                allowed_commands = list(cycle_row.get("allowed_commands", ["stop_run"]))
                scene_empty_confirmed = bool(cycle_row.get("scene_empty_confirmed", False))
                _empty_scene_progress = str(cycle_row.get("empty_scene_progress", "0/1"))
                _removed_by_sanity = list(cycle_row.get("removed_by_sanity", []))
                planner_state = planner_io.build_prompted_planner_state(
                    state=state,
                    phase_name=phase_name,
                    holding_object=state.holding_object,
                    cube_color=cube_color,
                    color_conf=float(color_conf),
                    centered_pos=centered_pos,
                    stack_levels=stack_levels,
                    picked_count=state.picked_count,
                    placed_count=state.placed_count,
                    scene_empty_confirmed=scene_empty_confirmed,
                    last_feedback=last_feedback,
                )
                if POLICY_LOG_ALLOWED_COMMANDS:
                    log_planner_stack_view("policy_input", planner_state)
                decision = None
                if policy_brain is None:
                    action_cmd = "stop_run"
                    state.stop_reason = "policy backend unavailable in prompted mode"
                    log_stop_reason(state.stop_reason)
                    record_policy_step(action_cmd, "backend_unavailable", progress=False)
                    break
                else:
                    llm_input_payload = {
                        "state": planner_state,
                        "allowed_commands": list(allowed_commands),
                    }
                    decision = policy_brain.decide(state=planner_state, allowed_commands=allowed_commands)
                    action_cmd = str(decision.command)
                    planner_io.maybe_append_policy_raw_row(
                        state,
                        cycle=state.cycle_count,
                        step_index=step_index,
                        phase=phase_name,
                        llm_input=llm_input_payload,
                        raw_output=str(decision.raw_output or ""),
                        prompt_path=str(LLM_POLICY_PROMPT_PATH),
                        prompt_text=policy_prompt_text,
                        normalization_reason=str(getattr(decision, "normalized_reason", "") or ""),
                        normalized_from=str(getattr(decision, "normalized_from", "") or ""),
                    )
                    if POLICY_PRINT_RAW:
                        raw_text = str(decision.raw_output or "")
                        if POLICY_PRINT_RAW_BLOCK:
                            print(
                                f"[PolicyRawBegin] cycle={state.cycle_count} "
                                f"step={step_index} phase={phase_name}"
                            )
                            print(raw_text if raw_text.strip() else "<empty>")
                            print("[PolicyRawEnd]")
                        else:
                            print(f"[PolicyRaw] {raw_text.strip() if raw_text.strip() else '<empty>'}")
                    if decision.valid:
                        print(
                            f"[Policy] step={step_index} phase={phase_name} command={action_cmd} "
                            f"conf={decision.confidence} latency={decision.latency_ms:.1f}ms reason={decision.reason}"
                        )
                        if str(getattr(decision, "normalized_reason", "")).strip():
                            print(
                                f"[PolicyNormalize] reason_code={str(decision.normalized_reason)} "
                                f"from={str(getattr(decision, 'normalized_from', ''))} to={action_cmd}"
                            )
                    else:
                        invalid_error = str(decision.error or "invalid_output")
                        state.policy_invalid_count += 1
                        state.stop_reason = f"policy_invalid:{invalid_error}"
                        print(f"[Policy] invalid ({invalid_error}); stopping (fallback disabled).")
                        if "10061" in invalid_error:
                            print(
                                f"[Policy] Ollama endpoint unreachable at {LLM_POLICY_ENDPOINT}. "
                                "Ensure Ollama is running before launch."
                            )
                        record_policy_step(action_cmd, f"policy_invalid:{invalid_error}", progress=False)
                        break
                dispatch_row = runtime_loop_dispatch.handle_pre_action_dispatch(
                    action_cmd=str(action_cmd),
                    state=state,
                    pick_other_streak=int(pick_other_streak),
                    log_stop_reason=log_stop_reason,
                    record_policy_step=record_policy_step,
                    run_observe_action=run_observe_action,
                    run_pick_other_action=run_pick_other_action,
                )
                pick_other_streak = int(dispatch_row.get("pick_other_streak", pick_other_streak))
                if bool(dispatch_row.get("handled", False)):
                    if bool(dispatch_row.get("break_loop", False)):
                        break
                    continue
                action_row = runtime_loop_actions_router.dispatch_post_pre_actions(
                    action_cmd=str(action_cmd),
                    state=state,
                    arm=arm,
                    det=det,
                    per=per,
                    stack_levels=stack_levels,
                    section_groups=section_groups,
                    centered_pos=centered_pos,
                    cube_color=str(cube_color),
                    color_conf=float(color_conf),
                    hold_grip=float(hold_grip),
                    carry_supervisor=carry_supervisor,
                    section_left_name=SECTION_LEFT_NAME,
                    section_right_name=SECTION_RIGHT_NAME,
                    prompted_safe_pick_reach_m=float(PROMPTED_SAFE_PICK_REACH_M),
                    pick_placed_empty_cooldown_steps=int(PICK_PLACED_EMPTY_COOLDOWN_STEPS),
                    pick_correction_fail_hydrate_refresh_enabled=bool(PICK_CORRECTION_FAIL_HYDRATE_REFRESH_ENABLED),
                    place_release_open_grip=float(PLACE_RELEASE_OPEN_GRIP),
                    place_fail_continue_reasons=tuple(PLACE_FAIL_CONTINUE_REASONS),
                    stack_verify_correction_enabled=bool(STACK_VERIFY_CORRECTION_ENABLED),
                    stack_verify_require_confirmed_for_advance=bool(STACK_VERIFY_REQUIRE_CONFIRMED_FOR_ADVANCE),
                    stack_verify_allow_downward_correction=bool(STACK_VERIFY_ALLOW_DOWNWARD_CORRECTION),
                    stack_verify_downward_require_stable_remeasure=bool(STACK_VERIFY_DOWNWARD_REQUIRE_STABLE_REMEASURE),
                    track_enable=bool(TRACK_ENABLE),
                    push_min_progress_m=float(PUSH_MIN_PROGRESS_M),
                    cycle_count=int(state.cycle_count),
                    home_pose=HOME,
                    nearest_visible_track_by_uv_fn=nearest_visible_track_by_uv,
                    choose_track_candidate_near_uv_fn=choose_track_candidate_near_uv,
                    classify_cube_color_patch_fn=classify_cube_color_patch,
                    estimate_base_xyz_from_uv_fast_fn=estimate_base_xyz_from_uv_fast,
                    execute_push_cube_action_fn=execute_push_cube_action,
                    finite_xyz_or_none_fn=core._finite_xyz_or_none,
                    clamp_grip_cmd_fn=clamp_grip_cmd,
                    sync_stack_levels_from_authoritative_state=sync_stack_levels_from_authoritative_state,
                    run_startup_stack_bootstrap_verify=run_startup_stack_bootstrap_verify,
                    sync_stack_levels_from_startup_bootstrap=sync_stack_levels_from_startup_bootstrap,
                    log_ledger_stack_snapshot=log_ledger_stack_snapshot,
                    run_post_lift_place_space_refresh=run_post_lift_place_space_refresh,
                    record_policy_step=record_policy_step,
                    run_observe_action=run_observe_action,
                    capture_pick_lock_snapshot_fn=capture_pick_lock_snapshot,
                )
                if bool(action_row.get("handled", False)):
                    if "centered_pos" in action_row:
                        centered_pos = action_row.get("centered_pos", centered_pos)
                    if "cube_color" in action_row:
                        cube_color = str(action_row.get("cube_color", cube_color))
                    if "color_conf" in action_row:
                        color_conf = float(action_row.get("color_conf", color_conf))
                    if "hold_grip" in action_row:
                        hold_grip = float(action_row.get("hold_grip", hold_grip))
                    if "carry_supervisor" in action_row:
                        carry_supervisor = action_row.get("carry_supervisor", carry_supervisor)
                    if bool(action_row.get("break_loop", False)):
                        break
                    continue
                # TODO: Consider centralizing repeated precondition recovery only after behavior validation.
                state.cycles_without_place_progress += 1
                state.invalid_precondition_recoveries += 1
                print(f"[Policy] unsupported command '{action_cmd}'; auto-observing.")
                record_policy_step(action_cmd, "unsupported_command", progress=False)
                runtime_loop_dispatch.run_auto_recovery_observe(run_observe_action=run_observe_action)
            if state.stop_reason != "completed":
                break
        finalize_run_home(arm, state, final_label="final_home_prompted")
        if POLICY_TRACE_SAVE_RAW:
            with raw_trace_path.open("w", encoding="utf-8") as fp:
                for row in state.raw_policy_rows:
                    fp.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"[Policy] wrote raw trace: {raw_trace_path}")
        log_summary("prompted", state, arm)
    except KeyboardInterrupt:
        print("\n\nWARNING: Interrupted by user")
    finally:
        shutdown_runtime(per, arm)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_xyz3(value: object) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) < 3:
        return None
    try:
        arr = np.array([float(value[0]), float(value[1]), float(value[2])], dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), float(value))))


def _write_jsonl_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, sort_keys=True) + "\n")


def main_tune():
    core.maybe_apply_calibration_profile_from_env(log_prefix="[CalibrationProfile]")
    print("=" * 60)
    print("QArm cam_off_y Sweep Tune Mode")
    print("=" * 60)
    trials_per_candidate = max(1, int(os.getenv("QARM_TUNE_Y_TRIALS_PER_CAND", "3")))
    fail_penalty_m = max(0.0, float(os.getenv("QARM_TUNE_Y_FAIL_PENALTY_M", "0.030")))
    coarse_offsets_raw = os.getenv("QARM_TUNE_Y_COARSE_OFFSETS", "-0.002,-0.001,0.000,0.001,0.002").strip()
    fine_half_span = max(0.0001, float(os.getenv("QARM_TUNE_Y_FINE_HALF_SPAN_M", "0.0010")))
    fine_step = max(0.0001, float(os.getenv("QARM_TUNE_Y_FINE_STEP_M", "0.0005")))
    y_max_delta = float(
        os.getenv(
            "QARM_TUNE_Y_MAX_DELTA_FROM_BASELINE_M",
            f"{float(TUNE_MAX_DELTA_FROM_BASELINE_M):.4f}",
        )
    )
    print(
        f"[TuneConfig] trials_per_candidate={trials_per_candidate} fail_penalty_m={fail_penalty_m:.4f} "
        f"fine_span={fine_half_span:.4f} fine_step={fine_step:.4f} max_delta={y_max_delta:.4f}"
    )

    arm = Arm()
    per = Perception(
        width=int(REALSENSE_COLOR_WIDTH),
        height=int(REALSENSE_COLOR_HEIGHT),
        fps=int(REALSENSE_FPS),
    )
    det = YOLODetector(
        YOLO_MODEL_PATH,
        target_classes=TARGET_CLASSES,
        conf=YOLO_CONF,
        track_persist=YOLO_TRACK_PERSIST,
    )
    state = CycleState()
    run_ts = int(time.time())
    tune_root = Path(TUNE_RUNS_ROOT)
    tune_root.mkdir(parents=True, exist_ok=True)
    tune_log_path = tune_root / f"tune_run_{run_ts}.jsonl"
    profiles_dir = core.ensure_tune_profiles_dir()

    baseline_offsets = projection_geometry.get_cam_offsets()
    baseline_grasp_z_pick_fraction = float(core.get_grasp_z_pick_fraction())
    baseline_grip_params = core.get_grip_tune_params()
    baseline_snapshot = {
        "profile_kind": "baseline_snapshot",
        "tune_mode": "cam_off_y_sweep",
        "frozen_at_utc": _utc_now_iso(),
        "run_id": int(run_ts),
        "cam_off_x_m": float(baseline_offsets["cam_off_x_m"]),
        "cam_off_y_m": float(baseline_offsets["cam_off_y_m"]),
        "cam_off_z_m": float(baseline_offsets["cam_off_z_m"]),
        "grasp_z_pick_fraction": float(baseline_grasp_z_pick_fraction),
        "grip_detect_a": float(baseline_grip_params["grip_detect_a"]),
        "grip_miss_max_a": float(baseline_grip_params["grip_miss_max_a"]),
        "grip_step": float(baseline_grip_params["grip_step"]),
        "max_grip_cmd": float(baseline_grip_params["max_grip_cmd"]),
    }
    baseline_snapshot_path = profiles_dir / f"baseline_{run_ts}.json"
    core.save_calibration_profile(baseline_snapshot_path, baseline_snapshot)
    print(f"[Tune] baseline snapshot saved: {baseline_snapshot_path}")

    def _candidate_score(rows: list[dict]) -> dict:
        success_rows = [r for r in rows if bool(r.get("success", False))]
        fail_count = int(len(rows) - len(success_rows))
        if success_rows:
            med_abs_err_y = float(np.median([abs(float(r.get("err_y_m", float("inf")))) for r in success_rows]))
            med_err_xy = float(np.median([float(r.get("err_xy_m", float("inf"))) for r in success_rows]))
            score = float(med_abs_err_y + 0.5 * med_err_xy + float(fail_penalty_m) * fail_count)
        else:
            med_abs_err_y = float("inf")
            med_err_xy = float("inf")
            score = float("inf")
        return {
            "score": float(score),
            "median_abs_err_y_m": float(med_abs_err_y),
            "median_err_xy_m": float(med_err_xy),
            "success_count": int(len(success_rows)),
            "fail_count": int(fail_count),
            "trials": int(len(rows)),
        }

    def _run_single_trial(y_candidate: float, trial_idx: int, eval_idx: int) -> dict:
        label_prefix = f"tune_y_{eval_idx:03d}_t{trial_idx:02d}"
        projection_geometry.set_cam_offsets(
            cam_off_x_m=float(baseline_offsets["cam_off_x_m"]),
            cam_off_y_m=float(y_candidate),
            cam_off_z_m=float(baseline_offsets["cam_off_z_m"]),
        )
        state.no_pick_miss_count = 0
        row: dict = {
            "ts_utc": _utc_now_iso(),
            "run_id": int(run_ts),
            "eval_idx": int(eval_idx),
            "trial_idx": int(trial_idx),
            "candidate_y_m": float(y_candidate),
            "status": "started",
            "success": False,
        }
        pick_status, centered_pos = pick_actions.run_pick_center_cycle(
            state=state,
            arm=arm,
            per=per,
            det=det,
            label_prefix=label_prefix,
            section_groups=None,
        )
        row["pick_status"] = str(pick_status)
        row["centered_uv"] = (
            None if centered_pos is None else [int(centered_pos[0]), int(centered_pos[1])]
        )
        if pick_status != "ok" or centered_pos is None:
            row["status"] = "pick_failed"
            return row

        carry_status, hold_grip, carry_supervisor = pick_actions.run_grasp_and_carry_common(
            state=state,
            arm=arm,
            per=per,
            centered_pos=centered_pos,
            label_prefix=label_prefix,
            safe_pick_reach_m=PROMPTED_SAFE_PICK_REACH_M,
        )
        row["grasp_status"] = str(carry_status)
        row["hold_grip"] = float(hold_grip)
        if carry_status != "ok":
            row["status"] = "grasp_failed"
            return row

        return_ok, return_reason, return_context = place_actions.execute_return_cube_action(
            state=state,
            arm=arm,
            det=det,
            per=per,
            hold_grip=hold_grip,
            carry_supervisor=carry_supervisor,
        )
        row["return_ok"] = bool(return_ok)
        row["return_reason"] = str(return_reason)
        if not return_ok:
            row["status"] = "return_failed"
            state.holding_object = False
            state.current_hold_grip = 0.0
            state.last_pick_return_xyz = None
            state.last_pick_measured_xyz = None
            return row

        state.holding_object = False
        state.current_hold_grip = 0.0
        state.last_pick_return_xyz = None
        state.last_pick_measured_xyz = None

        return_target_xyz = None
        if isinstance(return_context, dict):
            return_target_xyz = _as_xyz3(return_context.get("target_xyz"))
        verify_stage = misplaced_actions.run_return_verify_stage(
            state=state,
            arm=arm,
            per=per,
            det=det,
            label_prefix=label_prefix,
            return_target_xyz=return_target_xyz,
            required_hits=8,
            measurement_samples=max(int(PICK_OTHER_VALIDATE_SAMPLES), 8),
            reject_cap=max(1, int(PICK_OTHER_MAX_REJECTS)),
            hard_timeout_s=max(1.0, float(PICK_OTHER_HARD_TIMEOUT_S)),
        )
        rv = verify_stage.get("return_verify", {}) if isinstance(verify_stage, dict) else {}
        row["verify_status"] = str(rv.get("status", "unknown"))
        row["verify_confirmed"] = bool(rv.get("confirmed", False))
        row["hits"] = int(rv.get("hits", 0) or 0)
        row["samples"] = int(rv.get("samples", 0) or 0)
        row["err_xy_m"] = float(rv.get("xy_error_m", float("inf")))
        row["err_z_m"] = float(rv.get("z_error_m", float("inf")))
        expected_xyz = _as_xyz3(rv.get("expected_xyz"))
        measured_xyz = _as_xyz3(rv.get("measured_xyz"))
        if (not bool(row["verify_confirmed"])) or expected_xyz is None or measured_xyz is None:
            row["status"] = "verify_failed"
            return row
        err_y = float(measured_xyz[1] - expected_xyz[1])
        row["err_y_m"] = float(err_y)
        row["abs_err_y_m"] = float(abs(err_y))
        row["status"] = "ok"
        row["success"] = True
        return row

    def _parse_offsets(raw: str) -> list[float]:
        out: list[float] = []
        for token in str(raw).split(","):
            s = token.strip()
            if not s:
                continue
            try:
                out.append(float(s))
            except Exception:
                continue
        if not out:
            out = [-0.002, -0.001, 0.0, 0.001, 0.002]
        return out

    def _build_candidates(center_y: float, offsets: list[float]) -> list[float]:
        lo = float(center_y) - float(y_max_delta)
        hi = float(center_y) + float(y_max_delta)
        vals = [float(_clamp(float(center_y) + float(off), lo, hi)) for off in offsets]
        vals = sorted(set([round(v, 6) for v in vals]))
        return vals

    stop_reason = "completed_sweep"
    eval_counter = 0
    best_summary: dict | None = None
    best_y = float(baseline_offsets["cam_off_y_m"])

    try:
        arm.goto_task_space(HOME, duration=1.5, label="startup_home_tune")
        coarse_offsets = _parse_offsets(coarse_offsets_raw)
        coarse_candidates = _build_candidates(float(baseline_offsets["cam_off_y_m"]), coarse_offsets)
        print(f"[TuneY] coarse candidates={coarse_candidates}")

        all_candidate_summaries: list[dict] = []

        def _evaluate_stage(stage_name: str, candidates: list[float]) -> tuple[float, dict | None]:
            nonlocal eval_counter, best_summary, best_y, stop_reason
            stage_best_y = float(candidates[0]) if candidates else float(baseline_offsets["cam_off_y_m"])
            stage_best_summary: dict | None = None
            for y_cand in candidates:
                trial_rows: list[dict] = []
                for t_idx in range(1, trials_per_candidate + 1):
                    eval_counter += 1
                    row = _run_single_trial(float(y_cand), t_idx, eval_counter)
                    row["stage"] = str(stage_name)
                    _write_jsonl_row(tune_log_path, row)
                    trial_rows.append(row)
                summary = _candidate_score(trial_rows)
                summary_row = {
                    "ts_utc": _utc_now_iso(),
                    "run_id": int(run_ts),
                    "row_kind": "candidate_summary",
                    "stage": str(stage_name),
                    "candidate_y_m": float(y_cand),
                    **summary,
                }
                _write_jsonl_row(tune_log_path, summary_row)
                all_candidate_summaries.append(dict(summary_row))
                print(
                    f"[TuneY:{stage_name}] y={float(y_cand):+.6f} "
                    f"score={summary['score']:.5f} "
                    f"med|err_y|={summary['median_abs_err_y_m']:.5f} "
                    f"med_err_xy={summary['median_err_xy_m']:.5f} "
                    f"ok={summary['success_count']}/{summary['trials']}"
                )
                is_better_stage = (
                    stage_best_summary is None
                    or float(summary["score"]) < float(stage_best_summary.get("score", float("inf")))
                )
                if is_better_stage:
                    stage_best_summary = dict(summary_row)
                    stage_best_y = float(y_cand)
                is_better_global = (
                    best_summary is None
                    or float(summary["score"]) < float(best_summary.get("score", float("inf")))
                )
                if is_better_global:
                    best_summary = dict(summary_row)
                    best_y = float(y_cand)
            if stage_best_summary is None:
                stop_reason = f"{stage_name}_no_candidates"
            return float(stage_best_y), stage_best_summary

        coarse_best_y, coarse_best_summary = _evaluate_stage("coarse", coarse_candidates)
        if coarse_best_summary is None:
            stop_reason = "coarse_failed"
        else:
            fine_min = float(coarse_best_y) - float(fine_half_span)
            fine_max = float(coarse_best_y) + float(fine_half_span)
            fine_vals: list[float] = []
            cur = fine_min
            while cur <= (fine_max + 1e-9):
                fine_vals.append(cur)
                cur += float(fine_step)
            fine_offsets = [float(v - float(baseline_offsets["cam_off_y_m"])) for v in fine_vals]
            fine_candidates = _build_candidates(float(baseline_offsets["cam_off_y_m"]), fine_offsets)
            if float(coarse_best_y) not in fine_candidates:
                fine_candidates.append(float(coarse_best_y))
            fine_candidates = sorted(set([round(v, 6) for v in fine_candidates]))
            print(f"[TuneY] fine candidates={fine_candidates}")
            _evaluate_stage("fine", fine_candidates)

        final_y = float(best_y if best_summary is not None else baseline_offsets["cam_off_y_m"])
        projection_geometry.set_cam_offsets(
            cam_off_x_m=float(baseline_offsets["cam_off_x_m"]),
            cam_off_y_m=float(final_y),
            cam_off_z_m=float(baseline_offsets["cam_off_z_m"]),
        )

        final_offsets = projection_geometry.get_cam_offsets()
        final_grasp_z_pick_fraction = float(core.get_grasp_z_pick_fraction())
        final_grip = core.get_grip_tune_params()
        latest_payload = {
            "profile_kind": "tune_latest",
            "tune_mode": "cam_off_y_sweep",
            "updated_at_utc": _utc_now_iso(),
            "run_id": int(run_ts),
            "cam_off_x_m": float(final_offsets["cam_off_x_m"]),
            "cam_off_y_m": float(final_offsets["cam_off_y_m"]),
            "cam_off_z_m": float(final_offsets["cam_off_z_m"]),
            "grasp_z_pick_fraction": float(final_grasp_z_pick_fraction),
            "grip_detect_a": float(final_grip["grip_detect_a"]),
            "grip_miss_max_a": float(final_grip["grip_miss_max_a"]),
            "grip_step": float(final_grip["grip_step"]),
            "max_grip_cmd": float(final_grip["max_grip_cmd"]),
            "baseline_cam_off_x_m": float(baseline_offsets["cam_off_x_m"]),
            "baseline_cam_off_y_m": float(baseline_offsets["cam_off_y_m"]),
            "baseline_cam_off_z_m": float(baseline_offsets["cam_off_z_m"]),
            "best_summary": (None if best_summary is None else dict(best_summary)),
            "candidate_summaries": list(all_candidate_summaries),
            "iterations_run": int(eval_counter),
            "stop_reason": str(stop_reason),
            "baseline_snapshot_path": str(baseline_snapshot_path),
            "tune_log_path": str(tune_log_path),
            "tune_params": {
                "trials_per_candidate": int(trials_per_candidate),
                "fail_penalty_m": float(fail_penalty_m),
                "coarse_offsets": list(coarse_offsets),
                "fine_half_span_m": float(fine_half_span),
                "fine_step_m": float(fine_step),
                "max_delta_from_baseline_m": float(y_max_delta),
            },
        }
        run_profile_path = profiles_dir / f"tuned_{run_ts}.json"
        latest_profile_path = profiles_dir / "latest.json"
        core.save_calibration_profile(run_profile_path, latest_payload)
        core.save_calibration_profile(latest_profile_path, latest_payload)
        print(f"[Tune] completed stop_reason={stop_reason}")
        if best_summary is not None:
            print(
                f"[TuneY] best y={float(best_summary.get('candidate_y_m', float('nan'))):+.6f} "
                f"score={float(best_summary.get('score', float('inf'))):.5f} "
                f"med|err_y|={float(best_summary.get('median_abs_err_y_m', float('inf'))):.5f} "
                f"med_err_xy={float(best_summary.get('median_err_xy_m', float('inf'))):.5f}"
            )
        print(f"[Tune] run profile: {run_profile_path}")
        print(f"[Tune] latest profile: {latest_profile_path}")
        print(f"[Tune] trials log: {tune_log_path}")
        print("[Tune] To use tuned offsets in prompted mode, set QARM_CALIB_PROFILE_PATH to latest.json.")
    except KeyboardInterrupt:
        print("\n\nWARNING: Tune interrupted by user")
    finally:
        shutdown_runtime(per, arm)


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--mode",
        choices=("prompted", "tune"),
        default=(RUNTIME_MODE_DEFAULT if str(RUNTIME_MODE_DEFAULT) in {"prompted", "tune"} else "prompted"),
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[Mode] Ignoring unknown args: {unknown}")
    mode = str(args.mode).strip().lower()
    if mode == "tune":
        main_tune()
    else:
        main_prompted()


__all__ = ["main_prompted", "main_tune", "main"]
