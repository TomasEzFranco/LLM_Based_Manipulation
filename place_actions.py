#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

_CORE_BIND_READY = False


def _bind_core_globals() -> None:
    global _CORE_BIND_READY
    import runtime_core as core
    protected = {
        '_bind_core_globals', '_CORE_BIND_READY',
        'slot_target_xyz', 'placement_clearance_ok', 'slot_safety_status',
        'PlacePlan', '_resolve_place_target_xyz', '_validate_place_target',
        '_compute_place_transit_extra_m', '_build_place_plan',
        '_goto_place_pose', '_goto_place_vertical_segment',
        '_execute_place_plan', 'safe_place', 'execute_return_cube_action',
        'execute_prompted_place_action',
    }
    for name, value in core.__dict__.items():
        if name.startswith('__') or name in protected:
            continue
        globals()[name] = value
    _CORE_BIND_READY = True


def _apply_place_pick_bias_compensate_with_log(
    target_xyz: np.ndarray,
    *,
    section: str,
    level_label: int,
) -> tuple[np.ndarray, float, float, bool]:
    _bind_core_globals()
    anchor_x = float(target_xyz[0])
    anchor_y = float(target_xyz[1])
    compensated, dx, dy, applied = apply_place_pick_bias_compensate(target_xyz)
    if bool(applied):
        print(
            f"[PlacePickBiasComp] section={section} level={int(level_label)} "
            f"xy_delta=({float(dx):+.3f},{float(dy):+.3f}) "
            f"anchor=({anchor_x:.3f},{anchor_y:.3f}) "
            f"cmd=({float(compensated[0]):.3f},{float(compensated[1]):.3f}) "
        f"scale={float(PLACE_PICK_BIAS_COMPENSATE_SCALE):.2f}"
        )
    return compensated, float(dx), float(dy), bool(applied)


def _stack_anchor_x_comp_allowed(stack_anchor_source: str | None) -> bool:
    return str(stack_anchor_source) != "commanded_place_base_level0"


def slot_target_xyz(slot_index: int) -> np.ndarray:
    _bind_core_globals()
    slots = get_place_slots()
    if 0 <= int(slot_index) < len(slots):
        return slots[int(slot_index)].copy()
    return np.array([np.nan, np.nan, np.nan], dtype=float)
def placement_clearance_ok(
    target_xyz: np.ndarray,
    placed_targets: list[np.ndarray],
    min_sep_m: float | None = None,
):
    _bind_core_globals()
    if min_sep_m is None:
        min_sep_m = float(MIN_PLACE_SLOT_SEPARATION_M)
    if not placed_targets:
        return True, float("inf")
    tx, ty = float(target_xyz[0]), float(target_xyz[1])
    min_dist = float("inf")
    for prev in placed_targets:
        px, py = float(prev[0]), float(prev[1])
        d = math.hypot(tx - px, ty - py)
        if d < min_dist:
            min_dist = d
    return min_dist >= float(min_sep_m), min_dist
def slot_safety_status(
    slot_xyz: np.ndarray,
    placed_targets: list[np.ndarray],
    min_sep_m: float | None = None,
):
    _bind_core_globals()
    if min_sep_m is None:
        min_sep_m = float(MIN_PLACE_SLOT_SEPARATION_M)
    target = np.array(slot_xyz, dtype=float).reshape(-1)
    if target.size < 3 or not np.all(np.isfinite(target[:3])):
        return False, "invalid_slot", float("nan")
    reach = float(np.linalg.norm(target[:3]))
    if reach > MAX_REACH_M:
        return False, "too_far_from_base", float("inf")
    if reach < MIN_PLACE_REACH_M:
        return False, "too_close_to_base", float("inf")
    clear_ok, min_sep = placement_clearance_ok(
        target_xyz=target[:3],
        placed_targets=placed_targets,
        min_sep_m=min_sep_m,
    )
    if not clear_ok:
        return False, "too_close_to_placed", float(min_sep)
    return True, "ok", float(min_sep)


@dataclass(frozen=True)
class PlacePlan:
    target_xyz: np.ndarray
    place_high: np.ndarray
    place_near: np.ndarray
    place_down: np.ndarray
    place_open: np.ndarray
    place_release_clear: np.ndarray
    place_retreat: np.ndarray
    release_touch_grip: float
    release_open_grip: float

def _resolve_place_target_xyz(
    slot_index: int,
    blocked_slots: set[int] | None,
    custom_target_xyz: np.ndarray | None,
) -> tuple[np.ndarray | None, bool, str]:
    _bind_core_globals()
    use_custom_target = custom_target_xyz is not None
    if not use_custom_target and blocked_slots is not None and int(slot_index) in blocked_slots:
        print(f"Place slot {slot_index} is blocked by prior safety rejection.")
        return None, False, "slot_blocked"
    if use_custom_target:
        target_xyz = np.array(custom_target_xyz, dtype=float).reshape(-1)
        if target_xyz.size < 3 or not np.all(np.isfinite(target_xyz[:3])):
            print("Custom place target is invalid.")
            return None, True, "slot_invalid_custom_target"
        return target_xyz[:3].copy(), True, "ok"
    slots = get_place_slots()
    idx = int(slot_index)
    if idx < 0 or idx >= len(slots):
        print(f"Place slot {slot_index} is outside configured safe-grid slots.")
        return None, False, "slot_index_out_of_range"
    return slots[idx].copy(), False, "ok"

def _validate_place_target(
    slot_index: int,
    target_xyz: np.ndarray,
    placed_targets: list[np.ndarray],
    use_custom_target: bool,
    allow_stacked_target: bool,
) -> str:
    _bind_core_globals()
    if use_custom_target and allow_stacked_target:
        reach = float(np.linalg.norm(target_xyz))
        if reach > MAX_REACH_M:
            return "slot_too_far_from_base"
        if reach < MIN_PLACE_REACH_M:
            return "slot_too_close_to_base"
        return "ok"
    slot_ok, slot_reason, min_sep = slot_safety_status(
        slot_xyz=target_xyz,
        placed_targets=placed_targets,
        min_sep_m=MIN_PLACE_SLOT_SEPARATION_M,
    )
    if slot_ok:
        return "ok"
    if slot_reason == "too_far_from_base":
        print(f"Place slot {slot_index} out of reach ({np.linalg.norm(target_xyz):.3f} m > {MAX_REACH_M:.2f} m).")
    elif slot_reason == "too_close_to_base":
        print(f"Place slot {slot_index} too close to base/arm ({np.linalg.norm(target_xyz):.3f} m < {MIN_PLACE_REACH_M:.2f} m).")
    elif slot_reason == "too_close_to_placed":
        print(
            f"Place slot {slot_index} rejected by occupancy margin: "
            f"min_sep={min_sep:.3f} m < {MIN_PLACE_SLOT_SEPARATION_M:.3f} m."
        )
    else:
        print(f"Place slot {slot_index} rejected (reason={slot_reason}).")
    return f"slot_{slot_reason}"

def _compute_place_transit_extra_m(verified_max_stack_level: int) -> float:
    _bind_core_globals()
    start_level = max(1, int(PLACE_TRANSIT_STACK_START_LEVEL))
    stack_level_i = max(0, int(verified_max_stack_level))
    # 3+ behavior by default: level=3 adds one increment, then one per additional level.
    active_levels = max(0, stack_level_i - start_level + 1)
    extra = float(active_levels) * max(0.0, float(PLACE_TRANSIT_STACK_DZ_M))
    return float(min(extra, max(0.0, float(PLACE_TRANSIT_STACK_MAX_EXTRA_M))))


def _stack_release_z_for_level(base_z_m: float, stack_level: int) -> float:
    _bind_core_globals()
    level_i = max(0, int(stack_level))
    base_z = max(float(base_z_m), float(STACK_RELEASE_Z_GUARD_M))
    upper_extra = float(PLACE_STACK_UPPER_EXTRA_Z_M) if level_i >= 1 else 0.0
    level3_extra = float(PLACE_STACK_LEVEL3_EXTRA_Z_M) if level_i >= 2 else 0.0
    return float(base_z + (level_i * float(PLACE_STACK_LEVEL_DZ_M)) + upper_extra + level3_extra)


def _build_place_plan(
    target_xyz: np.ndarray,
    hold_grip: float,
    verified_max_stack_level: int = 0,
) -> tuple[PlacePlan | None, str]:
    _bind_core_globals()
    place_x, place_y, place_down_z = target_xyz.tolist()
    transit_extra_m = _compute_place_transit_extra_m(verified_max_stack_level)
    place_high_z = place_down_z + float(PLACE_APPROACH_LIFT_M) + float(transit_extra_m)
    place_near_z = place_down_z + max(0.005, float(PLACE_NEAR_DESCENT_OFFSET_M))
    if place_near_z >= place_high_z:
        return None, "slot_invalid_vertical_profile"
    release_open_grip = clamp_grip_cmd(PLACE_RELEASE_OPEN_GRIP)
    release_touch_grip = clamp_grip_cmd(PLACE_RELEASE_TOUCH_OPEN_GRIP)
    return (
        PlacePlan(
            target_xyz=target_xyz.copy(),
            place_high=np.array([place_x, place_y, place_high_z, hold_grip]),
            place_near=np.array([place_x, place_y, place_near_z, hold_grip]),
            place_down=np.array([place_x, place_y, place_down_z, hold_grip]),
            place_open=np.array([place_x, place_y, place_down_z, release_touch_grip]),
            place_release_clear=np.array(
                [place_x, place_y, place_down_z + max(0.005, float(PLACE_RELEASE_CLEARANCE_M)), release_open_grip]
            ),
            place_retreat=np.array([place_x, place_y, place_high_z, release_open_grip]),
            release_touch_grip=release_touch_grip,
            release_open_grip=release_open_grip,
        ),
        "ok",
    )

def _goto_place_pose(
    arm: Arm,
    pose: np.ndarray,
    duration: float,
    label: str,
    motion_supervisor: MotionGripSupervisor | None,
    fail_reason: str,
    fail_print: str,
) -> tuple[bool, str]:
    _bind_core_globals()
    if arm.goto_task_space(pose, duration=duration, label=label, motion_supervisor=motion_supervisor):
        return True, "ok"
    if arm.last_motion_reason == "move_overcurrent_unrecoverable":
        return False, arm.last_motion_reason
    print(fail_print)
    return False, fail_reason

def _goto_place_vertical_segment(
    arm: Arm,
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    duration: float,
    label: str,
    motion_supervisor: MotionGripSupervisor | None,
    fail_reason: str,
    fail_print: str,
) -> tuple[bool, str]:
    _bind_core_globals()
    x = float(end_pose[0])
    y = float(end_pose[1])
    z0 = float(start_pose[2])
    z1 = float(end_pose[2])
    grip = float(end_pose[3]) if len(end_pose) > 3 else float(arm._grip_hold)
    dz = abs(z1 - z0)
    step_m = max(0.001, float(PLACE_VERTICAL_STEP_M))
    n_segments = max(1, int(math.ceil(dz / step_m)))
    z_path = np.linspace(z0, z1, n_segments + 1)[1:]
    waypoints = [np.array([x, y, float(zv), grip], dtype=float) for zv in z_path]
    seg_duration = max(0.12, float(duration) / max(1, len(waypoints)))
    ok = arm.goto_task_waypoints_cubic(
        poses=waypoints,
        segment_duration=seg_duration,
        steps_per_segment=max(30, int(PLACE_VERTICAL_STEPS_PER_SEGMENT)),
        label=label,
        motion_supervisor=motion_supervisor,
    )
    if ok:
        return True, "ok"
    if arm.last_motion_reason == "move_overcurrent_unrecoverable":
        return False, arm.last_motion_reason
    print(fail_print)
    return False, fail_reason

def _execute_place_plan(
    arm: Arm,
    slot_index: int,
    plan: PlacePlan,
    det: YOLODetector | None,
    per: Perception | None,
    motion_supervisor: MotionGripSupervisor | None,
) -> tuple[bool, str]:
    _bind_core_globals()
    print(
        f"Placing cube in slot {slot_index} at x={plan.target_xyz[0]:.3f}, y={plan.target_xyz[1]:.3f}, "
        f"z={plan.target_xyz[2]:.3f} (touch_open={plan.release_touch_grip:.3f}, post_open={plan.release_open_grip:.3f})"
    )
    ok, reason = _goto_place_pose(
        arm=arm,
        pose=plan.place_high,
        duration=PLACE_ALIGN_DURATION_S,
        label=f"place_slot_{slot_index}_align_high",
        motion_supervisor=motion_supervisor,
        fail_reason="place_align_high_failed",
        fail_print="FAILED: Failed to reach high align pose before descent.",
    )
    if not ok:
        return False, reason
    ok, reason = _goto_place_vertical_segment(
        arm=arm,
        start_pose=plan.place_high,
        end_pose=plan.place_near,
        duration=PLACE_DESCEND_NEAR_DURATION_S,
        label=f"place_slot_{slot_index}_descend_near",
        motion_supervisor=motion_supervisor,
        fail_reason="place_descend_near_failed",
        fail_print="FAILED: Failed to descend to near-table intermediate pose.",
    )
    if not ok:
        return False, reason
    ok, reason = _goto_place_vertical_segment(
        arm=arm,
        start_pose=plan.place_near,
        end_pose=plan.place_down,
        duration=PLACE_DESCEND_FINAL_DURATION_S,
        label=f"place_slot_{slot_index}_descend_final",
        motion_supervisor=motion_supervisor,
        fail_reason="place_descend_final_failed",
        fail_print="FAILED: Failed to reach final place-down pose.",
    )
    if not ok:
        return False, reason
    ok, reason = _goto_place_pose(
        arm=arm,
        pose=plan.place_open,
        duration=PLACE_RELEASE_DURATION_S,
        label=f"place_slot_{slot_index}_release_touch_open",
        motion_supervisor=motion_supervisor,
        fail_reason="place_release_failed",
        fail_print="FAILED: Failed to release cube at place pose.",
    )
    if not ok:
        return False, reason
    time.sleep(PLACE_OPEN_HOLD_S)
    ok, reason = _goto_place_vertical_segment(
        arm=arm,
        start_pose=plan.place_open,
        end_pose=plan.place_release_clear,
        duration=PLACE_RELEASE_CLEARANCE_DURATION_S,
        label=f"place_slot_{slot_index}_release_clearance",
        motion_supervisor=motion_supervisor,
        fail_reason="place_release_clearance_failed",
        fail_print="FAILED: Failed to clear upward after touch-open release.",
    )
    if not ok:
        return False, reason
    ok, reason = _goto_place_vertical_segment(
        arm=arm,
        start_pose=plan.place_release_clear,
        end_pose=plan.place_retreat,
        duration=PLACE_RETREAT_DURATION_S,
        label=f"place_slot_{slot_index}_retreat_vertical",
        motion_supervisor=motion_supervisor,
        fail_reason="place_retreat_vertical_failed",
        fail_print="FAILED: Failed to retreat vertically after release.",
    )
    if not ok:
        return False, reason
    return True, "ok"


def safe_place(
    arm: Arm,
    slot_index: int,
    grip: float,
    placed_targets: list[np.ndarray] | None = None,
    blocked_slots: set[int] | None = None,
    det: YOLODetector | None = None,
    per: Perception | None = None,
    motion_supervisor: MotionGripSupervisor | None = None,
    custom_target_xyz: np.ndarray | None = None,
    allow_stacked_target: bool = False,
    verified_max_stack_level: int = 0,
):
    _bind_core_globals()
    hold_grip = clamp_grip_cmd(grip)
    placed_targets = [] if placed_targets is None else placed_targets
    target_xyz, use_custom_target, resolve_reason = _resolve_place_target_xyz(
        slot_index=slot_index,
        blocked_slots=blocked_slots,
        custom_target_xyz=custom_target_xyz,
    )
    if target_xyz is None:
        return False, resolve_reason
    validate_reason = _validate_place_target(
        slot_index=slot_index,
        target_xyz=target_xyz,
        placed_targets=placed_targets,
        use_custom_target=use_custom_target,
        allow_stacked_target=allow_stacked_target,
    )
    if validate_reason != "ok":
        return False, validate_reason
    plan, plan_reason = _build_place_plan(
        target_xyz=target_xyz,
        hold_grip=hold_grip,
        verified_max_stack_level=int(verified_max_stack_level),
    )
    if plan is None:
        transit_extra_m = _compute_place_transit_extra_m(int(verified_max_stack_level))
        print(
            f"Place slot {slot_index} rejected: near descent z "
            f"({target_xyz[2] + max(0.005, float(PLACE_NEAR_DESCENT_OFFSET_M)):.3f}) "
            f"is not below high align z "
            f"({target_xyz[2] + float(PLACE_APPROACH_LIFT_M) + float(transit_extra_m):.3f})."
        )
        return False, plan_reason
    return _execute_place_plan(
        arm=arm,
        slot_index=slot_index,
        plan=plan,
        det=det,
        per=per,
        motion_supervisor=motion_supervisor,
    )

def execute_return_cube_action(
    *,
    state: CycleState,
    arm: Arm,
    det: YOLODetector,
    per: Perception,
    hold_grip: float,
    carry_supervisor: MotionGripSupervisor | None,
) -> tuple[bool, str, dict | None]:
    _bind_core_globals()
    return_xyz = state.last_pick_return_xyz
    if not isinstance(return_xyz, (list, tuple)) or len(return_xyz) < 3:
        return False, "no_pick_return_target", None
    try:
        target_xyz = np.array([float(return_xyz[0]), float(return_xyz[1]), float(return_xyz[2])], dtype=float)
    except (TypeError, ValueError):
        return False, "invalid_pick_return_target", None
    if not np.all(np.isfinite(target_xyz[:3])):
        return False, "invalid_pick_return_target", None
    # Keep return release height conservative to avoid pressing into table on replay.
    return_z_before = float(target_xyz[2])
    return_z_floor = max(float(return_z_before), float(PLACE_RELEASE_Z_M))
    return_z_lift = max(0.0, float(RETURN_CUBE_Z_LIFT_M))
    target_xyz[2] = float(return_z_floor + return_z_lift)
    print(
        f"[ReturnCubeZ] z={return_z_before:.3f}->{float(target_xyz[2]):.3f} "
        f"floor={float(return_z_floor):.3f} lift={float(return_z_lift):+.3f}"
    )
    pre_reconcile = reconcile_scene(
        state=state,
        arm=arm,
        per=per,
        det=det,
        side="all",
        mode="pre_return_release",
        target_xyz=target_xyz,
        include_pick_rows=True,
    )
    if bool(pre_reconcile.get("collision_risk", False)):
        return False, "return_drop_occupied", {
            "target_xyz": _finite_xyz_or_none(target_xyz),
            "pre_reconcile": pre_reconcile,
            "timestamp_ms": int(time.time() * 1000),
        }
    ok, reason = safe_place(
        arm=arm,
        slot_index=-1,
        grip=hold_grip,
        placed_targets=state.placed_targets,
        blocked_slots=state.blocked_slots,
        det=det,
        per=per,
        motion_supervisor=carry_supervisor,
        custom_target_xyz=target_xyz,
        allow_stacked_target=True,
    )
    context = {
        "target_xyz": _finite_xyz_or_none(target_xyz),
        "pre_reconcile": pre_reconcile,
        "timestamp_ms": int(time.time() * 1000),
    }
    return bool(ok), str(reason), context

# ============================= Place logic =============================
def execute_prompted_place_action(
    action_cmd: str,
    state: CycleState,
    arm: Arm,
    det: YOLODetector,
    per: Perception,
    hold_grip: float,
    section_groups: dict[str, list[int]],
    placed_targets: list[np.ndarray],
    blocked_slots: set[int],
    stack_levels: dict[str, int],
    carry_supervisor: MotionGripSupervisor | None,
) -> tuple[bool, str, int | None, dict | None]:
    _bind_core_globals()
    if PROMPTED_COLUMN_ONLY and action_cmd in {"place_left_stack", "place_right_stack"}:
        return False, "stack_disabled_column_mode", None, None
    slots = get_place_slots()
    section = None
    stack_mode = False
    if action_cmd == "place_left":
        section = SECTION_LEFT_NAME
    elif action_cmd == "place_right":
        section = SECTION_RIGHT_NAME
    elif action_cmd == "place_left_stack":
        section = SECTION_LEFT_NAME
        stack_mode = True
    elif action_cmd == "place_right_stack":
        section = SECTION_RIGHT_NAME
        stack_mode = True
    else:
        return False, f"unsupported_action:{action_cmd}", None, None
    verified_max_stack_level = max([0] + [int(v) for v in stack_levels.values()])
    place_x_bias_m = 0.0
    place_y_bias_m = float(PLACE_Y_BIAS_M)
    if not stack_mode:
        slot_index = next_slot_in_section(section, section_groups, placed_targets, blocked_slots)
        if slot_index is None:
            return False, f"no_slot_available:{section}", None, None
        expected_xyz = slot_target_xyz(slot_index).copy()
        expected_xyz[0] = float(expected_xyz[0]) + float(place_x_bias_m)
        expected_xyz[1] = float(expected_xyz[1]) + float(place_y_bias_m)
        pick_comp_dx = 0.0
        pick_comp_dy = 0.0
        pick_comp_applied = False
        expected_xyz, pick_comp_dx, pick_comp_dy, pick_comp_applied = _apply_place_pick_bias_compensate_with_log(
            expected_xyz,
            section=str(section),
            level_label=int(stack_levels.get(section, 0)),
        )
        expected_xyz = apply_place_command_xy_offset(expected_xyz)
        pre_reconcile = reconcile_scene(
            state=state,
            arm=arm,
            per=per,
            det=det,
            side=str(section),
            mode="pre_place_release",
            target_xyz=expected_xyz,
            include_pick_rows=False,
        )
        if bool(pre_reconcile.get("collision_risk", False)):
            return False, "place_collision_risk", int(slot_index), {
                "command": action_cmd,
                "section": section,
                "slot_index": int(slot_index),
                "expected_xyz": _finite_xyz_or_none(expected_xyz),
                "pre_reconcile": pre_reconcile,
                "timestamp_ms": int(time.time() * 1000),
            }
        pending_stack_level = None
        stack_allowed = (not PROMPTED_COLUMN_ONLY) and ENABLE_STACK_ACTIONS
        base_candidates = section_groups.get(section, [])
        if stack_allowed and base_candidates:
            base_slot = int(base_candidates[0])
            if int(slot_index) == base_slot:
                pending_stack_level = max(int(stack_levels.get(section, 0)), 1)
        pre_obs: dict = {}
        if PLACE_VERIFY_V2_ENABLED:
            pre_obs = collect_slot_observations(
                det=det,
                arm=arm,
                per=per,
                expected_xyz=expected_xyz,
                samples=max(1, int(PLACE_VERIFY_V2_SAMPLES_PRE)),
                radius_m=max(0.0, float(PLACE_VERIFY_V2_RADIUS_M)),
                min_conf=float(PLACE_VERIFY_MIN_CONF),
                max_abs_z_error_m=max(0.01, float(PLACE_VERIFY_V2_Z_MARGIN_M) * 2.0),
            )
        ok, reason = safe_place(
            arm=arm,
            slot_index=slot_index,
            grip=hold_grip,
            placed_targets=placed_targets,
            blocked_slots=blocked_slots,
            det=det,
            per=per,
            motion_supervisor=carry_supervisor,
            verified_max_stack_level=verified_max_stack_level,
        )
        place_context = {
            "command": action_cmd,
            "section": section,
            "slot_index": int(slot_index),
            "expected_xyz": _finite_xyz_or_none(expected_xyz),
            "place_bias_xy_m": [float(place_x_bias_m), float(place_y_bias_m)],
            "place_cmd_offset_xy_m": [
                float(PLACE_CMD_X_OFFSET_M if PLACE_CMD_XY_OFFSET_ENABLED else 0.0),
                float(PLACE_CMD_Y_OFFSET_M if PLACE_CMD_XY_OFFSET_ENABLED else 0.0),
            ],
            "place_pick_bias_compensate_xy_m": [float(pick_comp_dx), float(pick_comp_dy)],
            "place_pick_bias_compensate_skipped": not bool(pick_comp_applied),
            "stack_level": int(stack_levels.get(section, 0)),
            "pending_stack_level": pending_stack_level,
            "pre_observation": dict(pre_obs),
            "pre_reconcile": pre_reconcile,
            "timestamp_ms": int(time.time() * 1000),
        }
        if ok:
            placed_targets.append(np.array([expected_xyz[0], expected_xyz[1], slots[slot_index][2]], dtype=float))
        return ok, reason, slot_index, place_context
    # Stack mode: reuse sticky commanded XY from the first level-0 place on this side (not verify measured).
    base_candidates = section_groups.get(section, [])
    if not base_candidates:
        return False, f"stack_no_base_slot:{section}", None, None
    base_slot = int(base_candidates[0])
    if base_slot < 0 or base_slot >= len(slots):
        return False, "stack_base_slot_invalid", None, None
    level = int(stack_levels.get(section, 0))
    if level >= max(1, int(MAX_STACK_LEVELS_PER_SECTION)):
        return False, f"stack_level_limit:{section}", None, None
    target_xyz = slot_target_xyz(base_slot).copy()
    stack_place_x_bias_m = float(place_x_bias_m)
    stack_place_y_bias_m = float(place_y_bias_m)
    stack_anchor_xyz = None
    stack_anchor_source = "none"
    stack_anchor_xyz, stack_anchor_source = get_latest_side_stack_anchor_xyz(state, section)
    stack_anchor_xy_locked = False
    if stack_anchor_xyz is not None:
        target_xyz[0] = float(stack_anchor_xyz[0])
        target_xyz[1] = float(stack_anchor_xyz[1])
        stack_anchor_xy_locked = True
        print(
            f"[StackAnchor] section={section} level={int(level)} "
            f"source={stack_anchor_source} "
            f"anchor_xy=({float(target_xyz[0]):.3f},{float(target_xyz[1]):.3f}) "
            f"bias_skipped=yes"
        )
    elif int(level) > 0:
        log_stack_anchor_missing(state, section, stack_level=int(level))
        return False, "stack_anchor_missing_side_xyz", int(base_slot), {
            "command": action_cmd,
            "section": section,
            "slot_index": int(base_slot),
            "stack_level": int(level),
            "stack_anchor_source": str(stack_anchor_source),
            "timestamp_ms": int(time.time() * 1000),
        }
    if not bool(stack_anchor_xy_locked):
        target_xyz[0] = float(target_xyz[0]) + float(stack_place_x_bias_m)
        target_xyz[1] = float(target_xyz[1]) + float(stack_place_y_bias_m)
    pick_comp_dx = 0.0
    pick_comp_dy = 0.0
    pick_comp_applied = False
    skip_stack_anchor_pick_comp = bool(
        stack_anchor_xy_locked
        and not bool(PLACE_PICK_BIAS_COMPENSATE_STACK_ANCHOR_ENABLED)
    )
    if bool(skip_stack_anchor_pick_comp):
        print(
            f"[PlacePickBiasComp] section={section} level={int(level)} "
            f"skipped=stack_anchor_locked source={stack_anchor_source} "
            f"anchor=({float(target_xyz[0]):.3f},{float(target_xyz[1]):.3f}) "
            f"stack_anchor_pick_comp={bool(PLACE_PICK_BIAS_COMPENSATE_STACK_ANCHOR_ENABLED)}"
        )
    else:
        target_xyz, pick_comp_dx, pick_comp_dy, pick_comp_applied = _apply_place_pick_bias_compensate_with_log(
            target_xyz,
            section=str(section),
            level_label=int(level),
        )
    target_xyz[2] = _stack_release_z_for_level(float(target_xyz[2]), int(level))
    skip_cmd_offset = bool(stack_anchor_xy_locked and PLACE_CMD_OFFSET_SKIP_STACK_ANCHOR)
    target_xyz = apply_place_command_xy_offset(target_xyz, skip=bool(skip_cmd_offset))
    if bool(PLACE_CMD_XY_OFFSET_ENABLED) and not bool(skip_cmd_offset):
        print(
            f"[PlaceCmdOffset] section={section} level={int(level)} "
            f"xy=({float(PLACE_CMD_X_OFFSET_M):+.3f},{float(PLACE_CMD_Y_OFFSET_M):+.3f}) z_unchanged=yes"
        )
    elif bool(skip_cmd_offset):
        print(
            f"[PlaceCmdOffset] section={section} level={int(level)} "
            f"skipped=yes source={stack_anchor_source} "
            f"(hydrate/commanded anchor XY is authoritative)"
        )
    nominal_target_xyz = target_xyz.copy()
    command_target_xyz = nominal_target_xyz.copy()
    stack_level_x_dx = 0.0
    stack_pick_x_dx = 0.0
    stack_pick_level2_extra_dx = 0.0
    stack_pick_near_z_extra = 0.0
    stack_pick_far_z_extra = 0.0
    stack_anchor_x_dx = 0.0
    stack_pick_x_meta = {
        "enabled": bool(STACK_PICK_X_OFFSET_ENABLED),
        "applied": False,
        "reason": "base_level_no_offset" if int(level) <= 0 else "disabled",
        "offset_m": 0.0,
    }
    if (
        bool(STACK_ANCHOR_X_COMP_ENABLED)
        and int(level) > 0
        and bool(stack_anchor_xy_locked)
        and bool(_stack_anchor_x_comp_allowed(stack_anchor_source))
    ):
        stack_anchor_x_dx = (
            float(STACK_ANCHOR_X_COMP_LEVEL1_M)
            if int(level) == 1
            else float(STACK_ANCHOR_X_COMP_M)
        )
        print(
            f"[StackAnchorXComp] section={section} level={int(level)} "
            f"source={stack_anchor_source} dx={float(stack_anchor_x_dx):+.3f}"
        )
    elif (
        bool(STACK_ANCHOR_X_COMP_ENABLED)
        and int(level) > 0
        and bool(stack_anchor_xy_locked)
    ):
        print(
            f"[StackAnchorXComp] section={section} level={int(level)} "
            f"source={stack_anchor_source} skipped=commanded_base_anchor dx=+0.000"
        )
    stack_pick_x_offset_allowed = str(stack_anchor_source) == "commanded_place_base_level0"
    if bool(STACK_PICK_X_OFFSET_ENABLED) and int(level) > 0 and bool(stack_pick_x_offset_allowed):
        stack_pick_x_dx, stack_pick_x_meta = compute_stack_pick_x_offset(
            getattr(state, "last_pick_measured_xyz", None)
        )
        stack_pick_reason = str(stack_pick_x_meta.get("reason", ""))
        stack_pick_nonfatal = stack_pick_reason == "positive_x_offset_suppressed"
        if stack_pick_reason != "ok" and not bool(stack_pick_nonfatal) and bool(STACK_PICK_X_OFFSET_REQUIRE_PICK_X):
            print(
                f"[StackPickXOffset] failed reason={stack_pick_x_meta.get('reason')} "
                f"level={int(level)} require_pick_x={bool(STACK_PICK_X_OFFSET_REQUIRE_PICK_X)}"
            )
            return False, "stack_pick_x_offset_missing_pick_xyz", int(base_slot), {
                "command": action_cmd,
                "section": section,
                "slot_index": int(base_slot),
                "expected_xyz": _finite_xyz_or_none(nominal_target_xyz),
                "command_xyz": _finite_xyz_or_none(command_target_xyz),
                "stack_pick_x_offset_m": 0.0,
                "stack_pick_level2_extra_m": 0.0,
                "stack_pick_near_z_extra_m": 0.0,
                "stack_pick_far_z_extra_m": 0.0,
                "stack_anchor_x_comp_m": float(stack_anchor_x_dx),
                "stack_hydrate_anchor_x_comp_m": float(stack_anchor_x_dx),
                "stack_pick_x_offset_meta": dict(stack_pick_x_meta),
                "stack_level_x_offset_m": 0.0,
                "stack_anchor_xyz": (None if stack_anchor_xyz is None else [float(stack_anchor_xyz[0]), float(stack_anchor_xyz[1]), float(stack_anchor_xyz[2])]),
                "stack_anchor_source": str(stack_anchor_source),
                "timestamp_ms": int(time.time() * 1000),
            }
    elif bool(STACK_PICK_X_OFFSET_ENABLED) and int(level) > 0:
        stack_pick_x_meta["reason"] = f"anchor_source_no_pick_x_offset:{stack_anchor_source}"
    elif bool(STACK_X_LEVEL_OFFSET_ENABLED):
        if int(level) == 1:
            stack_level_x_dx = float(STACK_X_LEVEL1_OFFSET)
        elif int(level) == 2:
            stack_level_x_dx = float(STACK_X_LEVEL2_OFFSET)
    if (
        bool(STACK_PICK_X_OFFSET_ENABLED)
        and int(level) >= 2
        and bool(stack_pick_x_offset_allowed)
    ):
        stack_pick_level2_extra_dx = float(STACK_PICK_X_LEVEL2_EXTRA_M)
        stack_pick_x_meta["level2_extra_m"] = float(stack_pick_level2_extra_dx)
    command_target_xyz[0] = (
        float(nominal_target_xyz[0])
        + float(stack_pick_x_dx)
        + float(stack_pick_level2_extra_dx)
        + float(stack_level_x_dx)
        + float(stack_anchor_x_dx)
    )
    if str(stack_pick_x_meta.get("reason", "")) == "ok" and float(stack_pick_x_dx) < 0.0:
        stack_pick_near_z_extra = float(STACK_PICK_X_NEAR_Z_EXTRA_M)
        command_target_xyz[2] = float(command_target_xyz[2]) + float(stack_pick_near_z_extra)
    elif str(stack_pick_x_meta.get("reason", "")) == "ok" and float(stack_pick_x_dx) > 0.0:
        stack_pick_far_z_extra = float(STACK_PICK_X_FAR_Z_EXTRA_M)
        command_target_xyz[2] = float(command_target_xyz[2]) + float(stack_pick_far_z_extra)
    print(
        f"[StackPickXOffset] level={int(level)} "
        f"enabled={bool(STACK_PICK_X_OFFSET_ENABLED)} "
        f"reason={stack_pick_x_meta.get('reason', 'legacy_level_offset')} "
        f"pick_x={float(stack_pick_x_meta.get('pick_x_m', float('nan'))):.3f} "
        f"near={float(stack_pick_x_meta.get('near_x_m', float('nan'))):.3f} "
        f"far={float(stack_pick_x_meta.get('far_x_m', float('nan'))):.3f} "
        f"t={float(stack_pick_x_meta.get('t', 0.0)):.2f} "
        f"x_nom={float(nominal_target_xyz[0]):.3f} "
        f"dx={float(stack_pick_x_dx):+.3f} "
        f"level2_extra_dx={float(stack_pick_level2_extra_dx):+.3f} "
        f"legacy_dx={float(stack_level_x_dx):+.3f} "
        f"anchor_dx={float(stack_anchor_x_dx):+.3f} "
        f"x_cmd={float(command_target_xyz[0]):.3f} "
        f"y={float(nominal_target_xyz[1]):.3f} "
        f"z={float(command_target_xyz[2]):.3f} "
        f"near_z_extra={float(stack_pick_near_z_extra):+.3f} "
        f"far_z_extra={float(stack_pick_far_z_extra):+.3f}"
    )
    pre_reconcile = reconcile_scene(
        state=state,
        arm=arm,
        per=per,
        det=det,
        side=str(section),
        mode="pre_stack_release",
        target_xyz=command_target_xyz,
        include_pick_rows=False,
    )
    if bool(pre_reconcile.get("collision_risk", False)):
        return False, "place_collision_risk", int(base_slot), {
            "command": action_cmd,
            "section": section,
            "slot_index": int(base_slot),
            "expected_xyz": _finite_xyz_or_none(nominal_target_xyz),
            "command_xyz": _finite_xyz_or_none(command_target_xyz),
            "stack_level_x_offset_m": float(stack_level_x_dx),
            "stack_pick_x_offset_m": float(stack_pick_x_dx),
            "stack_pick_level2_extra_m": float(stack_pick_level2_extra_dx),
            "stack_pick_near_z_extra_m": float(stack_pick_near_z_extra),
            "stack_pick_far_z_extra_m": float(stack_pick_far_z_extra),
            "stack_anchor_x_comp_m": float(stack_anchor_x_dx),
            "stack_hydrate_anchor_x_comp_m": float(stack_anchor_x_dx),
            "stack_pick_x_offset_meta": dict(stack_pick_x_meta),
            "stack_anchor_xyz": (None if stack_anchor_xyz is None else [float(stack_anchor_xyz[0]), float(stack_anchor_xyz[1]), float(stack_anchor_xyz[2])]),
            "stack_anchor_source": str(stack_anchor_source),
            "pre_reconcile": pre_reconcile,
            "timestamp_ms": int(time.time() * 1000),
        }
    pre_obs: dict = {}
    if PLACE_VERIFY_V2_ENABLED:
        stack_level_for_pre = int(level + 1)
        stack_min_z_pre = compute_verify_stack_min_z(float(target_xyz[2]), stack_level_for_pre)
        stack_prefer_top = bool(PLACE_VERIFY_V2_STACK_PREFER_TOP and stack_level_for_pre >= 2)
        pre_obs = collect_slot_observations(
            det=det,
            arm=arm,
            per=per,
            expected_xyz=nominal_target_xyz,
            samples=max(1, int(PLACE_VERIFY_V2_SAMPLES_PRE)),
            radius_m=max(0.0, float(PLACE_VERIFY_V2_RADIUS_M)),
            min_conf=float(PLACE_VERIFY_MIN_CONF),
            max_abs_z_error_m=max(0.01, float(PLACE_VERIFY_V2_Z_MARGIN_M) * 2.0),
            min_z_m=stack_min_z_pre,
            prefer_higher_z=stack_prefer_top,
        )
    ok, reason = safe_place(
        arm=arm,
        slot_index=base_slot,
        grip=hold_grip,
        placed_targets=placed_targets,
        blocked_slots=blocked_slots,
        det=det,
        per=per,
        motion_supervisor=carry_supervisor,
        custom_target_xyz=command_target_xyz,
        allow_stacked_target=True,
        verified_max_stack_level=verified_max_stack_level,
    )
    place_context = {
        "command": action_cmd,
        "section": section,
        "slot_index": int(base_slot),
        "expected_xyz": _finite_xyz_or_none(nominal_target_xyz),
        "command_xyz": _finite_xyz_or_none(command_target_xyz),
        "stack_level_x_offset_m": float(stack_level_x_dx),
        "stack_pick_x_offset_m": float(stack_pick_x_dx),
        "stack_pick_level2_extra_m": float(stack_pick_level2_extra_dx),
        "stack_pick_near_z_extra_m": float(stack_pick_near_z_extra),
        "stack_pick_far_z_extra_m": float(stack_pick_far_z_extra),
        "stack_anchor_x_comp_m": float(stack_anchor_x_dx),
        "stack_hydrate_anchor_x_comp_m": float(stack_anchor_x_dx),
        "stack_pick_x_offset_meta": dict(stack_pick_x_meta),
        "place_bias_xy_m": [float(stack_place_x_bias_m), float(stack_place_y_bias_m)],
        "place_cmd_offset_xy_m": [
            0.0 if bool(skip_cmd_offset) else float(PLACE_CMD_X_OFFSET_M if PLACE_CMD_XY_OFFSET_ENABLED else 0.0),
            0.0 if bool(skip_cmd_offset) else float(PLACE_CMD_Y_OFFSET_M if PLACE_CMD_XY_OFFSET_ENABLED else 0.0),
        ],
        "place_cmd_offset_skipped": bool(skip_cmd_offset),
        "place_pick_bias_compensate_xy_m": [float(pick_comp_dx), float(pick_comp_dy)],
        "place_pick_bias_compensate_skipped": bool(skip_stack_anchor_pick_comp) or not bool(pick_comp_applied),
        "place_pick_bias_compensate_skip_reason": (
            "stack_anchor_locked" if bool(skip_stack_anchor_pick_comp) else ""
        ),
        "stack_anchor_xyz": (None if stack_anchor_xyz is None else [float(stack_anchor_xyz[0]), float(stack_anchor_xyz[1]), float(stack_anchor_xyz[2])]),
        "stack_anchor_source": str(stack_anchor_source),
        "stack_level": int(level),
        "pending_stack_level": int(level + 1),
        "pre_observation": dict(pre_obs),
        "pre_reconcile": pre_reconcile,
        "timestamp_ms": int(time.time() * 1000),
    }
    if ok:
        placed_targets.append(np.array([nominal_target_xyz[0], nominal_target_xyz[1], slots[base_slot][2]], dtype=float))
    return ok, reason, base_slot, place_context
