"""Per-step phase and allowed-command gating extracted from runtime_loop."""

from __future__ import annotations


def compute_phase_and_allowed_commands(
    *,
    state,
    step_index: int,
    section_groups: dict[str, list[int]],
    stack_levels: dict[str, int],
    centered_pos: tuple[int, int] | None,
    cube_color: str,
    color_conf: float,
    observe_fail_streak: int,
    observe_fail_stop_after: int,
    empty_scene_confirm_passes: int,
    max_stack_levels_per_section: int,
    section_left_name: str,
    section_right_name: str,
    planner_io_module,
    policy_log_allowed_commands: bool,
) -> dict:
    if int(state.pick_placed_empty_cooldown_left) > 0:
        state.pick_placed_empty_cooldown_left = int(state.pick_placed_empty_cooldown_left) - 1
    if int(state.pick_placed_empty_cooldown_right) > 0:
        state.pick_placed_empty_cooldown_right = int(state.pick_placed_empty_cooldown_right) - 1
    eff_color, _eff_conf = planner_io_module.resolve_effective_centered_color(
        state, cube_color, color_conf
    )
    eff_color_norm = str(eff_color).strip().lower()
    if state.holding_object:
        phase_name = "place"
    elif centered_pos is None:
        phase_name = "observe"
    elif eff_color_norm in {"orange", "blue"}:
        phase_name = "grasp"
    else:
        phase_name = "classification"
    allowed_commands = planner_io_module.build_prompted_step_allowed_commands(
        state=state,
        section_groups=section_groups,
        stack_levels=stack_levels,
        centered_pos=centered_pos,
        cube_color=cube_color,
        color_conf=color_conf,
    )
    # Temporary anti-loop gate: once color is known for a centered target,
    # force the next decision away from repeated classify.
    if (
        (not bool(state.holding_object))
        and centered_pos is not None
        and eff_color_norm in {"orange", "blue"}
        and "classify_cube" in allowed_commands
    ):
        allowed_commands = [cmd for cmd in allowed_commands if cmd != "classify_cube"]
    removed_by_sanity: list[str] = []
    color_known = eff_color_norm in {"orange", "blue"}
    if not bool(state.holding_object):
        illegal_place = [cmd for cmd in allowed_commands if str(cmd).startswith("place_")]
        if illegal_place:
            removed_by_sanity.extend([str(cmd) for cmd in illegal_place])
            allowed_commands = [cmd for cmd in allowed_commands if not str(cmd).startswith("place_")]
        # Correction picks are not valid during place/hold phases, but must stay
        # available in classification (after centering, before/at classify) so the
        # policy can issue pick_placed_* once section_status shows a stack mismatch.
        correction_phases = {"observe", "grasp", "classification"}
        if str(phase_name).strip().lower() not in correction_phases:
            phase_blocked = [
                cmd for cmd in allowed_commands
                if str(cmd) in {"pick_placed_left", "pick_placed_right", "pick_misplaced_left", "pick_misplaced_right"}
            ]
            if phase_blocked:
                removed_by_sanity.extend([f"non_correction_phase_removed:{str(cmd)}" for cmd in phase_blocked])
                allowed_commands = [
                    cmd for cmd in allowed_commands
                    if str(cmd) not in {"pick_placed_left", "pick_placed_right", "pick_misplaced_left", "pick_misplaced_right"}
                ]
        if centered_pos is None:
            for bad_cmd in ["classify_cube", "grasp_cube", "push_cube"]:
                if bad_cmd in allowed_commands:
                    removed_by_sanity.append(str(bad_cmd))
                    allowed_commands = [cmd for cmd in allowed_commands if cmd != bad_cmd]
        elif (not color_known) and ("grasp_cube" in allowed_commands):
            removed_by_sanity.append("grasp_cube")
            allowed_commands = [cmd for cmd in allowed_commands if cmd != "grasp_cube"]
        elif color_known and ("classify_cube" in allowed_commands):
            removed_by_sanity.append("classify_cube")
            allowed_commands = [cmd for cmd in allowed_commands if cmd != "classify_cube"]
    else:
        disallow_when_holding = {
            "observe_scene",
            "classify_cube",
            "grasp_cube",
            "push_cube",
            "pick_other",
            "pick_placed_left",
            "pick_placed_right",
            "pick_misplaced_left",
            "pick_misplaced_right",
            "return_placed_cube",
            "verify_last_place",
        }
        illegal_holding = [cmd for cmd in allowed_commands if cmd in disallow_when_holding]
        if illegal_holding:
            removed_by_sanity.extend([str(cmd) for cmd in illegal_holding])
            allowed_commands = [cmd for cmd in allowed_commands if cmd not in disallow_when_holding]
    if not allowed_commands:
        allowed_commands = ["stop_run"]
        removed_by_sanity.append("all->stop_run")
    if observe_fail_streak >= int(observe_fail_stop_after):
        allowed_commands = ["stop_run"]
        removed_by_sanity.append(
            f"observe_fail_streak>={int(observe_fail_stop_after)}->stop_run_only"
        )
        print(
            f"[PolicySafety] forcing stop_run_only after observe misses="
            f"{int(observe_fail_streak)} (limit={int(observe_fail_stop_after)})"
        )
    stop_reason_norm = str(getattr(state, "stop_reason", "") or "").strip().lower()
    if stop_reason_norm == "recurrent_grasp_failures_check_pick_space_orientation":
        allowed_commands = ["stop_run"]
        removed_by_sanity.append("pick_space_orientation_check->stop_run_only")
        print(
            "[PolicySafety] forcing stop_run_only after recurrent failed grasp "
            "with visible-target observe miss; check pick-space cube orientation."
        )
    empty_scene_miss_raw = int(state.no_pick_miss_count)
    empty_scene_passes = max(1, int(empty_scene_confirm_passes))
    empty_scene_miss_capped = min(empty_scene_miss_raw, empty_scene_passes)
    empty_scene_progress = f"{empty_scene_miss_capped}/{empty_scene_passes}"
    scene_empty_confirmed = empty_scene_miss_raw >= empty_scene_passes
    if scene_empty_confirmed:
        allowed_commands = ["stop_run"]
        removed_by_sanity.append("scene_empty_confirmed->stop_run_only")
        print(
            f"[PolicySafety] forcing stop_run_only after empty-scene confirmations="
            f"{int(empty_scene_miss_raw)}/{int(empty_scene_passes)}"
        )
    try:
        max_stack_level = max(1, int(max_stack_levels_per_section))
    except Exception:
        max_stack_level = 3
    stacks_full = (
        int(stack_levels.get(section_left_name, 0)) >= int(max_stack_level)
        and int(stack_levels.get(section_right_name, 0)) >= int(max_stack_level)
    )
    if (not bool(state.holding_object)) and bool(stacks_full):
        correction_only = {
            "stop_run",
            "pick_placed_left",
            "pick_placed_right",
        }
        prev_allowed = list(allowed_commands)
        allowed_commands = [cmd for cmd in prev_allowed if str(cmd) in correction_only]
        if "stop_run" not in allowed_commands:
            allowed_commands = ["stop_run"] + list(allowed_commands)
        removed_full_capacity = [
            str(cmd) for cmd in prev_allowed if str(cmd) not in correction_only
        ]
        if removed_full_capacity:
            removed_by_sanity.extend(
                [f"stacks_full_capacity_removed:{cmd}" for cmd in removed_full_capacity]
            )
        print(
            "[PolicySafety] full stack capacity: correction-only commands "
            f"(left={int(stack_levels.get(section_left_name, 0))}/"
            f"{int(max_stack_level)}, right={int(stack_levels.get(section_right_name, 0))}/"
            f"{int(max_stack_level)}) -> {allowed_commands}"
        )
    if removed_by_sanity:
        print(
            f"[PolicyAllowedSanity] removed={removed_by_sanity} "
            f"phase={phase_name} holding={bool(state.holding_object)}"
        )
    if policy_log_allowed_commands:
        print(
            f"[PolicyAllowed] cycle={state.cycle_count} step={step_index} "
            f"phase={phase_name} allowed={allowed_commands}"
        )
        print(
            f"[PolicyEmptyScene] progress={empty_scene_progress} "
            f"confirmed={scene_empty_confirmed}"
        )
    return {
        "phase_name": str(phase_name),
        "allowed_commands": list(allowed_commands),
        "scene_empty_confirmed": bool(scene_empty_confirmed),
        "empty_scene_progress": str(empty_scene_progress),
        "removed_by_sanity": list(removed_by_sanity),
    }
