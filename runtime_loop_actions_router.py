"""Post-dispatch action router extracted from runtime_loop (behavior-preserving)."""

from __future__ import annotations

import runtime_loop_actions_correction
import runtime_loop_actions_object
import runtime_loop_actions_place
import runtime_loop_actions_return


def dispatch_post_pre_actions(
    *,
    action_cmd: str,
    state,
    arm,
    det,
    per,
    stack_levels: dict[str, int],
    section_groups: dict[str, list[int]],
    centered_pos: tuple[int, int] | None,
    cube_color: str,
    color_conf: float,
    hold_grip: float,
    carry_supervisor,
    section_left_name: str,
    section_right_name: str,
    prompted_safe_pick_reach_m: float,
    pick_placed_empty_cooldown_steps: int,
    pick_correction_fail_hydrate_refresh_enabled: bool,
    place_release_open_grip: float,
    place_fail_continue_reasons: tuple[str, ...] | list[str] | set[str],
    stack_verify_correction_enabled: bool,
    stack_verify_require_confirmed_for_advance: bool,
    stack_verify_allow_downward_correction: bool,
    stack_verify_downward_require_stable_remeasure: bool,
    track_enable: bool,
    push_min_progress_m: float,
    cycle_count: int,
    home_pose,
    nearest_visible_track_by_uv_fn,
    choose_track_candidate_near_uv_fn,
    classify_cube_color_patch_fn,
    estimate_base_xyz_from_uv_fast_fn,
    execute_push_cube_action_fn,
    finite_xyz_or_none_fn,
    clamp_grip_cmd_fn,
    sync_stack_levels_from_authoritative_state,
    run_startup_stack_bootstrap_verify,
    sync_stack_levels_from_startup_bootstrap,
    log_ledger_stack_snapshot,
    run_post_lift_place_space_refresh,
    record_policy_step,
    run_observe_action,
    capture_pick_lock_snapshot_fn,
) -> dict:
    action_cmd = str(action_cmd)
    if action_cmd in {"pick_placed_left", "pick_placed_right", "pick_misplaced_left", "pick_misplaced_right"}:
        correction_row = runtime_loop_actions_correction.handle_correction_action(
            action_cmd=str(action_cmd),
            state=state,
            arm=arm,
            det=det,
            per=per,
            stack_levels=stack_levels,
            section_left_name=section_left_name,
            section_right_name=section_right_name,
            prompted_safe_pick_reach_m=float(prompted_safe_pick_reach_m),
            pick_placed_empty_cooldown_steps=int(pick_placed_empty_cooldown_steps),
            pick_correction_fail_hydrate_refresh_enabled=bool(pick_correction_fail_hydrate_refresh_enabled),
            record_policy_step=record_policy_step,
            run_observe_action=run_observe_action,
            sync_stack_levels_from_authoritative_state=sync_stack_levels_from_authoritative_state,
            run_startup_stack_bootstrap_verify=run_startup_stack_bootstrap_verify,
            sync_stack_levels_from_startup_bootstrap=sync_stack_levels_from_startup_bootstrap,
        )
        return dict(correction_row)

    object_row = runtime_loop_actions_object.handle_object_action(
        action_cmd=str(action_cmd),
        state=state,
        arm=arm,
        per=per,
        det=det,
        centered_pos=centered_pos,
        cube_color=str(cube_color),
        color_conf=float(color_conf),
        hold_grip=float(hold_grip),
        carry_supervisor=carry_supervisor,
        track_enable=bool(track_enable),
        prompted_safe_pick_reach_m=float(prompted_safe_pick_reach_m),
        push_min_progress_m=float(push_min_progress_m),
        nearest_visible_track_by_uv_fn=nearest_visible_track_by_uv_fn,
        choose_track_candidate_near_uv_fn=choose_track_candidate_near_uv_fn,
        classify_cube_color_patch_fn=classify_cube_color_patch_fn,
        estimate_base_xyz_from_uv_fast_fn=estimate_base_xyz_from_uv_fast_fn,
        execute_push_cube_action_fn=execute_push_cube_action_fn,
        record_policy_step=record_policy_step,
        run_observe_action=run_observe_action,
        capture_pick_lock_snapshot_fn=capture_pick_lock_snapshot_fn,
    )
    if bool(object_row.get("handled", False)):
        return dict(object_row)

    return_row = runtime_loop_actions_return.handle_return_cube_action(
        action_cmd=str(action_cmd),
        state=state,
        arm=arm,
        det=det,
        per=per,
        hold_grip=float(hold_grip),
        carry_supervisor=carry_supervisor,
        cycle_count=int(cycle_count),
        section_left_name=section_left_name,
        section_right_name=section_right_name,
        home_pose=home_pose,
        clamp_grip_cmd_fn=clamp_grip_cmd_fn,
        record_policy_step=record_policy_step,
        run_observe_action=run_observe_action,
        capture_pick_lock_snapshot_fn=capture_pick_lock_snapshot_fn,
    )
    if bool(return_row.get("handled", False)):
        return dict(return_row)

    if action_cmd == "return_placed_cube":
        correction_row = runtime_loop_actions_correction.handle_correction_action(
            action_cmd=str(action_cmd),
            state=state,
            arm=arm,
            det=det,
            per=per,
            stack_levels=stack_levels,
            section_left_name=section_left_name,
            section_right_name=section_right_name,
            prompted_safe_pick_reach_m=float(prompted_safe_pick_reach_m),
            pick_placed_empty_cooldown_steps=int(pick_placed_empty_cooldown_steps),
            pick_correction_fail_hydrate_refresh_enabled=bool(pick_correction_fail_hydrate_refresh_enabled),
            record_policy_step=record_policy_step,
            run_observe_action=run_observe_action,
            sync_stack_levels_from_authoritative_state=sync_stack_levels_from_authoritative_state,
            run_startup_stack_bootstrap_verify=run_startup_stack_bootstrap_verify,
            sync_stack_levels_from_startup_bootstrap=sync_stack_levels_from_startup_bootstrap,
        )
        return dict(correction_row)

    place_row = runtime_loop_actions_place.handle_place_action(
        action_cmd=str(action_cmd),
        state=state,
        arm=arm,
        det=det,
        per=per,
        hold_grip=float(hold_grip),
        carry_supervisor=carry_supervisor,
        centered_pos=centered_pos,
        cube_color=str(cube_color),
        color_conf=float(color_conf),
        section_groups=section_groups,
        stack_levels=stack_levels,
        section_left_name=section_left_name,
        section_right_name=section_right_name,
        home_pose=home_pose,
        place_release_open_grip=float(place_release_open_grip),
        place_fail_continue_reasons=place_fail_continue_reasons,
        stack_verify_correction_enabled=bool(stack_verify_correction_enabled),
        stack_verify_require_confirmed_for_advance=bool(stack_verify_require_confirmed_for_advance),
        stack_verify_allow_downward_correction=bool(stack_verify_allow_downward_correction),
        stack_verify_downward_require_stable_remeasure=bool(stack_verify_downward_require_stable_remeasure),
        finite_xyz_or_none_fn=finite_xyz_or_none_fn,
        clamp_grip_cmd_fn=clamp_grip_cmd_fn,
        sync_stack_levels_from_authoritative_state=sync_stack_levels_from_authoritative_state,
        run_startup_stack_bootstrap_verify=run_startup_stack_bootstrap_verify,
        sync_stack_levels_from_startup_bootstrap=sync_stack_levels_from_startup_bootstrap,
        log_ledger_stack_snapshot=log_ledger_stack_snapshot,
        run_post_lift_place_space_refresh=run_post_lift_place_space_refresh,
        record_policy_step=record_policy_step,
        run_observe_action=run_observe_action,
    )
    if bool(place_row.get("handled", False)):
        return dict(place_row)

    return {
        "handled": False,
        "break_loop": False,
    }
