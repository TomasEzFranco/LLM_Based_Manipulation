"""Correction-action handlers extracted from runtime_loop (behavior-preserving)."""

from __future__ import annotations

import misplaced_actions
import runtime_loop_dispatch
import stack_scene
import verify_v2


def _compact_misplaced_summary(ctx: dict) -> str:
    """One-line failure summary; avoids dumping nested misplaced_lock/drop_grid blobs."""
    lock = ctx.get("misplaced_lock") if isinstance(ctx.get("misplaced_lock"), dict) else {}
    grid = ctx.get("drop_grid") if isinstance(ctx.get("drop_grid"), dict) else {}
    lock_track = ctx.get("lock_track_id")
    if lock_track is None:
        lock_track = lock.get("required_track_id") or lock.get("last_candidate_track_id")
    lock_uv = ctx.get("lock_uv")
    hits = lock.get("hits")
    req_hits = lock.get("required_hits")
    attempts = lock.get("lock_attempts") or lock.get("attempts_used")
    max_attempts = lock.get("max_lock_attempts")
    exit_reason = lock.get("exit_reason") or lock.get("lock_loop_exit_reason")
    lock_part = (
        f"lock=track={lock_track} uv={lock_uv} "
        f"hits={hits}/{req_hits} attempts={attempts}/{max_attempts} exit={exit_reason}"
    )
    grid_part = f"drop_grid=ok={grid.get('ok')} reason={grid.get('reason')}"
    return (
        f"side={ctx.get('target_section')} "
        f"target_xyz={ctx.get('target_xyz')} drop_xyz={ctx.get('drop_xyz')} "
        f"top_ref_xyz={ctx.get('top_ref_xyz')} top_ref_z={ctx.get('top_ref_z')} "
        f"{lock_part} {grid_part} "
        f"return_outcome={ctx.get('return_outcome')} "
        f"released_after_failure={ctx.get('released_after_failure')} "
        f"holding_after_failure={ctx.get('holding_after_failure')}"
    )


def handle_correction_action(
    *,
    action_cmd: str,
    state,
    arm,
    det,
    per,
    stack_levels: dict[str, int],
    section_left_name: str,
    section_right_name: str,
    prompted_safe_pick_reach_m: float,
    pick_placed_empty_cooldown_steps: int,
    pick_correction_fail_hydrate_refresh_enabled: bool,
    record_policy_step,
    run_observe_action,
    sync_stack_levels_from_authoritative_state,
    run_startup_stack_bootstrap_verify=None,
    sync_stack_levels_from_startup_bootstrap=None,
) -> dict:
    action_cmd = str(action_cmd)
    if action_cmd in {"pick_placed_left", "pick_placed_right", "pick_misplaced_left", "pick_misplaced_right"}:
        if state.holding_object:
            state.cycles_without_place_progress += 1
            state.invalid_precondition_recoveries += 1
            print(f"[Policy] {action_cmd} requested while holding object; auto-observing first.")
            record_policy_step(action_cmd, f"{action_cmd}_while_holding", progress=False)
            runtime_loop_dispatch.run_auto_recovery_observe(run_observe_action=run_observe_action)
            return {"handled": True, "break_loop": False}
        preferred_section = None
        if action_cmd in {"pick_placed_left", "pick_misplaced_left"}:
            preferred_section = section_left_name
        elif action_cmd in {"pick_placed_right", "pick_misplaced_right"}:
            preferred_section = section_right_name
        is_pick_placed_cmd = action_cmd in {"pick_placed_left", "pick_placed_right"}
        pm_ok, pm_reason, pm_context = misplaced_actions.execute_pick_misplaced_cube_action(
            state=state,
            arm=arm,
            det=det,
            per=per,
            label_prefix=f"prompted_cycle_{state.cycle_count}",
            preferred_section=preferred_section,
            safe_pick_reach_m=prompted_safe_pick_reach_m,
        )
        pm_context_row = pm_context if isinstance(pm_context, dict) else {}
        if pm_ok:
            if bool(pm_context_row.get("released_after_failure", False)):
                state.holding_object = False
                state.current_hold_grip = 0.0
                state.last_pick_measured_xyz = None
            if preferred_section == section_left_name:
                state.pick_placed_empty_cooldown_left = 0
            elif preferred_section == section_right_name:
                state.pick_placed_empty_cooldown_right = 0
            state.active_target_track_id = None
            state.cycles_without_place_progress = 0
            print(
                f"[PickMisplaced] success target_section="
                f"{None if not isinstance(pm_context, dict) else pm_context.get('target_section')} "
                f"target_xyz={None if not isinstance(pm_context, dict) else pm_context.get('target_xyz')} "
                f"drop_xyz={None if not isinstance(pm_context, dict) else pm_context.get('drop_xyz')} "
                f"outcome={None if not isinstance(pm_context, dict) else pm_context.get('return_outcome')} "
                f"auth_pop={None if not isinstance(pm_context, dict) else pm_context.get('authoritative_pop')}"
            )
            sync_stack_levels_from_authoritative_state()
            success_reason = f"{action_cmd}_success"
            if str(pm_context_row.get("return_outcome", "")).strip().lower() == "released_after_place_soft_fail":
                success_reason = "correction_return_released_after_place_soft_fail"
            record_policy_step(action_cmd, success_reason, progress=True)
            return {
                "handled": True,
                "break_loop": False,
                "centered_pos": None,
                "cube_color": "unknown",
                "color_conf": 0.0,
            }
        state.cycles_without_place_progress += 1
        if isinstance(pm_context, dict):
            print(f"[PickMisplaced] failed reason={pm_reason} {_compact_misplaced_summary(pm_context)}")
        else:
            print(f"[PickMisplaced] failed reason={pm_reason} context=None")
        if bool(pm_context_row.get("released_after_failure", False)):
            state.holding_object = False
            state.current_hold_grip = 0.0
            state.last_pick_measured_xyz = None
            state.active_target_track_id = None
            record_policy_step(
                action_cmd,
                "correction_return_released_after_failure",
                progress=True,
            )
            return {"handled": True, "break_loop": False}
        if str(pm_reason).startswith("misplaced_return_place_failed:"):
            place_reason = str(pm_reason).split(":", 1)[1].strip().lower()
            failure_row = pm_context_row.get("return_place_failure", {})
            failure_is_unrecoverable = bool(
                place_reason == "move_overcurrent_unrecoverable"
                or (
                    isinstance(failure_row, dict)
                    and bool(failure_row.get("unrecoverable", False))
                )
            )
            if bool(failure_is_unrecoverable):
                state.stop_reason = "move_overcurrent_unrecoverable"
                state.skip_final_motion = True
                record_policy_step(
                    action_cmd,
                    "correction_return_drop_failed_while_holding",
                    progress=False,
                )
                return {"handled": True, "break_loop": True}
            if bool(pm_context_row.get("holding_after_failure", False)) or bool(state.holding_object):
                record_policy_step(
                    action_cmd,
                    "correction_return_unavailable_holding_for_planner",
                    progress=False,
                )
                return {"handled": True, "break_loop": False}
            state.stop_reason = "correction_return_drop_failed_state_unknown"
            state.skip_final_motion = True
            record_policy_step(
                action_cmd,
                "correction_return_drop_failed_while_holding",
                progress=False,
            )
            return {"handled": True, "break_loop": True}
        if bool(pm_context_row.get("holding_after_failure", False)) and bool(state.holding_object):
            record_policy_step(
                action_cmd,
                "correction_return_unavailable_holding_for_planner",
                progress=False,
            )
            return {"handled": True, "break_loop": False}
        if action_cmd in {"pick_placed_left", "pick_placed_right", "pick_misplaced_left", "pick_misplaced_right"}:
            try:
                verify_row = verify_v2.verify_last_place_reliability(
                    state=state,
                    arm=arm,
                    per=per,
                    det=det,
                    count_in_stats=False,
                )
                verify_status = str(verify_row.get("status", "unknown")) if isinstance(verify_row, dict) else "unknown"
            except Exception as exc:
                verify_status = f"error:{exc}"
            refresh_status = "skipped_disabled"
            if (
                bool(pick_correction_fail_hydrate_refresh_enabled)
                and callable(run_startup_stack_bootstrap_verify)
                and callable(sync_stack_levels_from_startup_bootstrap)
            ):
                try:
                    print(f"[PickCorrectionFailRefresh] command={action_cmd} hydrate_begin mode=refresh")
                    startup_boot = run_startup_stack_bootstrap_verify(mode="refresh")
                    sync_stack_levels_from_startup_bootstrap(startup_boot)
                    sync_stack_levels_from_authoritative_state()
                    refresh_status = str((startup_boot or {}).get("hydrate_status", "unknown"))
                except Exception as exc:
                    refresh_status = f"error:{type(exc).__name__}:{exc}"
            elif bool(pick_correction_fail_hydrate_refresh_enabled):
                refresh_status = "skipped_missing_bootstrap"
            print(
                f"[PickCorrectionFailRefresh] command={action_cmd} verify_status={verify_status} "
                f"refresh_status={refresh_status}"
            )
        if is_pick_placed_cmd and str(pm_reason) == "no_misplaced_visible":
            preferred_level = 0
            try:
                if preferred_section in {section_left_name, section_right_name}:
                    preferred_level = int(
                        max(0, stack_levels.get(str(preferred_section), 0) or 0)
                    )
            except Exception:
                preferred_level = 0
            if preferred_level <= 0:
                cooldown_steps = max(1, int(pick_placed_empty_cooldown_steps))
                if preferred_section == section_left_name:
                    state.pick_placed_empty_cooldown_left = int(cooldown_steps)
                elif preferred_section == section_right_name:
                    state.pick_placed_empty_cooldown_right = int(cooldown_steps)
            else:
                print(
                    f"[PickPlacedCooldown] skipped side={preferred_section} "
                    f"reason=authoritative_nonempty level={int(preferred_level)}"
                )
        step_reason_code = f"{action_cmd}_fail:{pm_reason}"
        if is_pick_placed_cmd and str(pm_reason) == "no_misplaced_visible":
            step_reason_code = "pick_placed_fail:no_misplaced_visible"
        elif is_pick_placed_cmd and str(pm_reason) == "reacquire_failed_transient":
            step_reason_code = "pick_placed_fail:reacquire_failed_transient"
        record_policy_step(action_cmd, step_reason_code, progress=False)
        break_loop = bool(
            state.skip_final_motion or state.stop_reason == "move_overcurrent_unrecoverable"
        )
        return {"handled": True, "break_loop": break_loop}

    if action_cmd == "return_placed_cube":
        if state.holding_object:
            state.cycles_without_place_progress += 1
            state.invalid_precondition_recoveries += 1
            print("[Policy] return_placed_cube requested while holding object; auto-observing first.")
            record_policy_step(action_cmd, "return_placed_while_holding", progress=False)
            runtime_loop_dispatch.run_auto_recovery_observe(run_observe_action=run_observe_action)
            return {"handled": True, "break_loop": False}
        rpc_ok, rpc_reason, rpc_context = misplaced_actions.execute_return_placed_cube_correction(
            state=state,
            arm=arm,
            det=det,
            per=per,
            stack_levels=stack_levels,
            label_prefix=f"prompted_cycle_{state.cycle_count}",
            safe_pick_reach_m=prompted_safe_pick_reach_m,
        )
        rv = (
            rpc_context.get("return_verify", {})
            if isinstance(rpc_context, dict)
            else {}
        )
        state.last_return_verify = (
            dict(rv) if isinstance(rv, dict) else {"status": "missing", "confirmed": False}
        )
        if rpc_ok:
            state.active_target_track_id = None
            state.cycles_without_place_progress = 0
            sync_stack_levels_from_authoritative_state()
            post_side = "all"
            if isinstance(rpc_context, dict):
                tgt_section = str(rpc_context.get("target_section", "")).strip().lower()
                if tgt_section in {section_left_name, section_right_name}:
                    post_side = tgt_section
            post_return_placed_reconcile = stack_scene.reconcile_scene(
                state=state,
                arm=arm,
                per=per,
                det=det,
                side=str(post_side),
                mode="post_return_placed_release",
                include_pick_rows=False,
            )
            print(
                f"[SceneReconcile] mode=post_return_placed_release status={post_return_placed_reconcile.get('status')} "
                f"drift={bool(post_return_placed_reconcile.get('drift_detected', False))} "
                f"collision_risk={bool(post_return_placed_reconcile.get('collision_risk', False))} "
                f"rev={int(post_return_placed_reconcile.get('scene_revision', state.scene_revision))}"
            )
            print(
                f"[ReturnPlaced] success object_id="
                f"{None if not isinstance(rpc_context, dict) else rpc_context.get('object_id')} "
                f"section={None if not isinstance(rpc_context, dict) else rpc_context.get('target_section')} "
                f"target_xyz={None if not isinstance(rpc_context, dict) else rpc_context.get('target_xyz')} "
                f"pick_origin_xyz={None if not isinstance(rpc_context, dict) else rpc_context.get('pick_origin_xyz')} "
                f"auth_pop={None if not isinstance(rpc_context, dict) else rpc_context.get('authoritative_pop')}"
            )
            record_policy_step(action_cmd, "return_placed_cube_success", progress=True)
            return {
                "handled": True,
                "break_loop": False,
                "centered_pos": None,
                "cube_color": "unknown",
                "color_conf": 0.0,
            }
        state.cycles_without_place_progress += 1
        print(
            f"[ReturnPlaced] failed reason={rpc_reason} "
            f"context={None if not isinstance(rpc_context, dict) else {k: rpc_context.get(k) for k in ['object_id', 'target_section', 'target_xyz', 'pick_origin_xyz', 'lock_track_id', 'lock_uv']}}"
        )
        record_policy_step(action_cmd, f"return_placed_cube_fail:{rpc_reason}", progress=False)
        break_loop = bool(
            state.skip_final_motion or state.stop_reason == "move_overcurrent_unrecoverable"
        )
        return {"handled": True, "break_loop": break_loop}

    return {"handled": False, "break_loop": False}
