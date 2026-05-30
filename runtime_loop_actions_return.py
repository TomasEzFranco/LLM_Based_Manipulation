"""Return-action handlers extracted from runtime_loop (behavior-preserving)."""

from __future__ import annotations

import numpy as np

import misplaced_actions
import place_actions
import runtime_loop_dispatch
import stack_scene


def handle_return_cube_action(
    *,
    action_cmd: str,
    state,
    arm,
    det,
    per,
    hold_grip: float,
    carry_supervisor,
    cycle_count: int,
    section_left_name: str,
    section_right_name: str,
    home_pose,
    clamp_grip_cmd_fn,
    record_policy_step,
    run_observe_action,
    capture_pick_lock_snapshot_fn,
) -> dict:
    action_cmd = str(action_cmd)
    if action_cmd != "return_cube":
        return {"handled": False, "break_loop": False}

    if not state.holding_object:
        state.cycles_without_place_progress += 1
        state.invalid_precondition_recoveries += 1
        print("[Policy] return requested without object; auto-observing first.")
        record_policy_step(action_cmd, "return_without_object", progress=False)
        runtime_loop_dispatch.run_auto_recovery_observe(run_observe_action=run_observe_action)
        return {"handled": True, "break_loop": False}

    return_ok, return_reason, return_context = place_actions.execute_return_cube_action(
        state=state,
        arm=arm,
        det=det,
        per=per,
        hold_grip=hold_grip,
        carry_supervisor=carry_supervisor,
    )
    if return_ok:
        return_target_xyz = None
        if isinstance(return_context, dict):
            rt_xyz = return_context.get("target_xyz", None)
            if isinstance(rt_xyz, (list, tuple)) and len(rt_xyz) >= 3:
                try:
                    arr = np.array([float(rt_xyz[0]), float(rt_xyz[1]), float(rt_xyz[2])], dtype=float).reshape(-1)
                except (TypeError, ValueError):
                    arr = np.array([np.nan, np.nan, np.nan], dtype=float)
                if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                    return_target_xyz = [float(arr[0]), float(arr[1]), float(arr[2])]
        return_flow = misplaced_actions.run_return_verify_and_handoff_session(
            state=state,
            arm=arm,
            per=per,
            det=det,
            label_prefix=f"prompted_cycle_{int(cycle_count)}",
            return_target_xyz=return_target_xyz,
        )
        rv = return_flow.get("return_verify", {}) if isinstance(return_flow, dict) else {}
        rv_status = str(rv.get("status", "unknown"))
        rv_confirmed = bool(rv.get("confirmed", False))
        rv_hits = int(rv.get("hits", 0) or 0)
        rv_samples = int(rv.get("samples", 0) or 0)
        rv_xy = rv.get("xy_error_m", float("inf"))
        rv_z = rv.get("z_error_m", float("inf"))
        state.last_return_verify = dict(rv) if isinstance(rv, dict) else {"status": "missing", "confirmed": False}
        print(
            f"[ReturnVerify] status={rv_status} confirmed={rv_confirmed} "
            f"hits={rv_hits}/{rv_samples} err_xy={float(rv_xy):.3f} err_z={float(rv_z):.3f}"
        )

        returned_verified = bool(return_flow.get("returned_verified", False))
        if returned_verified:
            blocked_tid = return_flow.get("blocked_return_track_id", None)
            blocked_uv = return_flow.get("blocked_return_uv", None)
            blocked_xyz = return_flow.get("blocked_return_xyz", None)
            state.pick_other_block_track_id = (None if blocked_tid is None else int(blocked_tid))
            state.pick_other_block_uv = (
                None
                if not isinstance(blocked_uv, (list, tuple)) or len(blocked_uv) < 2
                else [int(blocked_uv[0]), int(blocked_uv[1])]
            )
            state.pick_other_block_xyz = (
                None
                if not isinstance(blocked_xyz, (list, tuple)) or len(blocked_xyz) < 3
                else [float(blocked_xyz[0]), float(blocked_xyz[1]), float(blocked_xyz[2])]
            )
            state.pick_other_block_track_ids = []
            state.pick_other_block_xyzs = []
            state.pick_other_block_uvs = []
            state.pick_other_block_source = "return"
            print(
                f"[ReturnBan] verified track_id={state.pick_other_block_track_id} "
                f"uv={state.pick_other_block_uv} xyz={state.pick_other_block_xyz}"
            )
        else:
            state.pick_other_block_track_id = None
            state.pick_other_block_uv = None
            state.pick_other_block_xyz = None
            state.pick_other_block_track_ids = []
            state.pick_other_block_xyzs = []
            state.pick_other_block_uvs = []
            state.pick_other_block_source = "none"
            print(
                f"[ReturnBan] skipped add; return verify not confirmed "
                f"(status={rv_status})"
            )

        state.holding_object = False
        state.current_hold_grip = 0.0
        state.last_pick_return_xyz = None
        state.last_pick_measured_xyz = None
        state.returned_count += 1
        state.cycles_without_place_progress = 0
        post_return_reconcile = stack_scene.reconcile_scene(
            state=state,
            arm=arm,
            per=per,
            det=det,
            side="all",
            mode="post_return_release",
            include_pick_rows=True,
        )
        print(
            f"[SceneReconcile] mode=post_return_release status={post_return_reconcile.get('status')} "
            f"drift={bool(post_return_reconcile.get('drift_detected', False))} "
            f"collision_risk={bool(post_return_reconcile.get('collision_risk', False))} "
            f"rev={int(post_return_reconcile.get('scene_revision', state.scene_revision))}"
        )
        print(
            f"[Return] cube returned to pick target xyz="
            f"{None if return_context is None else return_context.get('target_xyz')}"
        )
        next_centered_uv = return_flow.get("next_centered_uv", None)
        next_track_id = return_flow.get("next_track_id", None)
        next_color = str(return_flow.get("next_color", "unknown")).strip().lower()
        next_color_conf = float(return_flow.get("next_color_conf", 0.0) or 0.0)
        if isinstance(next_centered_uv, (list, tuple)) and len(next_centered_uv) >= 2:
            centered_pos = (int(next_centered_uv[0]), int(next_centered_uv[1]))
            cube_color = (next_color if next_color in {"orange", "blue"} else "unknown")
            color_conf = (next_color_conf if next_color in {"orange", "blue"} else 0.0)
            state.active_target_track_id = (
                None if next_track_id is None else int(next_track_id)
            )
            # Critical: refresh pre-grasp lock snapshot to the new handoff target.
            # Without this, the next grasp can restore stale lock data from before return,
            # causing recenter/classify to drift back to the previous cube.
            capture_pick_lock_snapshot_fn("return_handoff_lock")
            print(
                f"[ReturnHandoff] locked next track_id={state.active_target_track_id} "
                f"uv={centered_pos} color={cube_color} conf={float(color_conf):.3f}"
            )
            record_policy_step(action_cmd, "return_success_handoff_locked", progress=True)
            return {
                "handled": True,
                "break_loop": False,
                "centered_pos": centered_pos,
                "cube_color": str(cube_color),
                "color_conf": float(color_conf),
            }
        centered_pos = None
        cube_color = "unknown"
        color_conf = 0.0
        print(
            "[ReturnHandoff] no alternate target lock yet; awaiting observe/reobserve."
        )
        record_policy_step(action_cmd, "return_success_no_next_target", progress=False)
        return {
            "handled": True,
            "break_loop": False,
            "centered_pos": centered_pos,
            "cube_color": str(cube_color),
            "color_conf": float(color_conf),
        }

    state.cycles_without_place_progress += 1
    print(f"[Return] failed reason={return_reason}")
    if return_reason == "return_drop_occupied":
        record_policy_step(action_cmd, f"return_fail:{return_reason}", progress=False)
        return {"handled": True, "break_loop": False}
    if return_reason == "move_overcurrent_unrecoverable":
        state.stop_reason = return_reason
        state.skip_final_motion = True
        record_policy_step(action_cmd, f"return_fail:{return_reason}", progress=False)
        return {"handled": True, "break_loop": True}
    fail_home = home_pose.copy()
    fail_home[3] = clamp_grip_cmd_fn(hold_grip)
    arm.goto_task_space(
        fail_home,
        duration=1.2,
        label=f"prompted_cycle_{int(cycle_count)}_home_after_return_fail",
    )
    state.stop_reason = f"return_failed:{return_reason}"
    record_policy_step(action_cmd, f"return_fail:{return_reason}", progress=False)
    return {"handled": True, "break_loop": True}
