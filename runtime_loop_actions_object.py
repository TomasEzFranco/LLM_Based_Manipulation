"""Object-action handlers extracted from runtime_loop (behavior-preserving)."""

from __future__ import annotations

import numpy as np

import pick_actions
import runtime_loop_dispatch
import runtime_loop_observe
import verify_v2


def handle_object_action(
    *,
    action_cmd: str,
    state,
    arm,
    per,
    det,
    centered_pos: tuple[int, int] | None,
    cube_color: str,
    color_conf: float,
    hold_grip: float,
    carry_supervisor,
    track_enable: bool,
    prompted_safe_pick_reach_m: float,
    push_min_progress_m: float,
    nearest_visible_track_by_uv_fn,
    choose_track_candidate_near_uv_fn,
    classify_cube_color_patch_fn,
    estimate_base_xyz_from_uv_fast_fn,
    execute_push_cube_action_fn,
    record_policy_step,
    run_observe_action,
    capture_pick_lock_snapshot_fn,
) -> dict:
    action_cmd = str(action_cmd)
    if action_cmd == "verify_last_place":
        verify_result = verify_v2.verify_last_place_reliability(
            state=state,
            arm=arm,
            per=per,
            det=det,
            count_in_stats=False,
        )
        verify_status = str(verify_result.get("status", "unknown"))
        verify_progress = bool(verify_result.get("confirmed", False))
        record_policy_step(action_cmd, f"verify:{verify_status}", progress=verify_progress)
        return {"handled": True, "break_loop": False}

    if action_cmd == "classify_cube":
        if centered_pos is None:
            state.cycles_without_place_progress += 1
            state.invalid_precondition_recoveries += 1
            print("[Policy] classify requested without centered target; auto-observing first.")
            record_policy_step(action_cmd, "classify_without_center", progress=False)
            runtime_loop_dispatch.run_auto_recovery_observe(run_observe_action=run_observe_action)
            return {"handled": True, "break_loop": False}
        prev_color = str(cube_color)
        prev_conf = float(color_conf)
        try:
            color_frame, depth_frame = per.get_frames()
        except Exception as exc:
            state.cycles_without_place_progress += 1
            state.invalid_precondition_recoveries += 1
            print(f"[Color] frame unavailable for classify; auto-observing first ({exc})")
            record_policy_step(action_cmd, "classify_frame_unavailable", progress=False)
            runtime_loop_dispatch.run_auto_recovery_observe(run_observe_action=run_observe_action)
            return {"handled": True, "break_loop": False}
        img_now = np.asanyarray(color_frame.get_data())
        _img_ann, candidates = det.detect_candidates_and_draw(img_now, draw=False)
        lock_u = int(centered_pos[0])
        lock_v = int(centered_pos[1])
        track_id_for_color = (
            None if state.active_target_track_id is None else int(state.active_target_track_id)
        )
        if track_id_for_color is None and track_enable:
            linked_tid = nearest_visible_track_by_uv_fn(
                state,
                lock_u,
                lock_v,
                max_dist_px=90.0,
            )
            if linked_tid is not None:
                track_id_for_color = int(linked_tid)
        target = choose_track_candidate_near_uv_fn(
            candidates,
            track_id_for_color,
            lock_u,
            lock_v,
            min_conf=0.0,
        )
        if target is not None and track_id_for_color is not None:
            cube_color, color_conf = classify_cube_color_patch_fn(
                img_now,
                bbox_xyxy=target.get("bbox_xyxy", None),
                center_uv=None,
                bbox_core_ratio=0.55,
            )
        else:
            print(
                "[Color] track_bbox_missing_for_classify "
                f"track_id={None if track_id_for_color is None else int(track_id_for_color)} "
                f"uv=({int(lock_u)},{int(lock_v)})"
            )
            cube_color, color_conf = "unknown", 0.0
        print(f"[Color] centered cube classified as {cube_color} (conf={color_conf:.3f})")
        state.pick_other_block_track_id = (
            None if state.active_target_track_id is None else int(state.active_target_track_id)
        )
        state.pick_other_block_uv = [int(centered_pos[0]), int(centered_pos[1])]
        xyz_seed = estimate_base_xyz_from_uv_fast_fn(
            arm=arm,
            per=per,
            depth_frame=depth_frame,
            u=int(centered_pos[0]),
            v=int(centered_pos[1]),
        )
        xyz_arr = np.array(xyz_seed, dtype=float).reshape(-1)
        if xyz_arr.size >= 3 and np.all(np.isfinite(xyz_arr[:3])):
            state.pick_other_block_xyz = [float(xyz_arr[0]), float(xyz_arr[1]), float(xyz_arr[2])]
        else:
            state.pick_other_block_xyz = None
        state.pick_other_block_track_ids = []
        state.pick_other_block_xyzs = []
        state.pick_other_block_uvs = []
        state.pick_other_block_source = "classify"
        print(
            f"[PickOtherSeed] source=classify track={state.pick_other_block_track_id} "
            f"uv={state.pick_other_block_uv} xyz={state.pick_other_block_xyz}"
        )
        runtime_loop_observe.capture_pick_lock_snapshot(
            state=state,
            centered_pos=centered_pos,
            cube_color=str(cube_color),
            color_conf=float(color_conf),
            arm=arm,
            source="classify",
        )
        classify_progress = (
            (cube_color in {"orange", "blue"} and prev_color == "unknown")
            or (cube_color != prev_color)
            or (float(color_conf) > (prev_conf + 0.05))
        )
        if cube_color in {"orange", "blue"}:
            classify_reason = f"classify_success:{cube_color}:{float(color_conf):.3f}"
        else:
            classify_reason = "classify_no_change"
        record_policy_step(action_cmd, classify_reason, progress=classify_progress)
        return {
            "handled": True,
            "break_loop": False,
            "cube_color": str(cube_color),
            "color_conf": float(color_conf),
        }

    if action_cmd == "grasp_cube":
        if centered_pos is None:
            state.cycles_without_place_progress += 1
            state.invalid_precondition_recoveries += 1
            print("[Policy] grasp requested without centered target; auto-observing first.")
            record_policy_step(action_cmd, "grasp_without_center", progress=False)
            runtime_loop_dispatch.run_auto_recovery_observe(run_observe_action=run_observe_action)
            return {"handled": True, "break_loop": False}
        carry_status, hold_grip, carry_supervisor = pick_actions.run_grasp_and_carry_common(
            state=state,
            arm=arm,
            per=per,
            centered_pos=centered_pos,
            label_prefix=f"prompted_cycle_{state.cycle_count}",
            safe_pick_reach_m=prompted_safe_pick_reach_m,
        )
        if carry_status == "retry":
            record_policy_step(action_cmd, "grasp_retry", progress=False)
            return {
                "handled": True,
                "break_loop": True,
                "hold_grip": float(hold_grip),
                "carry_supervisor": carry_supervisor,
            }
        if carry_status == "stop":
            record_policy_step(action_cmd, "grasp_stop", progress=False)
            return {
                "handled": True,
                "break_loop": True,
                "hold_grip": float(hold_grip),
                "carry_supervisor": carry_supervisor,
            }
        record_policy_step(action_cmd, "grasp_success", progress=True)
        return {
            "handled": True,
            "break_loop": False,
            "hold_grip": float(hold_grip),
            "carry_supervisor": carry_supervisor,
        }

    if action_cmd == "push_cube":
        if state.holding_object:
            state.cycles_without_place_progress += 1
            state.invalid_precondition_recoveries += 1
            print("[Policy] push requested while holding object; auto-observing first.")
            record_policy_step(action_cmd, "push_while_holding", progress=False)
            runtime_loop_dispatch.run_auto_recovery_observe(run_observe_action=run_observe_action)
            return {"handled": True, "break_loop": False}
        push_ok, push_reason, push_context = execute_push_cube_action_fn(
            state=state,
            arm=arm,
            det=det,
            per=per,
            centered_pos=centered_pos,
            label_prefix=f"prompted_cycle_{state.cycle_count}",
        )
        if push_ok:
            state.active_target_track_id = None
            state.cycles_without_place_progress = 0
            print(
                f"[Push] success reason={push_reason} "
                f"context={None if not isinstance(push_context, dict) else {k: push_context.get(k) for k in ['status', 'steps', 'distance_before_m', 'distance_after_m', 'progress_m', 'start_xyz', 'end_xyz', 'target_xyz']}}"
            )
            record_policy_step(action_cmd, f"push_success:{push_reason}", progress=True, feedback_context=push_context)
            return {
                "handled": True,
                "break_loop": False,
                "centered_pos": None,
                "cube_color": "unknown",
                "color_conf": 0.0,
            }
        if push_reason == "partial_timeout":
            progress_m = 0.0
            if isinstance(push_context, dict):
                try:
                    progress_m = float(push_context.get("progress_m", 0.0) or 0.0)
                except Exception:
                    progress_m = 0.0
            had_progress = bool(progress_m >= float(push_min_progress_m))
            if had_progress:
                state.cycles_without_place_progress = 0
            else:
                state.cycles_without_place_progress += 1
            state.active_target_track_id = None
            print(
                f"[Push] partial timeout progress_m={progress_m:.3f} "
                f"(threshold={float(push_min_progress_m):.3f}) "
                f"context={None if not isinstance(push_context, dict) else {k: push_context.get(k) for k in ['steps', 'distance_before_m', 'distance_after_m', 'start_xyz', 'end_xyz', 'target_xyz']}}"
            )
            record_policy_step(action_cmd, "push_partial_timeout", progress=had_progress, feedback_context=push_context)
            return {
                "handled": True,
                "break_loop": False,
                "centered_pos": None,
                "cube_color": "unknown",
                "color_conf": 0.0,
            }
        state.cycles_without_place_progress += 1
        if push_reason == "move_overcurrent_unrecoverable":
            state.stop_reason = push_reason
            state.skip_final_motion = True
            record_policy_step(action_cmd, f"push_fail:{push_reason}", progress=False, feedback_context=push_context)
            return {"handled": True, "break_loop": True}
        print(
            f"[Push] failed reason={push_reason} "
            f"context={None if not isinstance(push_context, dict) else {k: push_context.get(k) for k in ['status', 'steps', 'distance_before_m', 'distance_after_m', 'start_xyz', 'end_xyz', 'target_xyz', 'stage', 'motion_reason', 'blocked']}}"
        )
        record_policy_step(action_cmd, f"push_fail:{push_reason}", progress=False, feedback_context=push_context)
        return {"handled": True, "break_loop": False}

    return {"handled": False, "break_loop": False}
