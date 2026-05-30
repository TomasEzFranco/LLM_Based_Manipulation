#!/usr/bin/env python3
from __future__ import annotations

import math
import numpy as np
import time

_CORE_BIND_READY = False


def _bind_core_globals() -> None:
    global _CORE_BIND_READY
    import runtime_core as core
    protected = {
        '_bind_core_globals', '_CORE_BIND_READY',
        'acquire_and_center_intended_cube', 'run_pick_center_cycle',
        'run_pick_other_session', 'run_grasp_and_carry_common',
        'retreat_after_correction_drop', 'goto_correction_drop_transit',
    }
    for name, value in core.__dict__.items():
        if name.startswith('__') or name in protected:
            continue
        globals()[name] = value
    _CORE_BIND_READY = True

def acquire_and_center_intended_cube(
    state: CycleState,
    arm: Arm,
    per: Perception,
    det: YOLODetector,
    section_groups: dict[str, list[int]] | None,
    label_prefix: str,
    blocked_track_ids: set[int] | None = None,
    blocked_xyzs: list[list[float]] | None = None,
    blocked_uvs: list[list[int]] | None = None,
) -> tuple[str, tuple[int, int] | None, int | None]:
    _bind_core_globals()
    _ = label_prefix
    blocked_track_ids = set() if blocked_track_ids is None else {int(x) for x in blocked_track_ids}
    blocked_xyzs = [] if blocked_xyzs is None else [list(x) for x in blocked_xyzs if isinstance(x, (list, tuple)) and len(x) >= 3]
    blocked_uvs = [] if blocked_uvs is None else [list(x) for x in blocked_uvs if isinstance(x, (list, tuple)) and len(x) >= 2]
    # Prime track memory with one frame before active centering.
    if TRACK_ENABLE:
        obs0 = observe_scene_frame(
            det=det,
            arm=arm,
            per=per,
            draw=False,
            projected_min_conf=float(TRACK_MIN_CONF),
        )
        if obs0 is not None:
            update_cube_tracks(
                state=state,
                detections=obs0.projected_rows,
                max_miss_frames=TRACK_MAX_MISS_FRAMES,
                image_center_uv=obs0.image_center_uv,
            )
        target_track_id = select_intended_track_for_pick(
            state=state,
            section_groups=section_groups,
            blocked_track_ids=blocked_track_ids,
            blocked_xyzs=blocked_xyzs,
        )
    else:
        target_track_id = None

    if target_track_id is not None:
        print(f"[Track] targeting track_id={target_track_id} for pick centering.")
    elif blocked_track_ids or blocked_xyzs or blocked_uvs:
        print(
            f"[Track] pick_other requested; blocked_track_ids={sorted(blocked_track_ids)} "
            f"blocked_xyzs={len(blocked_xyzs)} blocked_uvs={len(blocked_uvs)}"
        )

    def _selector(candidates: list[dict], view_cx: int, view_cy: int, context: dict) -> dict | None:
        target_tid = context.get("target_track_id", None)
        blocked_ids = context.get("blocked_track_ids", set())
        if not isinstance(blocked_ids, set):
            try:
                blocked_ids = {int(x) for x in blocked_ids}
            except Exception:
                blocked_ids = set()
        blocked_xyzs_norm: list[np.ndarray] = []
        for xyz in blocked_xyzs:
            try:
                arr = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2])], dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                blocked_xyzs_norm.append(arr[:3].copy())

        blocked_uvs_norm: list[tuple[int, int]] = []
        for uv in blocked_uvs:
            try:
                uu = int(uv[0])
                vv = int(uv[1])
            except (TypeError, ValueError, IndexError):
                continue
            blocked_uvs_norm.append((uu, vv))

        def _is_uv_blocked(u_px: int, v_px: int) -> bool:
            if not blocked_uvs_norm:
                return False
            r2 = float(max(0.0, float(PICK_OTHER_BLOCK_UV_PX))) ** 2
            for bu, bv in blocked_uvs_norm:
                du = float(int(u_px) - int(bu))
                dv = float(int(v_px) - int(bv))
                if (du * du + dv * dv) <= r2:
                    return True
            return False

        def _is_track_xyz_blocked(track_row: dict | None) -> bool:
            if track_row is None or not blocked_xyzs_norm:
                return False
            xyz = np.array(track_row.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
            if xyz.size < 3 or not np.all(np.isfinite(xyz[:3])):
                return False
            for bxyz in blocked_xyzs_norm:
                d_xy = float(math.hypot(float(xyz[0]) - float(bxyz[0]), float(xyz[1]) - float(bxyz[1])))
                d_z = float(abs(float(xyz[2]) - float(bxyz[2])))
                if d_xy <= float(PICK_OTHER_BLOCK_XY_M) and d_z <= float(PICK_OTHER_BLOCK_Z_M):
                    return True
            return False

        def _find_track_candidate(rows: list[dict], tid: int) -> dict | None:
            matches: list[dict] = []
            for row in rows:
                conf = float(row.get("conf", 0.0))
                if conf < float(DETECT_CONF):
                    continue
                row_tid = _candidate_track_id_or_none(row)
                if row_tid is None or int(row_tid) != int(tid):
                    continue
                item = dict(row)
                item["u"] = int(row.get("u", 0))
                item["v"] = int(row.get("v", 0))
                if _is_uv_blocked(int(item["u"]), int(item["v"])):
                    continue
                item["conf"] = conf
                item["track_id"] = int(row_tid)
                matches.append(item)
            if not matches:
                return None
            matches.sort(key=lambda r: (-float(r.get("conf", 0.0)), int(r.get("v", 0))))
            return matches[0]

        def _select_stack_top_candidate(rows: list[dict]) -> tuple[dict | None, dict]:
            xyz_blocked_ids: set[int] = set()
            if TRACK_ENABLE and blocked_xyzs_norm and state.track_memory:
                for row in rows:
                    row_tid = _candidate_track_id_or_none(row)
                    if row_tid is None:
                        continue
                    track_row = state.track_memory.get(int(row_tid), None)
                    if _is_track_xyz_blocked(track_row):
                        xyz_blocked_ids.add(int(row_tid))
            effective_blocked_ids = set(blocked_ids)
            effective_blocked_ids.update(xyz_blocked_ids)
            blocked_uv_rows = [[int(uv[0]), int(uv[1])] for uv in blocked_uvs_norm]
            selected, meta = select_pick_candidate_stack_top(
                candidates=rows,
                cx=int(view_cx),
                cy=int(view_cy),
                min_conf=float(DETECT_CONF),
                require_track_id=bool(TRACK_ENABLE),
                blocked_track_ids=effective_blocked_ids,
                blocked_uvs=blocked_uv_rows,
                top_exposed_only=bool(PICK_TOP_EXPOSED_ONLY),
                x_overlap_min=float(PICK_TOP_EXPOSED_X_OVERLAP_MIN),
                y_gap_px=float(PICK_TOP_EXPOSED_Y_GAP_PX),
                fallback_mode=str(PICK_TOP_EXPOSED_FALLBACK),
            )
            meta = dict(meta or {})
            meta["xyz_blocked_ids"] = sorted([int(tid) for tid in xyz_blocked_ids])
            return selected, meta

        # For stacked scenes, always re-evaluate top-exposed candidates each frame.
        # This avoids sticky lock behavior on lower cubes when a higher cube is visible.
        if bool(PICK_TOP_EXPOSED_ONLY):
            target_tid = None

        if target_tid is not None and TRACK_ENABLE:
            if int(target_tid) in blocked_ids:
                target_tid = None
            else:
                target_row = state.track_memory.get(int(target_tid), None) if state.track_memory else None
                if _is_track_xyz_blocked(target_row):
                    target_tid = None
                else:
                    locked = _find_track_candidate(candidates, int(target_tid))
                    if locked is not None:
                        context["selector_meta"] = {
                            "selector_mode": "track_lock",
                            "candidate_count": int(len(candidates)),
                            "conf_considered": 0,
                            "eligible_count": 1,
                            "exposed_top_count": 1,
                            "fallback_used": False,
                            "selected_track_id": int(locked.get("track_id")),
                            "selected_uv": [int(locked.get("u", 0)), int(locked.get("v", 0))],
                        }
                        return locked
        ranked, selector_meta = _select_stack_top_candidate(candidates)
        context["selector_meta"] = dict(selector_meta or {})
        if ranked is None:
            return None
        ranked_tid = _candidate_track_id_or_none(ranked)
        if TRACK_ENABLE and ranked_tid is not None and state.track_memory:
            ranked_row = state.track_memory.get(int(ranked_tid), None)
            if _is_track_xyz_blocked(ranked_row):
                return None
        return ranked

    centered_pos = center_object_slowly(
        det=det,
        arm=arm,
        per=per,
        required_centered_frames=CENTERED_FRAMES_REQUIRED,
        show_window=SHOW_WINDOW,
        save_localization_dir=(LOCALIZATION_CAPTURE_ROOT if AUTO_CAPTURE_LOCALIZATION_IMAGES else None),
        save_tag=f"{label_prefix}_{state.cycle_count:03d}",
        save_annotated=LOCALIZATION_CAPTURE_SAVE_ANNOTATED,
        candidate_selector=_selector,
        selector_context={"target_track_id": target_track_id, "blocked_track_ids": blocked_track_ids},
        state=state,
    )
    if centered_pos is None:
        return "retry", None, target_track_id
    if state is not None and TRACK_ENABLE:
        linked_track_id = nearest_visible_track_by_uv(state, centered_pos[0], centered_pos[1], max_dist_px=100.0)
        if linked_track_id is not None:
            state.active_target_track_id = int(linked_track_id)
    return "ok", centered_pos, target_track_id

def run_pick_center_cycle(
    state: CycleState,
    arm: Arm,
    per: Perception,
    det: YOLODetector,
    label_prefix: str,
    section_groups: dict[str, list[int]] | None = None,
    blocked_track_ids: set[int] | None = None,
    blocked_xyzs: list[list[float]] | None = None,
    blocked_uvs: list[list[int]] | None = None,
) -> tuple[str, tuple[int, int] | None]:
    _bind_core_globals()
    print("[Pick] Moving to PICK_LOOKING...")
    arm.goto_task_space(PICK_LOOKING, duration=1.2, label=f"{label_prefix}_pick_look")
    time.sleep(0.2)
    print(f"[Counts] Picked={state.picked_count} | Placed={state.placed_count}")
    print(
        f"[Pick] Scanning/centering for cube "
        f"(no_detection_timeout={CENTER_TIMEOUT_NO_DETECTION_S:.1f}s, "
        f"active_timeout={CENTER_TIMEOUT_ACTIVE_S:.1f}s)..."
    )
    print("[Pick] Centering selected cube...")
    _acq_status, centered_pos, target_track_id = acquire_and_center_intended_cube(
        state=state,
        arm=arm,
        per=per,
        det=det,
        section_groups=section_groups,
        label_prefix=label_prefix,
        blocked_track_ids=blocked_track_ids,
        blocked_xyzs=blocked_xyzs,
        blocked_uvs=blocked_uvs,
    )
    if target_track_id is not None:
        print(f"[Track] active_target_track_id={target_track_id}")
    if centered_pos is None:
        state.active_target_track_id = None
        center_failure = getattr(state, "last_center_failure", {}) or {}
        center_status = str(center_failure.get("status", "unknown")).strip().lower()
        selector_meta = center_failure.get("selector_meta", {})
        if not isinstance(selector_meta, dict):
            selector_meta = {}
        candidate_count = int(center_failure.get("candidate_count", 0) or 0)
        filtered_count = int(center_failure.get("filtered_count", 0) or 0)
        eligible_count = int(selector_meta.get("eligible_count", 0) or 0)
        selected_tid = selector_meta.get("selected_track_id", None)
        selected_uv = selector_meta.get("selected_uv", None)
        if center_status == "active_detection_timeout":
            state.recent_pick_active_miss_count += 1
            print(
                f"[ObserveMiss] cause=active_detection_timeout raw={candidate_count} "
                f"filtered={filtered_count} eligible={eligible_count} "
                f"selected_tid={selected_tid} selected_uv={selected_uv} "
                f"empty_count_unchanged={int(state.no_pick_miss_count)}/{int(EMPTY_SCENE_CONFIRM_PASSES)} "
                f"recent_grasp_fails={int(getattr(state, 'recent_grasp_fail_count', 0))} "
                f"active_misses={int(state.recent_pick_active_miss_count)}"
            )
            if (
                int(getattr(state, "recent_grasp_fail_count", 0))
                >= int(PICK_ORIENTATION_CHECK_AFTER_GRASP_FAILS)
                and int(state.recent_pick_active_miss_count)
                >= int(PICK_ORIENTATION_CHECK_AFTER_ACTIVE_MISSES)
            ):
                last_uv = getattr(state, "last_picked_uv", None)
                last_tid = getattr(state, "last_picked_track_id", None)
                last_xyz = getattr(state, "pick_other_block_xyz", None)
                print(
                    "[PickSpaceOrientationCheck] recurrent failed grasp followed by visible-target "
                    f"observe miss; check returned/pick-space cube orientation; action=warn_continue. "
                    f"grasp_fails={int(state.recent_grasp_fail_count)} "
                    f"active_misses={int(state.recent_pick_active_miss_count)} "
                    f"last_grasp_reason={getattr(state, 'last_grasp_fail_reason', '')} "
                    f"last_track={last_tid} last_uv={last_uv} last_xyz={last_xyz} "
                    f"blocked_track={getattr(state, 'pick_other_block_track_id', None)} "
                    f"blocked_xyz={getattr(state, 'pick_other_block_xyz', None)}"
                )
            state.cycles_without_place_progress += 1
            return "retry", None
        state.no_pick_miss_count += 1
        state.recent_pick_active_miss_count = 0
        print(
            f"[Pick] miss confirmation {state.no_pick_miss_count}/{EMPTY_SCENE_CONFIRM_PASSES} "
            f"(no centered lock in PICK_LOOKING scan; cause={center_status or 'unknown'})."
        )
        if state.no_pick_miss_count >= EMPTY_SCENE_CONFIRM_PASSES:
            state.stop_reason = "pick scene empty by confirmation passes"
            print("\nNo valid cube after required confirm passes. Scene considered empty.")
            return "stop", None
        state.cycles_without_place_progress += 1
        return "retry", None
    state.no_pick_miss_count = 0
    state.recent_pick_active_miss_count = 0
    return "ok", centered_pos


# ============================= Tracking / cube identity =============================
def run_pick_other_session(
    state: CycleState,
    arm: Arm,
    per: Perception | None,
    det: YOLODetector | None,
    label_prefix: str,
    blocked_track_id: int | None = None,
    blocked_xyz: list[float] | None = None,
    blocked_uv: list[int] | None = None,
    blocked_track_ids_extra: set[int] | list[int] | None = None,
    blocked_xyzs_extra: list[list[float]] | None = None,
    reject_same_color: str | None = None,
    status_prefix: str = "pick_other",
    log_prefix: str = "[PickOther]",
    enforce_quality_gate: bool = False,
    required_hits: int | None = None,
    measurement_samples: int | None = None,
) -> dict:
    _bind_core_globals()
    reject_cap = max(1, int(PICK_OTHER_MAX_REJECTS))
    hard_timeout_s = max(1.0, float(PICK_OTHER_HARD_TIMEOUT_S))
    xy_margin_m = float(PLACE_VERIFY_V2_XY_MARGIN_M)
    z_margin_m = float(PLACE_VERIFY_V2_Z_MARGIN_M)
    print(f"{log_prefix} Starting verify-style alternate candidate session...")
    print("[Pick] Moving to PICK_LOOKING...")
    arm.goto_task_space(PICK_LOOKING, duration=1.2, label=f"{label_prefix}_pick_other_look")
    time.sleep(0.2)
    blocked_xyz_text = None
    if isinstance(blocked_xyz, (list, tuple)) and len(blocked_xyz) >= 3:
        try:
            blocked_xyz_text = [float(blocked_xyz[0]), float(blocked_xyz[1]), float(blocked_xyz[2])]
        except (TypeError, ValueError):
            blocked_xyz_text = None
    required_hits_local = max(1, int(required_hits if required_hits is not None else PLACE_VERIFY_V2_MIN_HITS))
    measurement_samples_local = max(
        1,
        int(measurement_samples if measurement_samples is not None else PICK_OTHER_VALIDATE_SAMPLES),
    )
    reject_color_norm = str(reject_same_color or "").strip().lower()
    if reject_color_norm not in {"orange", "blue"}:
        reject_color_norm = ""
    print(
        f"{log_prefix} seed blocked track_id={blocked_track_id} xyz={blocked_xyz_text} uv={blocked_uv} (uv_ignored) "
        f"quality_gate={bool(enforce_quality_gate)} required_hits={required_hits_local} "
        f"measure_samples={measurement_samples_local} reject_same_color={reject_color_norm or 'off'} "
        f"hard_timeout={float(hard_timeout_s):.1f}s "
        f"no_candidate_timeout={float(TRACK_HANDOFF_NO_CANDIDATE_TIMEOUT_S):.1f}s "
        f"extra_block_tracks={len(blocked_track_ids_extra or [])} "
        f"extra_block_xyzs={len(blocked_xyzs_extra or [])}"
    )

    def _on_locked_candidate(
        *,
        obs: SceneObservation,
        selected_row: dict,
        track_id: int,
        collect_track_measurement,
        distance_to_blocked_xyz,
        blocked_xyzs,
        xy_margin_m: float,
        z_margin_m: float,
    ) -> dict:
        measure = collect_track_measurement(
            track_id=int(track_id),
            first_obs=obs,
            first_candidate=selected_row,
            sample_count_override=int(measurement_samples_local),
        )
        hits = int(measure.get("hits", 0))
        selected_xyz = measure.get("median_xyz")
        d_xy_block, d_z_block = distance_to_blocked_xyz(selected_xyz)
        xyz_arr = np.array(
            selected_xyz if isinstance(selected_xyz, (list, tuple, np.ndarray)) else [np.nan, np.nan, np.nan],
            dtype=float,
        ).reshape(-1)
        xyz_finite = bool(xyz_arr.size >= 3 and np.all(np.isfinite(xyz_arr[:3])))
        quality_ok = bool(
            xyz_finite
            and hits >= int(required_hits_local)
        )
        is_near_blocked = bool(
            blocked_xyzs
            and np.isfinite(d_xy_block)
            and np.isfinite(d_z_block)
            and float(d_xy_block) <= float(xy_margin_m)
            and float(d_z_block) <= float(z_margin_m)
        )
        if is_near_blocked:
            return {
                "decision": "reject",
                "reason": (
                    f"near_blocked_xyz d_xy={float(d_xy_block):.3f} "
                    f"d_z={float(d_z_block):.3f}"
                ),
                "blocked_xyz": selected_xyz,
                "selected_xyz": selected_xyz,
            }
        pick_space_reason = pick_workspace_reject_reason(selected_xyz)
        if pick_space_reason:
            return {
                "decision": "reject",
                "reason": str(pick_space_reason),
                "blocked_xyz": selected_xyz,
                "selected_xyz": selected_xyz,
            }
        if bool(enforce_quality_gate) and not quality_ok:
            return {
                "decision": "continue",
                "reason": (
                    f"quality_gate_wait hits={hits}/{int(required_hits_local)} "
                    f"xyz_ok={bool(xyz_finite)}"
                ),
                "selected_xyz": selected_xyz,
            }
        if reject_color_norm and bool(PICK_OTHER_REJECT_SAME_COLOR):
            try:
                cand_color, cand_conf = classify_cube_color_patch(
                    obs.image_bgr,
                    bbox_xyxy=selected_row.get("bbox_xyxy", None),
                    center_uv=None,
                    bbox_core_ratio=0.55,
                )
            except Exception as exc:
                cand_color, cand_conf = "unknown", 0.0
                print(f"{log_prefix} same_color_classify_failed track_id={int(track_id)} err={exc}")
            cand_color_norm = str(cand_color).strip().lower()
            if cand_color_norm == reject_color_norm:
                return {
                    "decision": "reject",
                    "reason": (
                        f"same_color_as_seed color={reject_color_norm} "
                        f"conf={float(cand_conf):.3f}"
                    ),
                    "blocked_xyz": selected_xyz,
                    "selected_xyz": selected_xyz,
                }
        print(
            f"{log_prefix} accept track_id={int(track_id)} hits={hits}/{int(required_hits_local)} "
            f"xyz={selected_xyz} d_xy_block={float(d_xy_block):.3f} d_z_block={float(d_z_block):.3f} "
            f"quality_ok={bool(quality_ok)}"
        )
        return {
            "decision": "accept",
            "reason": "alternate_target_accepted",
            "selected_xyz": selected_xyz,
        }

    session = run_track_handoff_session(
        state=state,
        arm=arm,
        per=per,
        det=det,
        reject_cap=int(reject_cap),
        hard_timeout_s=float(hard_timeout_s),
        xy_margin_m=float(xy_margin_m),
        z_margin_m=float(z_margin_m),
        blocked_track_id=blocked_track_id,
        blocked_xyz=blocked_xyz,
        blocked_track_ids_extra=blocked_track_ids_extra,
        blocked_xyzs_extra=blocked_xyzs_extra,
        blocked_uv=blocked_uv,
        status_prefix=str(status_prefix),
        log_prefix=str(log_prefix),
        on_locked_candidate=_on_locked_candidate,
        enforce_pick_workspace_only=True,
    )
    if str(session.get("status", "")) == "ok":
        print(
            f"{log_prefix} summary status=accepted track_id={session.get('selected_track_id')} "
            f"uv={session.get('centered_pos')} xyz={session.get('selected_xyz')} "
            f"rejects={int(session.get('reject_count', 0))}/{int(session.get('reject_cap', 0))} "
            f"blocked_tracks={session.get('blocked_track_ids')} exit_reason={session.get('exit_reason')}"
        )
    else:
        print(
            f"{log_prefix} summary status=observe_retry rejects={int(session.get('reject_count', 0))}/"
            f"{int(session.get('reject_cap', 0))} blocked_tracks={session.get('blocked_track_ids')} "
            f"exit_reason={session.get('exit_reason')}"
        )
    return session

# ============================= Return / drop-zone logic =============================
def retreat_after_correction_drop(
    arm: Arm,
    label_prefix: str,
    *,
    drop_xyz: np.ndarray | list[float] | None = None,
) -> dict:
    """After correction return-grid drop: go straight to HOME with grip open."""
    _ = drop_xyz
    _bind_core_globals()
    if not bool(CORRECTION_RETREAT_HOME_ENABLED):
        return {"home_ok": True, "skipped": True}
    home_pose = HOME.copy()
    home_pose[3] = 0.0
    home_ok = bool(
        arm.goto_task_space(
            home_pose,
            duration=1.2,
            label=f"{label_prefix}_correction_retreat_home",
        )
    )
    print(f"[CorrectionRetreat] home_ok={bool(home_ok)}")
    return {
        "home_ok": bool(home_ok),
        "skipped": False,
    }


def goto_correction_drop_transit(
    arm: Arm,
    hold_grip: float,
    motion_supervisor: MotionGripSupervisor | None,
    label_prefix: str,
) -> bool:
    """Optional high transit waypoint before descending to the return-grid drop."""
    _bind_core_globals()
    if not bool(CORRECTION_DROP_TRANSIT_ENABLED):
        return True
    transit = CORRECTION_DROP_TRANSIT.copy()
    transit[3] = float(clamp_grip_cmd(hold_grip))
    ok = bool(
        arm.goto_task_space(
            transit,
            duration=1.0,
            label=f"{label_prefix}_correction_drop_transit",
            motion_supervisor=motion_supervisor,
        )
    )
    print(f"[CorrectionDropTransit] ok={bool(ok)} pose=({float(transit[0]):.3f},{float(transit[1]):.3f},{float(transit[2]):.3f})")
    return bool(ok)


def run_grasp_and_carry_common(
    state: CycleState,
    arm: Arm,
    per: Perception,
    centered_pos: tuple[int, int],
    label_prefix: str,
    safe_pick_reach_m: float | None = None,
    correction_abort_vertical_retreat: bool = False,
    grip_step_override: float | None = None,
    extra_x_offset_m: float = 0.0,
    extra_y_offset_m: float = 0.0,
    extra_z_offset_m: float = 0.0,
) -> tuple[str, float, MotionGripSupervisor | None]:
    _bind_core_globals()
    print("[Pick] Grasp sequence starting...")
    state.last_picked_track_id = (
        None if state.active_target_track_id is None else int(state.active_target_track_id)
    )
    state.last_picked_uv = [int(centered_pos[0]), int(centered_pos[1])]
    grasp_ok, hold_grip, grasp_reason, grasp_info = safe_grasp(
        arm=arm,
        per=per,
        cx=centered_pos[0],
        cy=centered_pos[1],
        grip_default=GRIP_DEFAULT,
        safe_pick_reach_m=safe_pick_reach_m,
        grip_step_override=grip_step_override,
        extra_x_offset_m=float(extra_x_offset_m),
        extra_y_offset_m=float(extra_y_offset_m),
        extra_z_offset_m=float(extra_z_offset_m),
    )
    if not grasp_ok:
        state.recent_grasp_fail_count += 1
        state.last_grasp_fail_reason = str(grasp_reason)
        state.last_pick_return_xyz = None
        state.last_pick_measured_xyz = None
        if bool(correction_abort_vertical_retreat):
            retreat_pose = grasp_info.get("lift_pose_xyzg", None) if isinstance(grasp_info, dict) else None
            retreat_arr = None
            try:
                if isinstance(retreat_pose, (list, tuple, np.ndarray)) and len(retreat_pose) >= 4:
                    arr = np.array(
                        [float(retreat_pose[0]), float(retreat_pose[1]), float(retreat_pose[2]), float(retreat_pose[3])],
                        dtype=float,
                    ).reshape(-1)
                    if arr.size >= 4 and np.all(np.isfinite(arr[:4])):
                        retreat_arr = arr
            except Exception:
                retreat_arr = None
            if retreat_arr is not None:
                retreat_ok = arm.goto_task_space(
                    retreat_arr,
                    duration=0.8,
                    label=f"{label_prefix}_abort_retreat_vertical",
                )
                print(
                    f"[PickAbortRetreat] vertical_retreat_ok={bool(retreat_ok)} "
                    f"reason={grasp_reason} pose={[float(retreat_arr[0]), float(retreat_arr[1]), float(retreat_arr[2])]}"
                )
            else:
                print(
                    f"[PickAbortRetreat] vertical_retreat_skipped reason={grasp_reason} "
                    "cause=missing_lift_pose"
                )
        if grasp_reason == "move_overcurrent_unrecoverable":
            state.stop_reason = grasp_reason
            state.skip_final_motion = True
            log_stop_reason("Unrecoverable move-time overcurrent during grasp lift. Stopping immediately")
            return "stop", 0.0, None
        state.cycles_without_place_progress += 1
        print(f"FAILED: Grasp failed ({grasp_reason}). Homing and retrying.")
        fail_home = HOME.copy()
        fail_home_grip = 0.0
        if str(grasp_reason) not in {"lift_failed", "post_lift_verify_failed"}:
            if float(hold_grip) > 0.0:
                fail_home_grip = float(clamp_grip_cmd(hold_grip))
        if float(fail_home_grip) > 0.0:
            print(
                f"[PickFailHold] preserving grip during home after grasp failure "
                f"reason={grasp_reason} grip={float(fail_home_grip):.3f}"
            )
        else:
            print(
                f"[PickFailOpen] opening grip for home after grasp failure "
                f"reason={grasp_reason}"
            )
        fail_home[3] = float(fail_home_grip)
        arm.goto_task_space(fail_home, duration=1.2, label=f"{label_prefix}_home_after_grasp_fail")
        return "retry", 0.0, None
    state.holding_object = True
    state.recent_grasp_fail_count = 0
    state.recent_pick_active_miss_count = 0
    state.last_grasp_fail_reason = ""
    state.current_hold_grip = hold_grip
    state.hold_grip_samples.append(float(hold_grip))
    pick_return_xyz = grasp_info.get("pick_target_xyz", None) if isinstance(grasp_info, dict) else None
    pick_measured_xyz = grasp_info.get("pick_measured_xyz", None) if isinstance(grasp_info, dict) else None
    if isinstance(pick_return_xyz, (list, tuple)) and len(pick_return_xyz) >= 3:
        try:
            state.last_pick_return_xyz = [float(pick_return_xyz[0]), float(pick_return_xyz[1]), float(pick_return_xyz[2])]
        except (TypeError, ValueError):
            state.last_pick_return_xyz = None
    else:
        state.last_pick_return_xyz = None
    if isinstance(pick_measured_xyz, (list, tuple)) and len(pick_measured_xyz) >= 3:
        try:
            state.last_pick_measured_xyz = [float(pick_measured_xyz[0]), float(pick_measured_xyz[1]), float(pick_measured_xyz[2])]
        except (TypeError, ValueError):
            state.last_pick_measured_xyz = None
    else:
        state.last_pick_measured_xyz = None
    carry_supervisor = make_motion_supervisor(hold_grip, label=f"{label_prefix}_carry")
    carry_mid = CARRY_MID.copy()
    carry_mid[3] = hold_grip
    if not arm.goto_task_space(
        carry_mid,
        duration=1.0,
        label=f"{label_prefix}_carry_mid",
        motion_supervisor=carry_supervisor,
    ):
        if arm.last_motion_reason == "move_overcurrent_unrecoverable":
            state.stop_reason = arm.last_motion_reason
            state.skip_final_motion = True
            log_stop_reason("Unrecoverable move-time overcurrent during carry-mid")
        else:
            state.stop_reason = "failed to reach carry-mid pose after grasp"
            log_stop_reason(state.stop_reason)
        return "stop", 0.0, None
    pick_verify = verify_pick_stability_signal(arm, hold_grip)
    state.last_pick_verification = dict(pick_verify)
    if not bool(pick_verify.get("ok", True)):
        state.pick_stability_fail_count += 1
        state.recent_grasp_fail_count += 1
        state.last_grasp_fail_reason = "pick_stability_verify_failed"
        state.holding_object = False
        state.current_hold_grip = 0.0
        state.last_pick_return_xyz = None
        state.last_pick_measured_xyz = None
        state.cycles_without_place_progress += 1
        print(
            f"[PickVerify] unstable carry signal after grasp "
            f"(median={pick_verify.get('median_a')} A). Retrying pick."
        )
        fail_home = HOME.copy()
        fail_home[3] = 0.0
        arm.goto_task_space(fail_home, duration=1.2, label=f"{label_prefix}_home_after_pick_verify_fail")
        return "retry", 0.0, None
    state.picked_count += 1
    # Successful grasp commits the next target choice; clear previous pick-other seed.
    state.pick_other_block_track_id = None
    state.pick_other_block_xyz = None
    state.pick_other_block_uv = None
    state.pick_other_block_track_ids = []
    state.pick_other_block_xyzs = []
    state.pick_other_block_uvs = []
    state.pick_other_block_source = "none"
    return "ok", hold_grip, carry_supervisor

# ============================= Misplaced cube correction =============================
