"""Observe/reobserve helpers extracted from runtime_loop (behavior-preserving)."""

from __future__ import annotations

import numpy as np

import pick_actions
import runtime_core as core
import stack_scene


def classify_centered_track_color(
    *,
    per,
    det,
    centered_pos: tuple[int, int],
    track_id: int | None,
) -> tuple[str, float]:
    if per is None or det is None:
        return "unknown", 0.0
    try:
        color_frame, _depth = per.get_frames()
        img_now = np.asanyarray(color_frame.get_data())
        _img_ann, candidates = det.detect_candidates_and_draw(img_now, draw=False)
        target = core.choose_track_candidate_near_uv(
            candidates,
            (None if track_id is None else int(track_id)),
            int(centered_pos[0]),
            int(centered_pos[1]),
            min_conf=0.0,
        )
        if target is None:
            return "unknown", 0.0
        return core.classify_cube_color_patch(
            img_now,
            bbox_xyxy=target.get("bbox_xyxy", None),
            center_uv=None,
            bbox_core_ratio=0.55,
        )
    except Exception as exc:
        print(f"[Color] pick_other_auto_classify_failed: {exc}")
        return "unknown", 0.0


def capture_pick_lock_snapshot(
    *,
    state,
    centered_pos: tuple[int, int] | None,
    cube_color: str,
    color_conf: float,
    arm,
    source: str,
) -> None:
    if centered_pos is None:
        return
    state.pregrasp_pick_lock_uv = [int(centered_pos[0]), int(centered_pos[1])]
    state.pregrasp_pick_lock_track_id = (
        None if state.active_target_track_id is None else int(state.active_target_track_id)
    )
    cube_color_norm = str(cube_color).strip().lower()
    state.pregrasp_pick_lock_color = (
        cube_color_norm if cube_color_norm in {"orange", "blue"} else "unknown"
    )
    state.pregrasp_pick_lock_color_conf = float(color_conf)
    try:
        q_now = np.array(arm.arm.measJointPosition[0:4], dtype=float).reshape(-1)
    except Exception:
        q_now = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
    if q_now.size >= 4 and np.all(np.isfinite(q_now[:4])):
        state.pregrasp_pick_lock_joints = [float(q_now[0]), float(q_now[1]), float(q_now[2]), float(q_now[3])]
    else:
        state.pregrasp_pick_lock_joints = None
    print(
        f"[PickLock] source={source} uv={state.pregrasp_pick_lock_uv} "
        f"track={state.pregrasp_pick_lock_track_id} color={state.pregrasp_pick_lock_color} "
        f"conf={float(state.pregrasp_pick_lock_color_conf):.3f} "
        f"joints_saved={state.pregrasp_pick_lock_joints is not None}"
    )


def clear_pick_lock_snapshot(*, state, source: str) -> None:
    state.pregrasp_pick_lock_uv = None
    state.pregrasp_pick_lock_track_id = None
    state.pregrasp_pick_lock_color = "unknown"
    state.pregrasp_pick_lock_color_conf = 0.0
    state.pregrasp_pick_lock_joints = None
    print(f"[PickLock] cleared source={source}")


def _normalize_pick_other_block_context(state) -> tuple[set[int], list[list[float]], list[list[int]]]:
    max_items = max(1, int(getattr(core, "PICK_OTHER_PERSIST_BLOCK_MAX", 8)))
    blocked_track_ids: set[int] = set()
    blocked_xyzs: list[list[float]] = []
    blocked_uvs: list[list[int]] = []

    def _add_track_id(raw_tid) -> None:
        if raw_tid is None:
            return
        try:
            blocked_track_ids.add(int(raw_tid))
        except (TypeError, ValueError):
            return

    def _add_xyz(raw_xyz) -> None:
        if not isinstance(raw_xyz, (list, tuple)) or len(raw_xyz) < 3:
            return
        try:
            arr = np.array([float(raw_xyz[0]), float(raw_xyz[1]), float(raw_xyz[2])], dtype=float)
        except (TypeError, ValueError):
            return
        if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
            row = [float(arr[0]), float(arr[1]), float(arr[2])]
            if row not in blocked_xyzs:
                blocked_xyzs.append(row)

    def _add_uv(raw_uv) -> None:
        if not isinstance(raw_uv, (list, tuple)) or len(raw_uv) < 2:
            return
        try:
            row = [int(raw_uv[0]), int(raw_uv[1])]
        except (TypeError, ValueError):
            return
        if row not in blocked_uvs:
            blocked_uvs.append(row)

    _add_track_id(getattr(state, "pick_other_block_track_id", None))
    _add_xyz(getattr(state, "pick_other_block_xyz", None))
    _add_uv(getattr(state, "pick_other_block_uv", None))
    for tid in list(getattr(state, "pick_other_block_track_ids", []) or []):
        _add_track_id(tid)
    for xyz in list(getattr(state, "pick_other_block_xyzs", []) or []):
        _add_xyz(xyz)
    for uv in list(getattr(state, "pick_other_block_uvs", []) or []):
        _add_uv(uv)

    blocked_xyzs = blocked_xyzs[:max_items]
    blocked_uvs = blocked_uvs[:max_items]
    if len(blocked_track_ids) > max_items:
        blocked_track_ids = set(sorted(blocked_track_ids)[-max_items:])
    return blocked_track_ids, blocked_xyzs, blocked_uvs


def _remember_pick_other_session_blocks(state, session: dict) -> None:
    blocked_track_ids, blocked_xyzs, blocked_uvs = _normalize_pick_other_block_context(state)
    max_items = max(1, int(getattr(core, "PICK_OTHER_PERSIST_BLOCK_MAX", 8)))
    for tid in list((session or {}).get("blocked_track_ids", []) or []):
        try:
            blocked_track_ids.add(int(tid))
        except (TypeError, ValueError):
            continue
    for xyz in list((session or {}).get("blocked_xyzs", []) or []):
        if not isinstance(xyz, (list, tuple)) or len(xyz) < 3:
            continue
        try:
            arr = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2])], dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
            row = [float(arr[0]), float(arr[1]), float(arr[2])]
            if row not in blocked_xyzs:
                blocked_xyzs.append(row)
    state.pick_other_block_track_ids = sorted(blocked_track_ids)[-max_items:]
    state.pick_other_block_xyzs = blocked_xyzs[-max_items:]
    state.pick_other_block_uvs = blocked_uvs[-max_items:]
    print(
        f"[PickOtherBlock] persisted tracks={state.pick_other_block_track_ids} "
        f"xyzs={len(state.pick_other_block_xyzs)} uvs={len(state.pick_other_block_uvs)}"
    )


def run_observe_action(
    *,
    command_for_history: str,
    clear_first: bool,
    source: str,
    state,
    arm,
    per,
    det,
    section_groups,
    cycle_count: int,
    centered_pos: tuple[int, int] | None,
    cube_color: str,
    color_conf: float,
    observe_fail_streak: int,
    observe_fail_stop_after: int,
    record_policy_step,
    capture_pick_lock_snapshot_fn,
) -> tuple[tuple[int, int] | None, str, float, int]:
    if source == "policy_reobserve":
        state.reobserve_requests += 1
    elif source == "auto_recovery":
        state.auto_recovery_observes += 1
    clear_pick_lock_snapshot(state=state, source=f"{source}_begin")
    if clear_first:
        centered_pos = None
        cube_color = "unknown"
        color_conf = 0.0
        print("[Policy] reobserve requested; running active rescan.")
    block_source = str(getattr(state, "pick_other_block_source", "none")).strip().lower()
    if block_source in {"classify", "return"}:
        blocked_track_ids, blocked_xyzs, blocked_uvs = _normalize_pick_other_block_context(state)
    else:
        blocked_track_ids, blocked_xyzs, blocked_uvs = set(), [], []
    if blocked_track_ids or blocked_xyzs or blocked_uvs:
        print(
            f"[PickOtherBlock] applying_to_observe source={block_source} "
            f"tracks={sorted(blocked_track_ids)} xyzs={len(blocked_xyzs)} uvs={len(blocked_uvs)}"
        )
    pick_status, centered_pos = pick_actions.run_pick_center_cycle(
        state=state,
        arm=arm,
        per=per,
        det=det,
        label_prefix=f"prompted_cycle_{state.cycle_count}",
        section_groups=section_groups,
        blocked_track_ids=blocked_track_ids,
        blocked_xyzs=blocked_xyzs,
        blocked_uvs=blocked_uvs,
    )
    if pick_status == "retry":
        centered_pos = None
        cube_color = "unknown"
        color_conf = 0.0
        center_failure = getattr(state, "last_center_failure", {}) or {}
        if not isinstance(center_failure, dict):
            center_failure = {}
        center_status = str(center_failure.get("status", "") or "").strip().lower()
        if center_status == "active_detection_timeout":
            selector_meta = center_failure.get("selector_meta", {})
            if not isinstance(selector_meta, dict):
                selector_meta = {}
            print(
                f"[ObserveSafety] miss_streak_unchanged={int(observe_fail_streak)}/"
                f"{int(observe_fail_stop_after)} reason=visible_target_active_timeout "
                f"raw={int(center_failure.get('candidate_count', 0) or 0)} "
                f"filtered={int(center_failure.get('filtered_count', 0) or 0)} "
                f"eligible={int(selector_meta.get('eligible_count', 0) or 0)}"
            )
        else:
            observe_fail_streak += 1
            print(
                f"[ObserveSafety] miss_streak={int(observe_fail_streak)}/"
                f"{int(observe_fail_stop_after)}"
            )
        record_policy_step(command_for_history, "observe_retry", progress=False)
        return centered_pos, cube_color, float(color_conf), int(observe_fail_streak)
    if pick_status == "stop_orientation":
        centered_pos = None
        cube_color = "unknown"
        color_conf = 0.0
        print(
            "[Policy] observe_scene stopped for pick-space orientation inspection; "
            "awaiting policy stop_run."
        )
        record_policy_step(command_for_history, "observe_pick_space_orientation_check", progress=False)
        return centered_pos, cube_color, float(color_conf), int(observe_fail_streak)
    if pick_status == "stop":
        # Let policy decide stop_run explicitly after observing repeated empty scene.
        centered_pos = None
        cube_color = "unknown"
        color_conf = 0.0
        if str(getattr(state, "stop_reason", "") or "") == "pick scene empty by confirmation passes":
            state.stop_reason = "completed"
        print(
            "[Policy] observe_scene reported empty by confirmation passes; "
            "awaiting policy decision (likely stop_run)."
        )
        record_policy_step(command_for_history, "observe_empty_scene_confirmed", progress=False)
        return centered_pos, cube_color, float(color_conf), int(observe_fail_streak)
    assert centered_pos is not None
    cube_color = "unknown"
    color_conf = 0.0
    observe_fail_streak = 0
    capture_pick_lock_snapshot(
        state=state,
        centered_pos=centered_pos,
        cube_color="unknown",
        color_conf=0.0,
        arm=arm,
        source="observe_lock",
    )
    record_policy_step(command_for_history, "observe_locked_target", progress=True)
    return centered_pos, cube_color, float(color_conf), int(observe_fail_streak)


def run_pick_other_action(
    *,
    command_for_history: str,
    state,
    arm,
    per,
    det,
    centered_pos: tuple[int, int] | None,
    cube_color: str,
    color_conf: float,
    place_verify_v2_min_hits: int,
    pick_other_validate_samples: int,
    record_policy_step,
    capture_pick_lock_snapshot_fn,
) -> tuple[tuple[int, int] | None, str, float]:
    state.reobserve_requests += 1
    source = str(getattr(state, "pick_other_block_source", "none")).strip().lower()
    if source not in {"classify", "return"}:
        state.cycles_without_place_progress += 1
        print(
            f"[PickOther] unavailable: source={source}. "
            "Need classify or return seed first."
        )
        record_policy_step(command_for_history, "pick_other_unavailable", progress=False)
        return centered_pos, cube_color, float(color_conf)
    if (
        state.pick_other_block_track_id is None
        and state.pick_other_block_xyz is None
        and state.pick_other_block_uv is None
    ):
        state.cycles_without_place_progress += 1
        print("[PickOther] unavailable: no seed context.")
        record_policy_step(command_for_history, "pick_other_unavailable", progress=False)
        return centered_pos, cube_color, float(color_conf)
    print(
        f"[PickOther] source={source} seed(track={state.pick_other_block_track_id}, "
        f"uv={state.pick_other_block_uv}, xyz={state.pick_other_block_xyz})"
    )
    reject_same_color = ""
    if source == "classify":
        reject_same_color = str(cube_color).strip().lower()
        if reject_same_color not in {"orange", "blue"}:
            reject_same_color = ""
    session = pick_actions.run_pick_other_session(
        state=state,
        arm=arm,
        per=per,
        det=det,
        label_prefix=f"prompted_cycle_{state.cycle_count}",
        blocked_track_id=state.pick_other_block_track_id,
        blocked_xyz=state.pick_other_block_xyz,
        blocked_uv=state.pick_other_block_uv,
        blocked_track_ids_extra=list(getattr(state, "pick_other_block_track_ids", []) or []),
        blocked_xyzs_extra=list(getattr(state, "pick_other_block_xyzs", []) or []),
        reject_same_color=reject_same_color,
        status_prefix="pick_other",
        log_prefix="[PickOther]",
        required_hits=max(1, int(place_verify_v2_min_hits)),
        measurement_samples=max(
            int(pick_other_validate_samples),
            int(max(1, int(place_verify_v2_min_hits))),
        ),
        enforce_quality_gate=True,
    )
    if str(session.get("status", "")) != "ok":
        _remember_pick_other_session_blocks(state, session)
        centered_pos = None
        cube_color = "unknown"
        color_conf = 0.0
        state.cycles_without_place_progress += 1
        record_policy_step(command_for_history, "pick_other_observe_retry", progress=False)
        return centered_pos, cube_color, float(color_conf)
    uv = session.get("centered_pos", None)
    if not isinstance(uv, (list, tuple)) or len(uv) < 2:
        centered_pos = None
        cube_color = "unknown"
        color_conf = 0.0
        state.cycles_without_place_progress += 1
        record_policy_step(command_for_history, "pick_other_invalid_lock", progress=False)
        return centered_pos, cube_color, float(color_conf)
    centered_pos = (int(uv[0]), int(uv[1]))
    sel_tid = session.get("selected_track_id", None)
    state.active_target_track_id = (None if sel_tid is None else int(sel_tid))
    cube_color, color_conf = classify_centered_track_color(
        per=per,
        det=det,
        centered_pos=centered_pos,
        track_id=state.active_target_track_id,
    )
    if str(cube_color).strip().lower() in {"orange", "blue"}:
        print(f"[Color] pick_other_locked classified as {cube_color} (conf={float(color_conf):.3f})")
    # One-shot use: once alternate target is locked, clear seed context.
    state.pick_other_block_track_id = None
    state.pick_other_block_xyz = None
    state.pick_other_block_uv = None
    state.pick_other_block_track_ids = []
    state.pick_other_block_xyzs = []
    state.pick_other_block_uvs = []
    state.pick_other_block_source = "none"
    capture_pick_lock_snapshot(
        state=state,
        centered_pos=centered_pos,
        cube_color=str(cube_color),
        color_conf=float(color_conf),
        arm=arm,
        source="pick_other_lock",
    )
    print(
        f"[PickOther] locked track_id={state.active_target_track_id} uv={centered_pos}. "
        "Seed context cleared."
    )
    record_policy_step(command_for_history, "pick_other_locked_target", progress=True)
    return centered_pos, cube_color, float(color_conf)


def run_post_lift_place_space_refresh(
    *,
    source_tag: str,
    state,
    arm,
    per,
    det,
) -> dict | None:
    try:
        arm.goto_task_space(
            core.PLACE_LOOKING,
            duration=0.9,
            label=f"prompted_cycle_{state.cycle_count}_{source_tag}_place_look",
        )
    except Exception as exc:
        print(f"[PostLiftRefresh] move failed: {exc}")
        return None
    truth = stack_scene.run_place_space_truth_pass(
        state=state,
        arm=arm,
        per=per,
        det=det,
        centered_pos=None,
        active_track_id=None,
        mode=str(source_tag),
        detector_draw=False,
        show_window=False,
        status_line=str(source_tag),
    )
    return truth
