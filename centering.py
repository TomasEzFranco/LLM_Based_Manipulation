#!/usr/bin/env python3
"""Centering control owner module (extract-only)."""

from __future__ import annotations

import cv2
import numpy as np
import time
from collections import deque

_CORE_BIND_READY = False


def _bind_core_globals() -> None:
    global _CORE_BIND_READY
    import runtime_core as core
    protected = {
        '_bind_core_globals', '_CORE_BIND_READY',
        '_draw_center_reference_overlay', '_draw_center_stability_overlay',
        '_draw_forbidden_uv_overlay', '_show_center_frame',
        '_handle_selected_center_candidate', 'center_object_slowly',
        'center_object_on_expected_slot',
    }
    for name, value in core.__dict__.items():
        if name.startswith('__') or name in protected:
            continue
        globals()[name] = value
    _CORE_BIND_READY = True


def _filter_verify_negative_y_candidates(
    raw_candidates: list[dict],
    projected_rows: list[dict],
) -> tuple[list[dict], dict]:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    if not bool(PLACE_VERIFY_V2_AVOID_NEGATIVE_Y):
        return list(raw_candidates or []), {"negative_y_skips": 0, "min_y_m": float(PLACE_VERIFY_V2_MIN_TRACK_Y_M)}
    min_y_m = float(PLACE_VERIFY_V2_MIN_TRACK_Y_M)
    out: list[dict] = []
    skipped = 0
    for c in list(raw_candidates or []):
        try:
            u = int(c.get("u", 0))
            v = int(c.get("v", 0))
        except Exception:
            out.append(dict(c))
            continue
        proj = _match_projected_row_by_uv(list(projected_rows or []), u=u, v=v)
        if proj is not None:
            try:
                xyz = np.array(proj.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
            except Exception:
                xyz = np.array([np.nan, np.nan, np.nan], dtype=float)
            if xyz.size >= 2 and np.all(np.isfinite(xyz[:2])) and float(xyz[1]) < min_y_m:
                skipped += 1
                continue
        out.append(dict(c))
    return out, {"negative_y_skips": int(skipped), "min_y_m": float(min_y_m)}

def center_object_on_expected_slot(
    det: YOLODetector | None,
    arm: Arm,
    per: Perception | None,
    expected_xyz: np.ndarray,
    timeout_s: float | None = None,
    required_centered_frames: int = 4,
    min_conf: float | None = None,
    radius_m: float | None = None,
    show_window: bool | None = None,
    stack_level: int = 0,
    min_z_m: float | None = None,
    expected_section: str | None = None,
    expected_color: str | None = None,
    state: CycleState | None = None,
    blocked_track_ids_seed: set[int] | None = None,
    blocked_uv_seed: list[list[float]] | None = None,
    use_pixel_blacklist: bool = True,
    target_mode_override: str | None = None,
    apply_lock_wrong_xy_gate: bool = True,
    use_projected_xyz_for_filter: bool = True,
    no_target_timeout_s: float | None = None,
    reset_timeout_on_first_candidate: bool = False,
    preferred_track_id: int | None = None,
    strict_track_lock: bool = False,
) -> tuple[int, int] | None:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    if timeout_s is None:
        timeout_s = float(PLACE_VERIFY_V2_ACTIVE_CENTER_TIMEOUT_S)
    timeout_s = max(0.2, float(timeout_s))
    if no_target_timeout_s is None:
        no_target_timeout_s = float(timeout_s)
    no_target_timeout_s = max(0.2, float(no_target_timeout_s))
    if min_conf is None:
        min_conf = float(DETECT_CONF)
    if radius_m is None:
        radius_m = float(PLACE_VERIFY_V2_RADIUS_M)
    if show_window is None:
        show_window = bool(SHOW_WINDOW)
    if det is None or per is None:
        return None
    _reset_centering_integrator()
    if state is not None:
        state.last_verify_lock_uv = None
        state.last_verify_lock_xyz = None
        state.last_verify_lock_track_id = None
        state.last_verify_lock_source = "none"
    target = np.array(expected_xyz, dtype=float).reshape(-1)
    if target.size < 3 or not np.all(np.isfinite(target[:3])):
        return None
    centered_frames = deque(maxlen=max(1, int(required_centered_frames)))
    if min_z_m is None:
        min_z_m = compute_verify_stack_min_z(float(target[2]), int(stack_level))
    prefer_top = bool(PLACE_VERIFY_V2_STACK_PREFER_TOP and int(stack_level) >= 2)
    frame_idx = 0
    verify_track_id: int | None = None
    verify_track_smooth_frames_left = 0
    forced_target_uv: tuple[int, int] | None = None
    forced_target_frames_left = 0
    if blocked_track_ids_seed is None:
        blocked_verify_track_ids: set[int] = set()
    else:
        # Persist rejected tracks across recenter calls in the same verify cycle.
        normalized_ids: set[int] = set()
        for tid in list(blocked_track_ids_seed):
            try:
                normalized_ids.add(int(tid))
            except (TypeError, ValueError):
                continue
        blocked_track_ids_seed.clear()
        blocked_track_ids_seed.update(normalized_ids)
        blocked_verify_track_ids = blocked_track_ids_seed
    if (not bool(use_pixel_blacklist)) or blocked_uv_seed is None:
        dynamic_forbidden_uv: list[list[float]] = []
    else:
        normalized_uv: list[list[float]] = []
        for uv in list(blocked_uv_seed):
            if not isinstance(uv, (list, tuple)) or len(uv) < 2:
                continue
            try:
                uu = int(uv[0])
                vv = int(uv[1])
            except (TypeError, ValueError):
                continue
            rr = float(PLACE_VERIFY_V2_RECENTER_DYNAMIC_BLACKLIST_PX)
            if len(uv) >= 3:
                try:
                    rr = float(uv[2])
                except (TypeError, ValueError):
                    rr = float(PLACE_VERIFY_V2_RECENTER_DYNAMIC_BLACKLIST_PX)
            normalized_uv.append([int(uu), int(vv), float(max(0.0, rr))])
        blocked_uv_seed.clear()
        blocked_uv_seed.extend(normalized_uv)
        dynamic_forbidden_uv = blocked_uv_seed

    def _append_dynamic_forbidden_uv(u_px: int, v_px: int, radius_px: float) -> None:
        if not bool(use_pixel_blacklist):
            return
        r = float(max(0.0, float(radius_px)))
        # Avoid duplicate blacklist circles in near-identical spots.
        for uv in dynamic_forbidden_uv:
            if not isinstance(uv, (list, tuple)) or len(uv) < 2:
                continue
            try:
                du = float(int(u_px) - int(uv[0]))
                dv = float(int(v_px) - int(uv[1]))
            except (TypeError, ValueError):
                continue
            if (du * du + dv * dv) <= (6.0 * 6.0):
                return
        dynamic_forbidden_uv.append([int(u_px), int(v_px), float(r)])
    expected_section_name = str(expected_section or "").strip().lower()
    expected_color_name = str(expected_color or "").strip().lower()
    use_projected_xyz_for_filter = bool(use_projected_xyz_for_filter)
    expected_xy = (float(target[0]), float(target[1]))
    # Keep top-first scoped to dedicated flows (e.g., misplaced recovery). Default here is filtered-first.
    target_mode = str(target_mode_override or "filtered_first").strip().lower()
    if target_mode not in {"top_first", "filtered_first"}:
        target_mode = "filtered_first"
    max_candidate_tries = max(1, int(PLACE_VERIFY_V2_RECENTER_MAX_CANDIDATE_TRIES))
    lock_reject_count = 0
    t0 = time.time()
    active_t0: float | None = None
    timeout_reason = "timeout"
    if show_window:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 720)

    preferred_tid_norm: int | None = None
    if preferred_track_id is not None:
        try:
            preferred_tid_norm = int(preferred_track_id)
        except Exception:
            preferred_tid_norm = None
    last_target_uv: tuple[int, int] | None = None
    last_target_seen_t = 0.0
    track_reacquire_grace_s = float(max(0.25, min(1.8, 0.35 * float(max(0.2, timeout_s)))))

    def _select_pixel_candidate(
        raw_candidates: list[dict],
        view_cx: int,
        view_cy: int,
        conf_min: float,
        prefer_top_px: bool,
        target_uv: tuple[int, int] | None = None,
        forbidden_uv: list[list[float]] | None = None,
        forbid_radius_px: float = 0.0,
        blocked_track_ids: set[int] | None = None,
    ) -> dict | None:
        def _is_forbidden(u_px: int, v_px: int) -> bool:
            if not forbidden_uv:
                return False
            for uv in forbidden_uv:
                if not isinstance(uv, (list, tuple)) or len(uv) < 2:
                    continue
                entry_radius = float(forbid_radius_px)
                if len(uv) >= 3:
                    try:
                        entry_radius = float(uv[2])
                    except (TypeError, ValueError):
                        entry_radius = float(forbid_radius_px)
                r2 = float(max(0.0, float(entry_radius))) ** 2
                du = float(int(u_px) - int(uv[0]))
                dv = float(int(v_px) - int(uv[1]))
                if (du * du + dv * dv) <= r2:
                    return True
            return False

        rows: list[dict] = []
        for c in raw_candidates:
            conf = float(c.get("conf", 0.0))
            if conf < float(conf_min):
                continue
            u = int(c.get("u", 0))
            v = int(c.get("v", 0))
            if _is_forbidden(u, v):
                continue
            linked_track_id: int | None = None
            raw_track_id = c.get("track_id", None)
            if raw_track_id is not None:
                try:
                    linked_track_id = int(raw_track_id)
                except (TypeError, ValueError):
                    linked_track_id = None
            if linked_track_id is None and state is not None and TRACK_ENABLE:
                linked = nearest_visible_track_by_uv(state, u=u, v=v, max_dist_px=120.0)
                if linked is not None:
                    linked_track_id = int(linked)
            if linked_track_id is not None and blocked_track_ids and int(linked_track_id) in blocked_track_ids:
                continue
            if preferred_tid_norm is not None and bool(strict_track_lock):
                if linked_track_id is None or int(linked_track_id) != int(preferred_tid_norm):
                    continue
            d2 = float((u - int(view_cx)) ** 2 + (v - int(view_cy)) ** 2)
            row = dict(c)
            row["u"] = u
            row["v"] = v
            row["conf"] = conf
            row["linked_track_id"] = linked_track_id
            row["d2_px"] = d2
            if target_uv is not None:
                row["d2_track_px"] = float((u - int(target_uv[0])) ** 2 + (v - int(target_uv[1])) ** 2)
            else:
                row["d2_track_px"] = float("inf")
            rows.append(row)
        if not rows:
            return None
        if target_uv is not None:
            # When a track target is active, follow it first; this avoids jumping back to another top candidate.
            rows.sort(
                key=lambda r: (
                    float(r.get("d2_track_px", float("inf"))),
                    float(r.get("d2_px", float("inf"))),
                    -float(r.get("conf", 0.0)),
                )
            )
            return rows[0]
        if prefer_top_px:
            # Top-first is primary. Track lock is only a short smoothing tie-breaker.
            rows.sort(
                key=lambda r: (
                    int(r.get("v", 0)),
                    float(r.get("d2_track_px", float("inf"))),
                    float(r.get("d2_px", float("inf"))),
                    -float(r.get("conf", 0.0)),
                )
            )
        else:
            rows.sort(
                key=lambda r: (
                    float(r.get("d2_px", float("inf"))),
                    float(r.get("d2_track_px", float("inf"))),
                    -float(r.get("conf", 0.0)),
                    int(r.get("v", 0)),
                )
            )
        return rows[0]

    def _selected_is_wrong_target(
        *,
        selected_row: dict,
        bgr_img: np.ndarray,
        projected_rows: list[dict],
        depth_frame,
        view_cx: int | None = None,
    ) -> list[str]:
        reasons: list[str] = []
        u = int(selected_row.get("u", 0))
        v = int(selected_row.get("v", 0))
        xyz_eval: np.ndarray | None = None
        proj = _match_projected_row_by_uv(projected_rows, u=u, v=v)
        if proj is not None:
            xyz = np.array(proj.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
            if xyz.size >= 3 and np.all(np.isfinite(xyz[:3])):
                xyz_eval = xyz
        if xyz_eval is None:
            # Fallback for lock-time validation: estimate XYZ directly at locked pixel.
            # This helps avoid false "no blacklist" when projected_rows missed the selected candidate.
            xyz_fast = estimate_base_xyz_from_uv_fast(
                arm=arm,
                per=per,
                depth_frame=depth_frame,
                u=u,
                v=v,
            )
            xyz_fast = np.array(xyz_fast, dtype=float).reshape(-1)
            if xyz_fast.size >= 3 and np.all(np.isfinite(xyz_fast[:3])):
                xyz_eval = xyz_fast
        if xyz_eval is not None and xyz_eval.size >= 2 and np.all(np.isfinite(xyz_eval[:2])):
            if expected_section_name:
                section_centers = _verify_section_y_centers()
                inferred = _infer_section_for_base_y(float(xyz_eval[1]), section_centers)
                if inferred is not None and str(inferred).strip().lower() != expected_section_name:
                    reasons.append("wrong_section")
            if bool(apply_lock_wrong_xy_gate) and PLACE_VERIFY_V2_RECENTER_DISALLOW_WRONG_XY:
                d_xy_expected = float(
                    math.hypot(float(xyz_eval[0]) - expected_xy[0], float(xyz_eval[1]) - expected_xy[1])
                )
                if d_xy_expected > float(PLACE_VERIFY_V2_RECENTER_WRONG_XY_M):
                    reasons.append("wrong_xy")
        if expected_section_name and PLACE_VERIFY_V2_SECTION_PIXEL_GATE:
            img_cx = None
            if view_cx is not None and np.isfinite(float(view_cx)):
                img_cx = int(view_cx)
            elif isinstance(bgr_img, np.ndarray) and bgr_img.ndim >= 2 and bgr_img.shape[1] > 0:
                img_cx = int(bgr_img.shape[1] // 2)
            if img_cx is not None:
                margin_px = max(0, int(PLACE_VERIFY_V2_SECTION_PIXEL_MARGIN_PX))
                if expected_section_name == str(SECTION_LEFT_NAME).strip().lower() and int(u) > int(img_cx + margin_px):
                    reasons.append("wrong_section_px")
                elif expected_section_name == str(SECTION_RIGHT_NAME).strip().lower() and int(u) < int(img_cx - margin_px):
                    reasons.append("wrong_section_px")
        if expected_color_name and PLACE_VERIFY_V2_RECENTER_COLOR_FILTER:
            color_name, color_conf = classify_cube_color_patch(
                bgr_img,
                bbox_xyxy=selected_row.get("bbox_xyxy", None),
                center_uv=(u, v),
            )
            color_name_n = str(color_name).strip().lower()
            if (
                float(color_conf) >= float(PLACE_VERIFY_V2_RECENTER_COLOR_MIN_CONF)
                and color_name_n not in {"", "unknown", expected_color_name}
            ):
                reasons.append("wrong_color")
        return reasons

    def _resolve_selected_xyz(selected_row: dict, projected_rows_now: list[dict], depth_frame_now) -> list[float] | None:
        u = int(selected_row.get("u", 0))
        v = int(selected_row.get("v", 0))
        proj_row = _match_projected_row_by_uv(projected_rows_now, u=u, v=v)
        if proj_row is not None:
            xyz_proj = np.array(proj_row.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
            if xyz_proj.size >= 3 and np.all(np.isfinite(xyz_proj[:3])):
                return [float(xyz_proj[0]), float(xyz_proj[1]), float(xyz_proj[2])]
        xyz_fast = estimate_base_xyz_from_uv_fast(
            arm=arm,
            per=per,
            depth_frame=depth_frame_now,
            u=u,
            v=v,
        )
        xyz_fast = np.array(xyz_fast, dtype=float).reshape(-1)
        if xyz_fast.size >= 3 and np.all(np.isfinite(xyz_fast[:3])):
            return [float(xyz_fast[0]), float(xyz_fast[1]), float(xyz_fast[2])]
        return None

    try:
        while True:
            now_loop = time.time()
            if bool(reset_timeout_on_first_candidate):
                if active_t0 is None:
                    if (now_loop - t0) >= float(no_target_timeout_s):
                        timeout_reason = "no_candidate_timeout"
                        break
                    elapsed_for_overlay = float(now_loop - t0)
                    timeout_for_overlay = float(no_target_timeout_s)
                else:
                    if (now_loop - float(active_t0)) >= float(timeout_s):
                        timeout_reason = "active_timeout"
                        break
                    elapsed_for_overlay = float(now_loop - float(active_t0))
                    timeout_for_overlay = float(timeout_s)
            else:
                if (now_loop - t0) >= float(timeout_s):
                    timeout_reason = "timeout"
                    break
                elapsed_for_overlay = float(now_loop - t0)
                timeout_for_overlay = float(timeout_s)
            detector_draw = bool(show_window and ((UI_MODE == "debug") or UI_DRAW_ALL_BOXES))
            obs = observe_scene_frame(
                det=det,
                arm=arm,
                per=per,
                draw=detector_draw,
                projected_min_conf=float(min_conf),
                state=state,
                update_tracks=True,
            )
            if obs is None:
                break
            img = obs.image_bgr
            depth = obs.depth_frame
            cx, cy = obs.image_center_uv
            img_display = obs.image_display
            candidates = obs.candidates
            projected_rows = obs.projected_rows
            verify_candidates, negative_y_meta = _filter_verify_negative_y_candidates(
                list(candidates or []),
                list(projected_rows or []),
            )
            projected_rows_for_filter = projected_rows if bool(use_projected_xyz_for_filter) else []
            _draw_center_reference_overlay(
                img_display=img_display,
                cx=cx,
                cy=cy,
                elapsed_s=float(elapsed_for_overlay),
                timeout_s=float(timeout_for_overlay),
            )

            selected_uv = None
            selected: dict | None = None
            select_mode = "none"
            filtered_candidates, filter_meta = _filter_verify_candidates(
                raw_candidates=verify_candidates,
                bgr_img=img,
                projected_rows=projected_rows_for_filter,
                expected_section=expected_section,
                expected_color=expected_color,
                expected_xyz=target,
                use_projected_geometry=bool(use_projected_xyz_for_filter),
            )
            top_first_mode = (target_mode == "top_first")
            forbidden_uv: list[list[float]] = []
            forbid_radius_px = 0.0
            if bool(use_pixel_blacklist):
                if bool(
                    PLACE_VERIFY_V2_RECENTER_DISALLOW_WRONG_COLOR
                    and expected_color
                    and (not top_first_mode or PLACE_VERIFY_V2_RECENTER_PREBLACKLIST_WRONG_COLOR_TOP_FIRST)
                ):
                    forbidden_uv.extend(list(filter_meta.get("wrong_color_uv", [])))
                if bool(
                    PLACE_VERIFY_V2_RECENTER_DISALLOW_WRONG_XY
                    and (not top_first_mode)
                ):
                    forbidden_uv.extend(list(filter_meta.get("wrong_xy_uv", [])))
                if dynamic_forbidden_uv:
                    forbidden_uv.extend(dynamic_forbidden_uv)
                forbid_radius_px = float(
                    max(
                        float(PLACE_VERIFY_V2_RECENTER_WRONG_COLOR_BLACKLIST_PX)
                        if bool(PLACE_VERIFY_V2_RECENTER_DISALLOW_WRONG_COLOR and expected_color)
                        else 0.0,
                        float(PLACE_VERIFY_V2_RECENTER_WRONG_XY_BLACKLIST_PX)
                        if bool(PLACE_VERIFY_V2_RECENTER_DISALLOW_WRONG_XY)
                        else 0.0,
                    )
                )
            if bool(use_pixel_blacklist) and bool(PLACE_VERIFY_V2_RECENTER_SHOW_BLACKLIST_OVERLAY):
                _draw_forbidden_uv_overlay(
                    img_display=img_display,
                    forbidden_uv=forbidden_uv,
                    default_radius_px=forbid_radius_px,
                )
            target_uv = None
            if forced_target_uv is not None and int(forced_target_frames_left) > 0:
                target_uv = (int(forced_target_uv[0]), int(forced_target_uv[1]))
                forced_target_frames_left = max(0, int(forced_target_frames_left) - 1)
                if forced_target_frames_left == 0:
                    forced_target_uv = None
            elif (
                verify_track_id is not None
                and verify_track_smooth_frames_left > 0
                and int(verify_track_id) not in blocked_verify_track_ids
                and state is not None
            ):
                row = state.track_memory.get(int(verify_track_id), None)
                if row is not None and int(row.get("miss_frames", 0)) == 0:
                    uv = row.get("uv", None)
                    if isinstance(uv, (list, tuple)) and len(uv) >= 2:
                        target_uv = (int(uv[0]), int(uv[1]))
            if target_uv is not None and forced_target_uv is None:
                verify_track_smooth_frames_left = max(0, int(verify_track_smooth_frames_left) - 1)

            if target_mode == "top_first":
                # Top-first mode restores active centering continuity:
                # pick from all visible YOLO candidates (above min_conf), top-most first.
                top_first_blacklist = (
                    forbidden_uv
                    if bool(
                        (PLACE_VERIFY_V2_RECENTER_DISALLOW_WRONG_COLOR and expected_color)
                        or PLACE_VERIFY_V2_RECENTER_DISALLOW_WRONG_XY
                    )
                    else []
                )
                selected = _select_pixel_candidate(
                    raw_candidates=verify_candidates,
                    view_cx=cx,
                    view_cy=cy,
                    conf_min=float(min_conf),
                    prefer_top_px=bool(PLACE_VERIFY_V2_RECENTER_PIXEL_TOP),
                    target_uv=target_uv,
                    forbidden_uv=top_first_blacklist,
                    forbid_radius_px=forbid_radius_px,
                    blocked_track_ids=blocked_verify_track_ids,
                )
                if selected is not None:
                    if target_uv is not None:
                        select_mode = "pixel_top_first_track_lock"
                    else:
                        select_mode = "pixel_top_first" if PLACE_VERIFY_V2_RECENTER_PIXEL_TOP else "pixel_center_first"
            else:
                if not PLACE_VERIFY_V2_RECENTER_PIXEL_ONLY:
                    tx, ty, tz = map(float, target[:3])
                    roi_rows: list[dict] = []
                    for row in projected_rows:
                        xyz = np.array(row.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
                        if xyz.size < 3 or not np.all(np.isfinite(xyz[:3])):
                            continue
                        d_xy = float(math.hypot(float(xyz[0]) - tx, float(xyz[1]) - ty))
                        d_z = float(abs(float(xyz[2]) - tz))
                        if min_z_m is not None and np.isfinite(float(min_z_m)) and float(xyz[2]) < float(min_z_m):
                            continue
                        if d_xy <= max(radius_m, PLACE_VERIFY_V2_RADIUS_M) and d_z <= max(PLACE_VERIFY_V2_Z_MARGIN_M * 2.0, 0.09):
                            row2 = dict(row)
                            row2["d_xy_m"] = d_xy
                            row2["d_z_m"] = d_z
                            roi_rows.append(row2)
                    if prefer_top:
                        roi_rows.sort(
                            key=lambda r: (
                                -float(r.get("xyz", [0.0, 0.0, -np.inf])[2]),
                                float(r.get("d_xy_m", float("inf"))),
                                float(r.get("d_z_m", float("inf"))),
                                -float(r.get("conf", 0.0)),
                            )
                        )
                    else:
                        roi_rows.sort(
                            key=lambda r: (
                                float(r.get("d_xy_m", float("inf"))),
                                float(r.get("d_z_m", float("inf"))),
                                -float(r.get("conf", 0.0)),
                            )
                        )
                    if roi_rows:
                        selected = roi_rows[0]
                        select_mode = "roi"

                if selected is None:
                    # Stage 1: strict filter (section + color when configured)
                    selected = _select_pixel_candidate(
                        raw_candidates=filtered_candidates,
                        view_cx=cx,
                        view_cy=cy,
                        conf_min=float(min_conf),
                        prefer_top_px=bool(PLACE_VERIFY_V2_RECENTER_PIXEL_TOP),
                        target_uv=target_uv,
                        forbidden_uv=forbidden_uv,
                        forbid_radius_px=forbid_radius_px,
                        blocked_track_ids=blocked_verify_track_ids,
                    )
                    if selected is not None:
                        if target_uv is not None:
                            select_mode = "pixel_track_lock"
                        else:
                            select_mode = "pixel_filtered_top" if PLACE_VERIFY_V2_RECENTER_PIXEL_TOP else "pixel_filtered_center"

                if selected is None and expected_color and PLACE_VERIFY_V2_RECENTER_COLOR_FILTER:
                    # Stage 2: relax section, keep expected-color lock.
                    color_only_candidates, _color_only_meta = _filter_verify_candidates(
                        raw_candidates=verify_candidates,
                        bgr_img=img,
                        projected_rows=projected_rows_for_filter,
                        expected_section=None,
                        expected_color=expected_color,
                        expected_xyz=target,
                        use_projected_geometry=bool(use_projected_xyz_for_filter),
                    )
                    selected = _select_pixel_candidate(
                        raw_candidates=color_only_candidates,
                        view_cx=cx,
                        view_cy=cy,
                        conf_min=float(min_conf),
                        prefer_top_px=bool(PLACE_VERIFY_V2_RECENTER_PIXEL_TOP),
                        target_uv=target_uv,
                        forbidden_uv=forbidden_uv,
                        forbid_radius_px=forbid_radius_px,
                        blocked_track_ids=blocked_verify_track_ids,
                    )
                    if selected is not None:
                        select_mode = "pixel_color_only"

                if selected is None and expected_section and not (
                    expected_color and PLACE_VERIFY_V2_RECENTER_DISALLOW_WRONG_COLOR
                ):
                    # Stage 3: relax color, keep expected section.
                    section_only_candidates, _section_only_meta = _filter_verify_candidates(
                        raw_candidates=verify_candidates,
                        bgr_img=img,
                        projected_rows=projected_rows_for_filter,
                        expected_section=expected_section,
                        expected_color=None,
                        expected_xyz=target,
                        use_projected_geometry=bool(use_projected_xyz_for_filter),
                    )
                    selected = _select_pixel_candidate(
                        raw_candidates=section_only_candidates,
                        view_cx=cx,
                        view_cy=cy,
                        conf_min=float(min_conf),
                        prefer_top_px=bool(PLACE_VERIFY_V2_RECENTER_PIXEL_TOP),
                        target_uv=target_uv,
                        forbidden_uv=forbidden_uv,
                        forbid_radius_px=forbid_radius_px,
                        blocked_track_ids=blocked_verify_track_ids,
                    )
                    if selected is not None:
                        select_mode = "pixel_section_only"

            if selected is None:
                ghost_uv: tuple[int, int] | None = None
                if bool(strict_track_lock) and preferred_tid_norm is not None and state is not None:
                    track_row = state.track_memory.get(int(preferred_tid_norm), None)
                    if isinstance(track_row, dict):
                        row_uv = track_row.get("uv", None)
                        row_miss = int(track_row.get("miss_frames", 999))
                        if isinstance(row_uv, (list, tuple)) and len(row_uv) >= 2 and row_miss <= max(1, int(TRACK_MAX_MISS_FRAMES)):
                            ghost_uv = (int(row_uv[0]), int(row_uv[1]))
                if ghost_uv is None and last_target_uv is not None and (time.time() - float(last_target_seen_t)) <= float(track_reacquire_grace_s):
                    ghost_uv = (int(last_target_uv[0]), int(last_target_uv[1]))
                if ghost_uv is None:
                    centered_frames.clear()
                else:
                    gu, gv = int(ghost_uv[0]), int(ghost_uv[1])
                    cv2.circle(img_display, (gu, gv), 10, (0, 215, 255), 2)
                    cv2.drawMarker(img_display, (gu, gv), (0, 215, 255), cv2.MARKER_CROSS, 16, 2)
                    cv2.line(img_display, (cx, cy), (gu, gv), (0, 215, 255), 2)
                if bool(strict_track_lock) and preferred_tid_norm is not None:
                    cv2.putText(
                        img_display,
                        f"Verify reacquire track={int(preferred_tid_norm)}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv2.putText(img_display, "Verify target not found", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if show_window:
                    img_display = render_operator_overlay(
                        frame=img_display,
                        state=None,
                        ui_mode=UI_MODE,
                        tracks={},
                        active_track_id=(None if preferred_tid_norm is None else int(preferred_tid_norm)),
                        cx=cx,
                        cy=cy,
                        selected_uv=(None if ghost_uv is None else (int(ghost_uv[0]), int(ghost_uv[1]))),
                        status_line=(
                            "verify_recenter_waiting"
                            if not (bool(strict_track_lock) and preferred_tid_norm is not None)
                            else f"verify_recenter_waiting track={int(preferred_tid_norm)}"
                        ),
                    )
                    if _show_center_frame(show_window, img_display):
                        return None
                time.sleep(0.05)
                continue
            u, v, conf = int(selected["u"]), int(selected["v"]), float(selected["conf"])
            if bool(reset_timeout_on_first_candidate) and active_t0 is None:
                active_t0 = time.time()
                print(
                    f"[VerifyCenter] target acquired; active window "
                    f"{float(timeout_s):.1f}s (search={float(time.time() - t0):.2f}s)"
                )
            selected_uv = (u, v)
            linked_track_id_raw = selected.get("linked_track_id", None)
            linked_track_id: int | None = None
            if linked_track_id_raw is not None:
                try:
                    linked_track_id = int(linked_track_id_raw)
                except (TypeError, ValueError):
                    linked_track_id = None
            if linked_track_id is None and state is not None and TRACK_ENABLE:
                linked = nearest_visible_track_by_uv(state, u=u, v=v, max_dist_px=120.0)
                if linked is not None:
                    linked_track_id = int(linked)
            if linked_track_id is not None and int(linked_track_id) not in blocked_verify_track_ids:
                if verify_track_id != int(linked_track_id):
                    verify_track_id = int(linked_track_id)
                    verify_track_smooth_frames_left = max(0, int(PLACE_VERIFY_V2_RECENTER_TRACK_SMOOTH_FRAMES))
            last_target_uv = (int(u), int(v))
            last_target_seen_t = float(time.time())

            ex, ey = u - cx, v - cy
            cv2.circle(img_display, (u, v), 12, (0, 255, 0), 2)
            cv2.drawMarker(img_display, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
            cv2.line(img_display, (cx, cy), (u, v), (255, 0, 255), 2)
            cv2.putText(
                img_display,
                f"Verify recenter ({select_mode}) err: X={ex}px Y={ey}px "
                f"filt={int(filter_meta.get('filtered_count', 0))} "
                f"wrong_c={int(filter_meta.get('wrong_color_hits', 0))} "
                f"wrong_xy={int(filter_meta.get('wrong_xy_hits', 0))} "
                f"neg_y={int(negative_y_meta.get('negative_y_skips', 0))}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            if abs(ex) <= PX_TOL and abs(ey) <= PX_TOL and conf >= COMMIT_CONF:
                centered_frames.append((u, v))
                if len(centered_frames) >= centered_frames.maxlen:
                    lock = centered_frames[-1]
                    lock_row = dict(selected)
                    lock_row["u"] = int(lock[0])
                    lock_row["v"] = int(lock[1])
                    lock_row["conf"] = float(conf)
                    wrong_reasons = _selected_is_wrong_target(
                        selected_row=lock_row,
                        bgr_img=img,
                        projected_rows=projected_rows,
                        depth_frame=depth,
                        view_cx=cx,
                    )
                    if wrong_reasons:
                        lock_reject_count += 1
                        centered_frames.clear()
                        lock_xyz = _resolve_selected_xyz(lock_row, projected_rows, depth)
                        if state is not None:
                            state.last_verify_lock_uv = [int(lock[0]), int(lock[1])]
                            state.last_verify_lock_xyz = lock_xyz
                            state.last_verify_lock_track_id = (
                                None if linked_track_id is None else int(linked_track_id)
                            )
                            state.last_verify_lock_source = "lock_rejected"
                        reject_radius = float(max(0.0, float(PLACE_VERIFY_V2_RECENTER_DYNAMIC_BLACKLIST_PX)))
                        _append_dynamic_forbidden_uv(int(lock[0]), int(lock[1]), float(reject_radius))
                        reject_track_id = linked_track_id
                        if reject_track_id is None and state is not None and TRACK_ENABLE:
                            linked2 = nearest_visible_track_by_uv(
                                state, u=int(lock[0]), v=int(lock[1]), max_dist_px=120.0
                            )
                            if linked2 is not None:
                                reject_track_id = int(linked2)
                        if reject_track_id is not None:
                            blocked_verify_track_ids.add(int(reject_track_id))
                            if state is not None:
                                trow = state.track_memory.get(int(reject_track_id), None)
                                if trow is not None:
                                    tuv = trow.get("uv", None)
                                    if bool(use_pixel_blacklist) and isinstance(tuv, (list, tuple)) and len(tuv) >= 2:
                                        _append_dynamic_forbidden_uv(int(tuv[0]), int(tuv[1]), float(reject_radius))
                        verify_track_id = None
                        verify_track_smooth_frames_left = 0
                        retry_forbidden_uv: list[list[float]] = []
                        if forbidden_uv:
                            retry_forbidden_uv.extend(list(forbidden_uv))
                        if dynamic_forbidden_uv:
                            retry_forbidden_uv.extend(list(dynamic_forbidden_uv))
                        # Immediate handoff: pick the next non-blacklisted candidate and seed track-lock.
                        handoff_applied = False
                        next_selected = _select_pixel_candidate(
                            raw_candidates=verify_candidates,
                            view_cx=cx,
                            view_cy=cy,
                            conf_min=float(min_conf),
                            prefer_top_px=bool(PLACE_VERIFY_V2_RECENTER_PIXEL_TOP),
                            target_uv=None,
                            forbidden_uv=retry_forbidden_uv,
                            forbid_radius_px=forbid_radius_px,
                            blocked_track_ids=blocked_verify_track_ids,
                        )
                        if next_selected is not None:
                            forced_target_uv = (
                                int(next_selected.get("u", 0)),
                                int(next_selected.get("v", 0)),
                            )
                            forced_target_frames_left = max(
                                1, int(PLACE_VERIFY_V2_RECENTER_FORCED_TARGET_FRAMES)
                            )
                            handoff_applied = True
                            next_tid_raw = next_selected.get("linked_track_id", None)
                            if next_tid_raw is not None:
                                try:
                                    verify_track_id = int(next_tid_raw)
                                    verify_track_smooth_frames_left = max(
                                        0, int(PLACE_VERIFY_V2_RECENTER_TRACK_SMOOTH_FRAMES)
                                    )
                                except (TypeError, ValueError):
                                    verify_track_id = None
                                    verify_track_smooth_frames_left = 0
                        select_mode = "pixel_lock_rejected_" + "_".join(sorted(set([str(r) for r in wrong_reasons])))
                        if handoff_applied:
                            select_mode = select_mode + "_handoff"
                        cv2.putText(
                            img_display,
                            f"Verify reject after lock ({lock_reject_count}/{max_candidate_tries})",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 165, 255),
                            2,
                        )
                        if show_window:
                            img_display = render_operator_overlay(
                                frame=img_display,
                                state=None,
                                ui_mode=UI_MODE,
                                tracks={},
                                active_track_id=None,
                                cx=cx,
                                cy=cy,
                                selected_uv=(int(lock[0]), int(lock[1])),
                                status_line=select_mode,
                            )
                            if _show_center_frame(show_window, img_display):
                                return None
                        time.sleep(0.03)
                        continue
                    if show_window:
                        cv2.putText(img_display, "VERIFY LOCKED", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        img_display = render_operator_overlay(
                            frame=img_display,
                            state=None,
                            ui_mode=UI_MODE,
                            tracks={},
                            active_track_id=None,
                            cx=cx,
                            cy=cy,
                            selected_uv=selected_uv,
                            status_line="verify_recenter_locked",
                        )
                        cv2.imshow(WINDOW_NAME, img_display)
                        cv2.waitKey(max(1, int(PLACE_VERIFY_V2_RECENTER_LOCK_PAUSE_MS)))
                    if state is not None:
                        state.last_verify_lock_uv = [int(lock[0]), int(lock[1])]
                        state.last_verify_lock_xyz = _resolve_selected_xyz(lock_row, projected_rows, depth)
                        state.last_verify_lock_track_id = (
                            None if linked_track_id is None else int(linked_track_id)
                        )
                        state.last_verify_lock_source = "locked"
                    return int(lock[0]), int(lock[1])
                if show_window:
                    img_display = render_operator_overlay(
                        frame=img_display,
                        state=None,
                        ui_mode=UI_MODE,
                        tracks={},
                        active_track_id=None,
                        cx=cx,
                        cy=cy,
                        selected_uv=selected_uv,
                        status_line=f"verify_recenter_hold {len(centered_frames)}/{centered_frames.maxlen}",
                    )
                    if _show_center_frame(show_window, img_display):
                        return None
                time.sleep(0.03)
                continue
            centered_frames.clear()
            frame_idx = _maybe_apply_centering_nudge(
                arm,
                ex,
                ey,
                conf,
                frame_idx,
                detect_conf=float(DETECT_CONF),
                center_verbose=bool(CENTER_VERBOSE),
            )
            if show_window:
                img_display = render_operator_overlay(
                    frame=img_display,
                    state=None,
                    ui_mode=UI_MODE,
                    tracks={},
                    active_track_id=None,
                    cx=cx,
                    cy=cy,
                    selected_uv=selected_uv,
                    status_line="verify_recentering",
                )
                if _show_center_frame(show_window, img_display):
                    return None
    finally:
        _reset_centering_integrator()
        if show_window and (not PLACE_VERIFY_V2_RECENTER_PERSIST_WINDOW):
            cv2.destroyAllWindows()
    print(
        f"[VerifyCenter] failed: {timeout_reason} "
        f"(search={float(no_target_timeout_s):.1f}s active={float(timeout_s):.1f}s)"
    )
    return None



def _draw_center_reference_overlay(
    img_display: np.ndarray,
    cx: int,
    cy: int,
    elapsed_s: float,
    timeout_s: float | None = None,
):
    if not _CORE_BIND_READY:
        _bind_core_globals()
    timeout_used = float(CENTER_TIMEOUT_S if timeout_s is None else timeout_s)
    timeout_used = max(0.0, timeout_used)
    cv2.drawMarker(img_display, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.circle(img_display, (cx, cy), PX_TOL, (0, 255, 255), 2)
    cv2.putText(
        img_display,
        f"Centering... {elapsed_s:.1f}s / {timeout_used:.1f}s",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

def _draw_center_stability_overlay(img_display: np.ndarray, centered_count: int, required_centered_frames: int):
    if not _CORE_BIND_READY:
        _bind_core_globals()
    centered_display = int(max(0, centered_count))
    cv2.putText(
        img_display,
        f"Centered & confident! Holding: {centered_display}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    bar_width = 200
    bar_height = 20
    bar_x, bar_y = 10, 120
    cv2.rectangle(img_display, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
    filled_width = int(bar_width * centered_count / required_centered_frames)
    cv2.rectangle(img_display, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), -1)

def _draw_forbidden_uv_overlay(
    img_display: np.ndarray,
    forbidden_uv: list[list[float]] | None,
    default_radius_px: float,
):
    if not _CORE_BIND_READY:
        _bind_core_globals()
    if not forbidden_uv:
        return
    overlay = img_display.copy()
    drawn = 0
    for uv in forbidden_uv:
        if not isinstance(uv, (list, tuple)) or len(uv) < 2:
            continue
        u = int(uv[0])
        v = int(uv[1])
        radius = float(default_radius_px)
        if len(uv) >= 3:
            try:
                radius = float(uv[2])
            except (TypeError, ValueError):
                radius = float(default_radius_px)
        r = max(2, int(round(max(0.0, radius))))
        # Transparent red fill + strong red outline so blacklist zones are obvious.
        cv2.circle(overlay, (u, v), r, (0, 0, 255), -1)
        cv2.circle(img_display, (u, v), r, (0, 0, 255), 2)
        drawn += 1
    cv2.addWeighted(overlay, 0.20, img_display, 0.80, 0.0, dst=img_display)
    cv2.putText(
        img_display,
        f"Blacklist zones: {drawn}",
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
    )

def _show_center_frame(show_window: bool, img_display: np.ndarray) -> bool:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    if not show_window:
        return False
    cv2.imshow(WINDOW_NAME, img_display)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        print("User requested quit")
        cv2.destroyAllWindows()
        return True
    return False

def _handle_selected_center_candidate(
    selected: tuple[int, int, float],
    img_display: np.ndarray,
    img: np.ndarray,
    cx: int,
    cy: int,
    centered_frames: deque,
    required_centered_frames: int,
    show_window: bool,
    save_localization_dir: str | None,
    save_tag: str,
    save_annotated: bool,
    arm: Arm,
    frame_idx: int,
) -> tuple[int, tuple[int, int] | None]:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    u, v, conf = selected
    ex, ey = u - cx, v - cy
    need_visual = bool(show_window or save_localization_dir)
    if need_visual:
        try:
            if (not show_window) and save_localization_dir and np.shares_memory(img_display, img):
                img_display = img_display.copy()
        except Exception:
            pass
        cv2.line(img_display, (cx, cy), (u, v), (255, 0, 255), 2)
        cv2.putText(
            img_display,
            f"Error: X={ex}px, Y={ey}px | Conf={conf:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    if abs(ex) <= PX_TOL and abs(ey) <= PX_TOL and conf >= COMMIT_CONF:
        centered_frames.append((u, v, conf))
        centered_count = len(centered_frames)
        if need_visual:
            _draw_center_stability_overlay(img_display, centered_count, required_centered_frames)
        if centered_count < required_centered_frames:
            time.sleep(0.05)
            return frame_idx, None
        avg_u = int(np.mean([d[0] for d in centered_frames]))
        avg_v = int(np.mean([d[1] for d in centered_frames]))
        if need_visual:
            cv2.putText(img_display, "LOCKED ON TARGET!", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if show_window:
            cv2.imshow(WINDOW_NAME, img_display)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        if save_localization_dir:
            saved_path = save_localization_capture(
                capture_root=save_localization_dir,
                capture_tag=save_tag,
                raw_bgr=img,
                annotated_bgr=(img_display if save_annotated else None),
                lock_u=avg_u,
                lock_v=avg_v,
                lock_conf=conf,
            )
            print(f"[DataCapture] Saved localization image: {saved_path}")
        print(f"[Center] done at ({avg_u}, {avg_v})")
        return frame_idx, (avg_u, avg_v)
    centered_frames.clear()
    if need_visual and abs(ex) <= PX_TOL and abs(ey) <= PX_TOL:
        cv2.putText(
            img_display,
            f"Centered but conf too low ({conf:.2f} < {COMMIT_CONF:.2f})",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )
    else:
        cv2.putText(img_display, "Adjusting position...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    frame_idx = _maybe_apply_centering_nudge(
        arm,
        ex,
        ey,
        conf,
        frame_idx,
        detect_conf=float(DETECT_CONF),
        center_verbose=bool(CENTER_VERBOSE),
    )
    return frame_idx, None

def center_object_slowly(
    det,
    arm,
    per,
    required_centered_frames: int | None = None,
    show_window: bool | None = None,
    save_localization_dir: str | None = None,
    save_tag: str = "capture",
    save_annotated: bool = True,
    candidate_selector=None,
    selector_context: dict | None = None,
    state: CycleState | None = None,
):
    if not _CORE_BIND_READY:
        _bind_core_globals()
    if required_centered_frames is None:
        required_centered_frames = int(CENTERED_FRAMES_REQUIRED)
    if show_window is None:
        show_window = bool(SHOW_WINDOW)
    if show_window:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 720)
    print("[Center] start")
    _reset_centering_integrator()
    if state is not None:
        state.last_center_failure = {}
    if PICK_FILTER_BY_BASE_Y and CENTER_VERBOSE:
        print(f"[PickFilter] Enabled base-Y gate: accepting detections with estimated y <= {PICK_MAX_BASE_Y_M:.3f} m")
    t0 = time.time()
    no_detection_timeout_s = float(max(0.5, CENTER_TIMEOUT_NO_DETECTION_S))
    active_detection_timeout_s = float(max(0.5, CENTER_TIMEOUT_ACTIVE_S))
    detection_seen = False
    detect_t0: float | None = None
    last_candidate_count = 0
    last_filtered_count = 0
    last_selector_meta: dict = {}
    centered_frames = deque(maxlen=required_centered_frames)
    frame_idx = 0
    selector_context = {} if selector_context is None else selector_context
    try:
        while True:
            now = time.time()
            if not detection_seen:
                elapsed_phase_s = float(now - t0)
                phase_timeout_s = float(no_detection_timeout_s)
            else:
                detect_anchor = float(detect_t0 if detect_t0 is not None else t0)
                elapsed_phase_s = float(now - detect_anchor)
                phase_timeout_s = float(active_detection_timeout_s)
            if elapsed_phase_s >= phase_timeout_s:
                break
            detector_draw = bool(show_window and ((UI_MODE == "debug") or UI_DRAW_ALL_BOXES))
            obs = observe_scene_frame(
                det=det,
                arm=arm,
                per=per,
                draw=detector_draw,
                projected_min_conf=float(TRACK_MIN_CONF),
                state=state,
                update_tracks=True,
            )
            if obs is None:
                break
            img = obs.image_bgr
            depth = obs.depth_frame
            cx, cy = obs.image_center_uv
            img_display = obs.image_display
            candidates = obs.candidates
            filtered, rejected = _filter_pick_candidates_by_base_y(candidates, arm, per, depth)
            last_candidate_count = int(len(candidates))
            last_filtered_count = int(len(filtered))
            if rejected > 0:
                cv2.putText(
                    img_display,
                    f"PickFilter rejected {rejected} candidate(s)",
                    (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 140, 255),
                    2,
                )
            _draw_center_reference_overlay(
                img_display=img_display,
                cx=cx,
                cy=cy,
                elapsed_s=float(elapsed_phase_s),
                timeout_s=float(phase_timeout_s),
            )
            selector_meta: dict | None = None
            if callable(candidate_selector):
                selected = candidate_selector(filtered, cx, cy, selector_context)
                meta_from_context = selector_context.get("selector_meta", None) if isinstance(selector_context, dict) else None
                if isinstance(meta_from_context, dict):
                    selector_meta = dict(meta_from_context)
            else:
                selected, selector_meta = select_pick_candidate_stack_top(
                    candidates=filtered,
                    cx=int(cx),
                    cy=int(cy),
                    min_conf=float(DETECT_CONF),
                    require_track_id=False,
                    blocked_track_ids=None,
                    blocked_uvs=None,
                    top_exposed_only=bool(PICK_TOP_EXPOSED_ONLY),
                    x_overlap_min=float(PICK_TOP_EXPOSED_X_OVERLAP_MIN),
                    y_gap_px=float(PICK_TOP_EXPOSED_Y_GAP_PX),
                    fallback_mode=str(PICK_TOP_EXPOSED_FALLBACK),
                )
                if isinstance(selector_context, dict):
                    selector_context["selector_meta"] = dict(selector_meta or {})
            if isinstance(selector_meta, dict):
                last_selector_meta = dict(selector_meta)
                mode = str(selector_meta.get("selector_mode", "none"))
                candidate_count = int(selector_meta.get("eligible_count", 0))
                top_count = int(selector_meta.get("exposed_top_count", 0))
                selected_tid = selector_meta.get("selected_track_id", None)
                selected_tid_txt = "n/a" if selected_tid is None else str(int(selected_tid))
                diag_line = f"Selector={mode} cand={candidate_count} tops={top_count} id={selected_tid_txt}"
                if isinstance(selector_context, dict):
                    diag_sig = (
                        mode,
                        int(selector_meta.get("candidate_count", 0)),
                        candidate_count,
                        top_count,
                        selected_tid_txt,
                    )
                    if selector_context.get("_last_selector_diag_sig", None) != diag_sig:
                        if CENTER_VERBOSE:
                            print(f"[CenterSelect] {diag_line}")
                        selector_context["_last_selector_diag_sig"] = diag_sig
            selected_uv = None
            if not selected:
                centered_frames.clear()
                _leak_centering_integrator()
                cv2.putText(img_display, "Lost target - waiting...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if show_window:
                    active_track_id = None if state is None else state.active_target_track_id
                    wait_status = "track_waiting" if active_track_id is None else f"track_waiting id={int(active_track_id)}"
                    img_display = render_operator_overlay(
                        frame=img_display,
                        state=state,
                        ui_mode=UI_MODE,
                        tracks=({} if state is None else state.track_memory),
                        active_track_id=active_track_id,
                        cx=cx,
                        cy=cy,
                        selected_uv=None,
                        status_line=wait_status,
                    )
                time.sleep(0.2)
                if _show_center_frame(show_window, img_display):
                    return None
                continue
            if not detection_seen:
                detection_seen = True
                detect_t0 = float(time.time())
                print(
                    f"[Center] detection acquired; active centering window "
                    f"{float(active_detection_timeout_s):.1f}s"
                )
            if isinstance(selected, dict):
                selected_uv = (int(selected.get("u", 0)), int(selected.get("v", 0)))
                selected_tuple = (int(selected.get("u", 0)), int(selected.get("v", 0)), float(selected.get("conf", 0.0)))
            else:
                selected_tuple = (int(selected[0]), int(selected[1]), float(selected[2]))
                selected_uv = (int(selected_tuple[0]), int(selected_tuple[1]))
            if state is not None and selected_uv is not None:
                linked_track_id = nearest_visible_track_by_uv(state, selected_uv[0], selected_uv[1], max_dist_px=90.0)
                if linked_track_id is not None:
                    state.active_target_track_id = int(linked_track_id)
            frame_idx, lock_uv = _handle_selected_center_candidate(
                selected=selected_tuple,
                img_display=img_display,
                img=img,
                cx=cx,
                cy=cy,
                centered_frames=centered_frames,
                required_centered_frames=required_centered_frames,
                show_window=show_window,
                save_localization_dir=save_localization_dir,
                save_tag=save_tag,
                save_annotated=save_annotated,
                arm=arm,
                frame_idx=frame_idx,
            )
            if show_window:
                active_track_id = None if state is None else state.active_target_track_id
                center_status = "centering_target" if active_track_id is None else f"track_locked id={int(active_track_id)}"
                img_display = render_operator_overlay(
                    frame=img_display,
                    state=state,
                    ui_mode=UI_MODE,
                    tracks=({} if state is None else state.track_memory),
                    active_track_id=active_track_id,
                    cx=cx,
                    cy=cy,
                    selected_uv=selected_uv,
                    status_line=center_status,
                )
            if lock_uv is not None:
                return lock_uv
            if _show_center_frame(show_window, img_display):
                return None
    finally:
        _reset_centering_integrator()
        if show_window:
            cv2.destroyAllWindows()
    timeout_label = "active_detection_timeout" if detection_seen else "no_detection_timeout"
    if state is not None:
        state.last_center_failure = {
            "status": str(timeout_label),
            "detection_seen": bool(detection_seen),
            "candidate_count": int(last_candidate_count),
            "filtered_count": int(last_filtered_count),
            "selector_meta": dict(last_selector_meta),
        }
    print(
        f"[Center] failed: {timeout_label} "
        f"(no_det={float(no_detection_timeout_s):.1f}s active={float(active_detection_timeout_s):.1f}s)"
    )
    return None
__all__ = [
    "center_object_slowly",
    "center_object_on_expected_slot",
]
