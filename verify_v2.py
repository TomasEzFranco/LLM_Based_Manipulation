#!/usr/bin/env python3
"""Verification V2 owner module (extract-only)."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

_CORE_BIND_READY = False


def _bind_core_globals() -> None:
    global _CORE_BIND_READY
    import runtime_core as core
    protected = {
        '_bind_core_globals', '_CORE_BIND_READY',
        '_filter_verify_candidates', 'compute_verify_stack_min_z',
        'compute_verify_z_margin', '_filter_projected_slot_candidates',
        'collect_slot_observations', 'associate_newest_placement',
        'build_verify_expected_for_score', '_verify_stack_level_for_placement',
        'score_place_geometry', 'verify_last_place_reliability',
    }
    for name, value in core.__dict__.items():
        if name.startswith('__') or name in protected:
            continue
        globals()[name] = value
    _CORE_BIND_READY = True

def _filter_verify_candidates(
    raw_candidates: list[dict],
    bgr_img: np.ndarray,
    projected_rows: list[dict],
    expected_section: str | None,
    expected_color: str | None,
    expected_xyz: np.ndarray | None = None,
    use_projected_geometry: bool = True,
) -> tuple[list[dict], dict]:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    expected_section_name = str(expected_section or "").strip().lower()
    expected_color_name = str(expected_color or "").strip().lower()
    expected_xy: tuple[float, float] | None = None
    if isinstance(expected_xyz, np.ndarray):
        xyz = np.array(expected_xyz, dtype=float).reshape(-1)
        if xyz.size >= 2 and np.all(np.isfinite(xyz[:2])):
            expected_xy = (float(xyz[0]), float(xyz[1]))
    elif isinstance(expected_xyz, (list, tuple)):
        xyz = np.array(expected_xyz, dtype=float).reshape(-1)
        if xyz.size >= 2 and np.all(np.isfinite(xyz[:2])):
            expected_xy = (float(xyz[0]), float(xyz[1]))
    section_centers = _verify_section_y_centers() if expected_section_name else {}
    out: list[dict] = []
    section_hits = 0
    color_hits = 0
    wrong_color_hits = 0
    wrong_color_uv: list[list[int]] = []
    wrong_xy_hits = 0
    wrong_xy_uv: list[list[int]] = []
    for c in raw_candidates:
        u = int(c.get("u", 0))
        v = int(c.get("v", 0))
        row = dict(c)
        row["u"] = u
        row["v"] = v
        proj = _match_projected_row_by_uv(projected_rows, u=u, v=v) if bool(use_projected_geometry) else None

        section_ok = True
        inferred_section = None
        if expected_section_name and bool(use_projected_geometry):
            section_ok = False
            if proj is not None:
                xyz = np.array(proj.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
                if xyz.size >= 2 and np.all(np.isfinite(xyz[:2])):
                    inferred_section = _infer_section_for_base_y(float(xyz[1]), section_centers)
                    section_ok = (inferred_section == expected_section_name)
            if section_ok:
                section_hits += 1

        color_ok = True
        if expected_color_name and PLACE_VERIFY_V2_RECENTER_COLOR_FILTER:
            color_name, color_conf = classify_cube_color_patch(
                bgr_img,
                bbox_xyxy=row.get("bbox_xyxy", None),
                center_uv=(u, v),
            )
            row["color_name"] = str(color_name)
            row["color_conf"] = float(color_conf)
            if (
                float(color_conf) >= float(PLACE_VERIFY_V2_RECENTER_COLOR_MIN_CONF)
                and str(color_name).strip().lower() not in {"", "unknown", expected_color_name}
            ):
                wrong_color_hits += 1
                wrong_color_uv.append([int(u), int(v)])
            color_ok = (
                str(color_name).strip().lower() == expected_color_name
                and float(color_conf) >= float(PLACE_VERIFY_V2_RECENTER_COLOR_MIN_CONF)
            )
            if color_ok:
                color_hits += 1

        xy_ok = True
        if bool(use_projected_geometry) and expected_xy is not None and proj is not None:
            xyz = np.array(proj.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
            if xyz.size >= 2 and np.all(np.isfinite(xyz[:2])):
                d_xy_expected = float(math.hypot(float(xyz[0]) - float(expected_xy[0]), float(xyz[1]) - float(expected_xy[1])))
                row["d_xy_expected_m"] = float(d_xy_expected)
                if (
                    PLACE_VERIFY_V2_RECENTER_DISALLOW_WRONG_XY
                    and d_xy_expected > float(PLACE_VERIFY_V2_RECENTER_WRONG_XY_M)
                ):
                    xy_ok = False
                    wrong_xy_hits += 1
                    wrong_xy_uv.append([int(u), int(v)])

        if section_ok and color_ok and xy_ok:
            if proj is not None:
                row["xyz"] = list(proj.get("xyz", [np.nan, np.nan, np.nan]))
            row["inferred_section"] = inferred_section
            out.append(row)

    return out, {
        "section_hits": int(section_hits),
        "color_hits": int(color_hits),
        "wrong_color_hits": int(wrong_color_hits),
        "wrong_color_uv": wrong_color_uv,
        "wrong_xy_hits": int(wrong_xy_hits),
        "wrong_xy_uv": wrong_xy_uv,
        "filtered_count": int(len(out)),
    }


def compute_verify_stack_min_z(expected_z: float, stack_level: int) -> float | None:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    if int(stack_level) < 2:
        return None
    frac = max(0.0, min(1.0, float(PLACE_VERIFY_V2_STACK_MIN_LAYER_FRAC)))
    return float(expected_z) - (float(STACK_LEVEL_DZ_M) * frac)


def compute_verify_z_margin(stack_level: int) -> float:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    _ = stack_level
    z_margin = float(PLACE_VERIFY_V2_Z_MARGIN_M)
    return float(max(0.005, z_margin))


def _verify_stack_level_for_placement(placement: dict) -> int:
    """Return the intended post-place stack level for verify targeting."""
    try:
        placed_level = int((placement or {}).get("stack_level", 0) or 0)
    except Exception:
        placed_level = 0
    try:
        pending_raw = (placement or {}).get("pending_stack_level", None)
        pending_level = None if pending_raw is None else int(pending_raw)
    except Exception:
        pending_level = None
    if pending_level is None:
        return max(0, int(placed_level))
    return max(0, int(placed_level), int(pending_level))


def _filter_projected_slot_candidates(
    projected_rows: list[dict],
    expected_xyz: np.ndarray,
    radius_m: float,
    min_conf: float,
    max_abs_z_error_m: float,
    min_z_m: float | None = None,
    prefer_higher_z: bool = False,
) -> tuple[list[dict], int]:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    target = np.array(expected_xyz, dtype=float).reshape(-1)
    if target.size < 3 or not np.all(np.isfinite(target[:3])):
        return [], 0
    tx, ty, tz = map(float, target[:3])
    radius_m = max(0.0, float(radius_m))
    max_abs_z_error_m = max(0.0, float(max_abs_z_error_m))
    roi_rows: list[dict] = []
    projected_valid = 0
    for row in projected_rows:
        conf = float(row.get("conf", 0.0))
        if conf < float(min_conf):
            continue
        xyz = np.array(row.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
        if xyz.size < 3 or not np.all(np.isfinite(xyz[:3])):
            continue
        if min_z_m is not None and np.isfinite(float(min_z_m)) and float(xyz[2]) < float(min_z_m):
            continue
        projected_valid += 1
        d_xy = float(math.hypot(float(xyz[0]) - tx, float(xyz[1]) - ty))
        d_z = float(abs(float(xyz[2]) - tz))
        if d_xy <= radius_m and d_z <= max_abs_z_error_m:
            roi_rows.append(
                {
                    "u": int(row.get("u", 0)),
                    "v": int(row.get("v", 0)),
                    "conf": conf,
                    "track_id": row.get("track_id", None),
                    "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                    "d_xy_m": d_xy,
                    "d_z_m": d_z,
                }
            )
    if prefer_higher_z:
        roi_rows.sort(
            key=lambda row: (
                -float(row["xyz"][2]),
                float(row["d_xy_m"]),
                float(row["d_z_m"]),
                -float(row["conf"]),
            )
        )
    else:
        roi_rows.sort(key=lambda row: (float(row["d_xy_m"]), float(row["d_z_m"]), -float(row["conf"])))
    return roi_rows, projected_valid


def collect_slot_observations(
    det: YOLODetector | None,
    arm: Arm,
    per: Perception | None,
    expected_xyz: np.ndarray,
    samples: int,
    radius_m: float | None = None,
    min_conf: float | None = None,
    max_abs_z_error_m: float | None = None,
    min_z_m: float | None = None,
    prefer_higher_z: bool = False,
) -> dict:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    if radius_m is None:
        radius_m = float(PLACE_VERIFY_V2_RADIUS_M)
    if min_conf is None:
        min_conf = float(PLACE_VERIFY_MIN_CONF)
    if max_abs_z_error_m is None:
        max_abs_z_error_m = float(PLACE_VERIFY_V2_Z_MARGIN_M)
    target = np.array(expected_xyz, dtype=float).reshape(-1)
    if det is None or per is None or target.size < 3 or not np.all(np.isfinite(target[:3])):
        return {
            "samples": max(1, int(samples)),
            "hits": 0,
            "hits_ratio": 0.0,
            "valid_frames": 0,
            "projected_valid": 0,
            "min_xy_error_m": float("inf"),
            "min_z_error_m": float("inf"),
            "median_xyz": None,
            "nearest_candidates": [],
        }

    n = max(1, int(samples))
    hit_frames = 0
    valid_frames = 0
    projected_valid_total = 0
    nearest_rows: list[dict] = []
    xyz_rows: list[np.ndarray] = []
    min_xy_error_m = float("inf")
    min_z_error_m = float("inf")

    for _ in range(n):
        obs = observe_scene_frame(
            det=det,
            arm=arm,
            per=per,
            draw=False,
            projected_min_conf=float(min_conf),
        )
        if obs is None:
            break
        roi_rows, projected_valid = _filter_projected_slot_candidates(
            projected_rows=obs.projected_rows,
            expected_xyz=target,
            radius_m=radius_m,
            min_conf=min_conf,
            max_abs_z_error_m=max_abs_z_error_m,
            min_z_m=min_z_m,
            prefer_higher_z=prefer_higher_z,
        )
        projected_valid_total += int(projected_valid)
        if projected_valid > 0:
            valid_frames += 1
        if roi_rows:
            hit_frames += 1
            best = roi_rows[0]
            nearest_rows.append(best)
            xyz_rows.append(np.array(best["xyz"], dtype=float))
            min_xy_error_m = min(min_xy_error_m, float(best["d_xy_m"]))
            min_z_error_m = min(min_z_error_m, float(best["d_z_m"]))
        time.sleep(max(0.0, arm.sample_time))

    median_xyz = None
    if xyz_rows:
        median_xyz = _finite_xyz_or_none(np.median(np.array(xyz_rows, dtype=float), axis=0))

    return {
        "samples": int(n),
        "hits": int(hit_frames),
        "hits_ratio": float(hit_frames / max(1, n)),
        "valid_frames": int(valid_frames),
        "projected_valid": int(projected_valid_total),
        "min_xy_error_m": float(min_xy_error_m),
        "min_z_error_m": float(min_z_error_m),
        "median_xyz": median_xyz,
        "nearest_candidates": nearest_rows,
    }


def associate_newest_placement(placement_record: dict, post_obs: dict) -> dict:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    return {
        "object_id": placement_record.get("object_id"),
        "slot_index": placement_record.get("slot_index"),
        "expected_xyz": _finite_xyz_or_none(placement_record.get("expected_xyz")),
        "measured_xyz": _finite_xyz_or_none(post_obs.get("median_xyz")),
    }


def build_verify_expected_for_score(expected_xyz) -> tuple[np.ndarray, float, float, float]:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    expected_for_score = np.array(expected_xyz, dtype=float).reshape(-1)
    if bool(PLACE_VERIFY_V2_EXPECTED_EVAL_USE_OFFSETS):
        expected_x_offset_m = float(PLACE_VERIFY_V2_EXPECTED_X_OFFSET_M)
        expected_y_offset_m = float(PLACE_VERIFY_V2_EXPECTED_Y_OFFSET_M)
        expected_z_offset_m = float(PLACE_VERIFY_V2_EXPECTED_Z_OFFSET_M)
    else:
        expected_x_offset_m = 0.0
        expected_y_offset_m = 0.0
        expected_z_offset_m = 0.0
    expected_z_offset_m += float(PLACE_VERIFY_V2_SURFACE_Z_OFFSET_M)
    if expected_for_score.size >= 1 and np.isfinite(expected_for_score[0]):
        expected_for_score[0] = float(expected_for_score[0]) + float(expected_x_offset_m)
    if expected_for_score.size >= 2 and np.isfinite(expected_for_score[1]):
        expected_for_score[1] = float(expected_for_score[1]) + float(expected_y_offset_m)
    if expected_for_score.size >= 3 and np.isfinite(expected_for_score[2]):
        expected_for_score[2] = float(expected_for_score[2]) + float(expected_z_offset_m)
    return expected_for_score, float(expected_x_offset_m), float(expected_y_offset_m), float(expected_z_offset_m)


def score_place_geometry(
    expected_xyz: np.ndarray,
    measured_xyz: np.ndarray | None,
    hits: int,
    min_hits: int,
    xy_margin_m: float,
    z_margin_m: float,
    delta_score: float,
    delta_min: float,
    cube_edge_m: float | None = None,
    min_overlap: float | None = None,
) -> dict:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    if cube_edge_m is None:
        cube_edge_m = float(PLACE_VERIFY_V2_CUBE_EDGE_M)
    if min_overlap is None:
        min_overlap = float(PLACE_VERIFY_V2_MIN_OVERLAP)
    target = np.array(expected_xyz, dtype=float).reshape(-1)
    measured = None if measured_xyz is None else np.array(measured_xyz, dtype=float).reshape(-1)
    xy_err = float("inf")
    z_err = float("inf")
    dx_m = float("inf")
    dy_m = float("inf")
    overlap_ratio = 0.0
    if measured is not None and measured.size >= 3 and np.all(np.isfinite(measured[:3])) and np.all(np.isfinite(target[:3])):
        dx_m = float(measured[0]) - float(target[0])
        dy_m = float(measured[1]) - float(target[1])
        xy_err = float(math.hypot(dx_m, dy_m))
        z_err = float(abs(float(measured[2]) - float(target[2])))
        edge = max(1e-6, float(cube_edge_m))
        overlap_x = max(0.0, 1.0 - abs(dx_m) / edge)
        overlap_y = max(0.0, 1.0 - abs(dy_m) / edge)
        overlap_ratio = float(overlap_x * overlap_y)

    min_hits = max(1, int(min_hits))
    xy_margin_m = max(0.0, float(xy_margin_m))
    z_margin_m = max(0.0, float(z_margin_m))
    delta_min = float(delta_min)
    min_overlap = float(max(0.0, min(1.0, min_overlap)))

    within_xy = bool(np.isfinite(xy_err) and (xy_err <= xy_margin_m or overlap_ratio >= min_overlap))
    within_z = bool(np.isfinite(z_err) and z_err <= z_margin_m)

    if not np.isfinite(xy_err) or not np.isfinite(z_err) or int(hits) <= 0:
        status = "placed_uncertain_no_valid_depth"
        confirmed = False
    elif not within_xy or not within_z:
        status = "placed_mismatch_out_of_margin"
        confirmed = False
    elif int(hits) < min_hits or float(delta_score) < delta_min:
        status = "placed_uncertain_weak_delta"
        confirmed = False
    else:
        status = "placed_confirmed_geometry"
        confirmed = True

    return {
        "status": status,
        "confirmed": bool(confirmed),
        "xy_error_m": xy_err,
        "z_error_m": z_err,
        "dx_m": dx_m,
        "dy_m": dy_m,
        "overlap_ratio": overlap_ratio,
        "xy_margin_m": xy_margin_m,
        "z_margin_m": z_margin_m,
        "min_overlap": min_overlap,
    }


def _color_geometry_ok_for_commit(
    result: dict,
    *,
    fallback_xy_margin_m: float | None = None,
    fallback_z_margin_m: float | None = None,
) -> bool:
    if bool((result or {}).get("confirmed", False)):
        return True
    measured = _finite_xyz_or_none((result or {}).get("measured_xyz", None))
    if measured is None:
        return False
    try:
        xy_err = float((result or {}).get("xy_error_m", float("inf")))
    except Exception:
        xy_err = float("inf")
    try:
        z_err = float((result or {}).get("z_error_m", float("inf")))
    except Exception:
        z_err = float("inf")
    try:
        xy_margin = float((result or {}).get("effective_xy_margin_m", fallback_xy_margin_m))
    except Exception:
        xy_margin = float("inf")
    try:
        z_margin = float((result or {}).get("effective_z_margin_m", fallback_z_margin_m))
    except Exception:
        z_margin = float("inf")
    return bool(
        np.isfinite(xy_err)
        and np.isfinite(z_err)
        and np.isfinite(xy_margin)
        and np.isfinite(z_margin)
        and xy_err <= xy_margin
        and z_err <= z_margin
    )


# ============================= Centering controller =============================

def verify_last_place_reliability(
    state: CycleState,
    arm: Arm,
    per: Perception | None,
    det: YOLODetector | None,
    count_in_stats: bool = False,
) -> dict:
    if not _CORE_BIND_READY:
        _bind_core_globals()
    if not PLACE_VERIFY_V2_ENABLED:
        result = {"status": "disabled_v2", "confirmed": False}
        state.last_place_verification_v2 = dict(result)
        return result
    if not state.placed_ledger:
        result = {"status": "no_recent_placement", "confirmed": False}
        state.last_place_verification_v2 = dict(result)
        return result

    placement = state.placed_ledger[-1]
    expected = np.array(placement.get("expected_xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
    expected_for_score, expected_x_offset_m, expected_y_offset_m, expected_z_offset_m = (
        build_verify_expected_for_score(expected)
    )
    expected_section = str(placement.get("section", "")).strip().lower()
    expected_color, expected_color_source = _resolve_verify_expected_color(
        str(placement.get("cube_color", "")),
    )
    stack_level = _verify_stack_level_for_placement(placement)
    stack_min_z_m = compute_verify_stack_min_z(float(expected[2]), int(stack_level))
    prefer_top = bool(PLACE_VERIFY_V2_STACK_PREFER_TOP and int(stack_level) >= 2)
    pre_obs = dict(placement.get("pre_observation", {}))
    active_center_used = False
    recenter_attempted = True
    verify_look_pose = PLACE_LOOKING.copy()
    verify_look_pose[3] = 0.0
    verify_look_status = "fixed_place_looking"
    verify_blocked_track_ids: set[int] = set()
    state.last_verify_lock_uv = None
    state.last_verify_lock_xyz = None
    state.last_verify_lock_track_id = None
    state.last_verify_lock_source = "none"
    max_rejects = max(
        max(1, int(PLACE_VERIFY_V2_MAX_REJECTS)),
        max(1, int(PLACE_VERIFY_V2_MIN_REJECTS_PER_SESSION)),
    )
    hard_timeout_s = max(1.0, float(PLACE_VERIFY_V2_HARD_TIMEOUT_S))
    reject_count = 0
    selected_track_id: int | None = None
    last_score = {
        "status": "placed_uncertain_no_valid_depth",
        "confirmed": False,
        "xy_error_m": float("inf"),
        "z_error_m": float("inf"),
        "dx_m": float("inf"),
        "dy_m": float("inf"),
        "overlap_ratio": 0.0,
    }
    post_obs = {
        "samples": int(PLACE_VERIFY_V2_SAMPLES_POST),
        "hits": 0,
        "hits_ratio": 0.0,
        "valid_frames": 0,
        "projected_valid": 0,
        "min_xy_error_m": float("inf"),
        "min_z_error_m": float("inf"),
        "median_xyz": None,
        "nearest_candidates": [],
    }
    verify_color_result = {
        "color": "unknown",
        "conf": 0.0,
        "hits": 0,
        "samples": int(max(1, int(PLACE_VERIFY_V2_COLOR_SAMPLES))),
        "counts": {"orange": 0, "blue": 0, "unknown": 0},
        "source": "not_collected",
    }
    delta_score = 0.0
    effective_delta_min = -1.0
    xy_margin_m = float(PLACE_VERIFY_V2_XY_MARGIN_M)
    z_margin_m = compute_verify_z_margin(int(stack_level))
    if stack_level >= 2:
        xy_margin_m = max(xy_margin_m, float(PLACE_VERIFY_V2_STACK_XY_MARGIN_M))
    score_xy_margin_m = float(max(0.0, float(xy_margin_m) + float(PLACE_VERIFY_V2_MISMATCH_RELAX_XY_M)))
    score_z_margin_m = float(max(0.0, float(z_margin_m) + float(PLACE_VERIFY_V2_MISMATCH_RELAX_Z_M)))
    measured_xyz = None
    assoc = associate_newest_placement(placement, post_obs)
    loop_exit_reason = "timeout"
    verify_measurement_fallback_source = "not_run"
    expected_recenter_attempts: list[dict] = []
    top_candidate_attempts: list[dict] = []
    generic_handoff_deferred = False
    pending_stack_level = placement.get("pending_stack_level", None)

    def _ladder_log(message: str) -> None:
        if bool(PLACE_VERIFY_V2_LADDER_LOGS):
            print(str(message))

    def _candidate_track_id(row: dict) -> int | None:
        raw_tid = row.get("track_id", None)
        if raw_tid is None:
            return None
        try:
            return int(raw_tid)
        except (TypeError, ValueError):
            return None

    def _select_top_track_candidate(candidates: list[dict], cx: int, cy: int, min_conf: float) -> dict | None:
        rows: list[dict] = []
        for c in candidates:
            conf = float(c.get("conf", 0.0))
            if conf < float(min_conf):
                continue
            tid = _candidate_track_id(c)
            if tid is None or int(tid) in verify_blocked_track_ids:
                continue
            u = int(c.get("u", 0))
            v = int(c.get("v", 0))
            row = dict(c)
            row["u"] = u
            row["v"] = v
            row["conf"] = conf
            row["track_id"] = int(tid)
            row["d2_px"] = float((u - int(cx)) ** 2 + (v - int(cy)) ** 2)
            rows.append(row)
        if not rows:
            return None
        rows.sort(key=lambda r: (int(r.get("v", 0)), float(r.get("d2_px", float("inf"))), -float(r.get("conf", 0.0))))
        return rows[0]

    def _find_track_candidate(candidates: list[dict], tid: int, min_conf: float) -> dict | None:
        rows: list[dict] = []
        for c in candidates:
            conf = float(c.get("conf", 0.0))
            if conf < float(min_conf):
                continue
            c_tid = _candidate_track_id(c)
            if c_tid is None or int(c_tid) != int(tid):
                continue
            row = dict(c)
            row["u"] = int(c.get("u", 0))
            row["v"] = int(c.get("v", 0))
            row["conf"] = conf
            row["track_id"] = int(c_tid)
            rows.append(row)
        if not rows:
            return None
        rows.sort(key=lambda r: (-float(r.get("conf", 0.0)), int(r.get("v", 0))))
        return rows[0]

    def _resolve_candidate_xyz(obs_now: SceneObservation, candidate_row: dict) -> list[float] | None:
        u = int(candidate_row.get("u", 0))
        v = int(candidate_row.get("v", 0))
        proj = _match_projected_row_by_uv(obs_now.projected_rows, u=u, v=v)
        if proj is not None:
            xyz = np.array(proj.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
            if xyz.size >= 3 and np.all(np.isfinite(xyz[:3])):
                return [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        xyz_fast = estimate_base_xyz_from_uv_fast(
            arm=arm,
            per=per,
            depth_frame=obs_now.depth_frame,
            u=u,
            v=v,
        )
        xyz_fast = np.array(xyz_fast, dtype=float).reshape(-1)
        if xyz_fast.size >= 3 and np.all(np.isfinite(xyz_fast[:3])):
            return [float(xyz_fast[0]), float(xyz_fast[1]), float(xyz_fast[2])]
        return None

    def _collect_track_measurement(track_id: int, first_obs: SceneObservation, first_candidate: dict) -> dict:
        samples = max(1, int(PLACE_VERIFY_V2_SAMPLES_POST))
        hard_measure_timeout_s = max(0.8, min(3.0, samples * max(0.01, float(arm.sample_time)) * 2.0))
        xyz_rows: list[np.ndarray] = []
        min_xy_error_m = float("inf")
        min_z_error_m = float("inf")
        hits = 0
        valid_frames = 0
        projected_valid = 0
        nearest_rows: list[dict] = []
        obs_local: SceneObservation | None = first_obs
        candidate_local: dict | None = dict(first_candidate)
        t_measure0 = time.time()
        while len(xyz_rows) < samples and (time.time() - t_measure0) < hard_measure_timeout_s:
            if obs_local is None:
                obs_local = observe_scene_frame(
                    det=det,
                    arm=arm,
                    per=per,
                    draw=False,
                    projected_min_conf=float(PLACE_VERIFY_MIN_CONF),
                    state=state,
                    update_tracks=True,
                )
                if obs_local is None:
                    break
                candidate_local = _find_track_candidate(
                    obs_local.candidates,
                    int(track_id),
                    min_conf=float(PLACE_VERIFY_MIN_CONF),
                )
            if candidate_local is not None:
                projected_valid += 1
                xyz = _resolve_candidate_xyz(obs_local, candidate_local)
                if xyz is not None:
                    xyz_arr = np.array(xyz, dtype=float).reshape(-1)
                    if xyz_arr.size >= 3 and np.all(np.isfinite(xyz_arr[:3])):
                        valid_frames += 1
                        hits += 1
                        xyz_rows.append(np.array([xyz_arr[0], xyz_arr[1], xyz_arr[2]], dtype=float))
                        d_xy = float(
                            math.hypot(
                                float(xyz_arr[0]) - float(expected_for_score[0]),
                                float(xyz_arr[1]) - float(expected_for_score[1]),
                            )
                        )
                        d_z = float(abs(float(xyz_arr[2]) - float(expected_for_score[2])))
                        min_xy_error_m = min(min_xy_error_m, d_xy)
                        min_z_error_m = min(min_z_error_m, d_z)
                        nearest_rows.append(
                            {
                                "track_id": int(track_id),
                                "u": int(candidate_local.get("u", 0)),
                                "v": int(candidate_local.get("v", 0)),
                                "conf": float(candidate_local.get("conf", 0.0)),
                                "xyz": [float(xyz_arr[0]), float(xyz_arr[1]), float(xyz_arr[2])],
                                "d_xy_m": float(d_xy),
                                "d_z_m": float(d_z),
                            }
                        )
                        state.last_verify_lock_uv = [int(candidate_local.get("u", 0)), int(candidate_local.get("v", 0))]
                        state.last_verify_lock_xyz = [float(xyz_arr[0]), float(xyz_arr[1]), float(xyz_arr[2])]
                        state.last_verify_lock_track_id = int(track_id)
                        state.last_verify_lock_source = "locked"
            if SHOW_WINDOW and obs_local is not None:
                disp = obs_local.image_display
                cx_m, cy_m = obs_local.image_center_uv
                if candidate_local is not None:
                    u_m = int(candidate_local.get("u", 0))
                    v_m = int(candidate_local.get("v", 0))
                    cv2.circle(disp, (u_m, v_m), 10, (0, 255, 0), 2)
                    cv2.line(disp, (int(cx_m), int(cy_m)), (u_m, v_m), (255, 0, 255), 2)
                cv2.putText(
                    disp,
                    f"Verify measuring id={int(track_id)} hits={hits}/{samples}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                disp = render_operator_overlay(
                    frame=disp,
                    state=None,
                    ui_mode=UI_MODE,
                    tracks={},
                    active_track_id=None,
                    cx=int(cx_m),
                    cy=int(cy_m),
                    selected_uv=(None if candidate_local is None else (int(candidate_local.get("u", 0)), int(candidate_local.get("v", 0)))),
                    status_line=f"verify_measure_track id={int(track_id)}",
                )
                if _show_center_frame(SHOW_WINDOW, disp):
                    break
            obs_local = None
            candidate_local = None
            time.sleep(max(0.0, float(arm.sample_time)))

        median_xyz = None
        if xyz_rows:
            med = np.median(np.array(xyz_rows, dtype=float), axis=0)
            median_xyz = _finite_xyz_or_none(med)
        return {
            "samples": int(samples),
            "hits": int(hits),
            "hits_ratio": float(hits / max(1, samples)),
            "valid_frames": int(valid_frames),
            "projected_valid": int(projected_valid),
            "min_xy_error_m": float(min_xy_error_m),
            "min_z_error_m": float(min_z_error_m),
            "median_xyz": median_xyz,
            "nearest_candidates": nearest_rows,
        }

    def _collect_track_color_samples(track_id: int, first_obs: SceneObservation, first_candidate: dict) -> dict:
        samples = max(1, int(PLACE_VERIFY_V2_COLOR_SAMPLES))
        min_color_conf = max(0.0, float(PLACE_VERIFY_V2_COLOR_MIN_CONF))
        min_hits_local = max(1, int(PLACE_VERIFY_V2_COLOR_MIN_HITS))
        hard_timeout_s = max(0.6, min(3.0, samples * max(0.01, float(arm.sample_time)) * 2.0))
        counts = {"orange": 0, "blue": 0, "unknown": 0}
        score_by_color = {"orange": 0.0, "blue": 0.0}
        hits = 0
        captured = 0
        obs_local: SceneObservation | None = first_obs
        candidate_local: dict | None = dict(first_candidate)
        t0 = time.time()
        while captured < samples and (time.time() - t0) < hard_timeout_s:
            if obs_local is None:
                obs_local = observe_scene_frame(
                    det=det,
                    arm=arm,
                    per=per,
                    draw=False,
                    projected_min_conf=float(PLACE_VERIFY_MIN_CONF),
                    state=state,
                    update_tracks=True,
                )
                if obs_local is None:
                    break
                candidate_local = _find_track_candidate(
                    obs_local.candidates,
                    int(track_id),
                    min_conf=float(PLACE_VERIFY_MIN_CONF),
                )
            if candidate_local is not None:
                color_name_raw, color_conf = classify_cube_color_patch(
                    obs_local.image_bgr,
                    bbox_xyxy=candidate_local.get("bbox_xyxy", None),
                    center_uv=(int(candidate_local.get("u", 0)), int(candidate_local.get("v", 0))),
                )
                color_name = str(color_name_raw).strip().lower()
                if color_name not in {"orange", "blue"}:
                    color_name = "unknown"
                if color_name in {"orange", "blue"} and float(color_conf) >= min_color_conf:
                    counts[color_name] = int(counts.get(color_name, 0)) + 1
                    score_by_color[color_name] = float(score_by_color.get(color_name, 0.0)) + float(color_conf)
                    hits += 1
                else:
                    counts["unknown"] = int(counts.get("unknown", 0)) + 1
                captured += 1
            obs_local = None
            candidate_local = None
            time.sleep(max(0.0, float(arm.sample_time)))

        best_color = "unknown"
        best_conf = 0.0
        if hits >= int(min_hits_local):
            if float(score_by_color.get("orange", 0.0)) >= float(score_by_color.get("blue", 0.0)):
                best_color = "orange"
            else:
                best_color = "blue"
            best_count = max(1, int(counts.get(best_color, 0)))
            best_conf = float(score_by_color.get(best_color, 0.0)) / float(best_count)
        return {
            "color": str(best_color),
            "conf": float(best_conf),
            "hits": int(hits),
            "samples": int(samples),
            "counts": {
                "orange": int(counts.get("orange", 0)),
                "blue": int(counts.get("blue", 0)),
                "unknown": int(counts.get("unknown", 0)),
            },
            "source": "verify_track_samples",
        }

    def _score_track_measurement(
        *,
        track_id: int,
        first_obs: SceneObservation,
        first_candidate: dict,
        source: str,
    ) -> bool:
        nonlocal active_center_used, post_obs, assoc, measured_xyz, delta_score, effective_delta_min
        nonlocal last_score, loop_exit_reason, verify_color_result, selected_track_id
        nonlocal verify_measurement_fallback_source
        active_center_used = True
        selected_track_id = int(track_id)
        post_obs = _collect_track_measurement(
            track_id=int(track_id),
            first_obs=first_obs,
            first_candidate=first_candidate,
        )
        verify_color_result = _collect_track_color_samples(
            track_id=int(track_id),
            first_obs=first_obs,
            first_candidate=first_candidate,
        )
        assoc = associate_newest_placement(placement, post_obs)
        measured_xyz = assoc.get("measured_xyz")
        pre_hits_ratio = float(pre_obs.get("hits_ratio", 0.0)) if pre_obs else 0.0
        post_hits_ratio = float(post_obs.get("hits_ratio", 0.0))
        pre_near = float(pre_obs.get("min_xy_error_m", float("inf"))) if pre_obs else float("inf")
        post_near = float(post_obs.get("min_xy_error_m", float("inf")))
        if not np.isfinite(pre_near):
            pre_near = float(max(0.001, PLACE_VERIFY_V2_RADIUS_M))
        if not np.isfinite(post_near):
            post_near = float(max(0.001, PLACE_VERIFY_V2_RADIUS_M))
        delta_score = (post_hits_ratio - pre_hits_ratio) + max(
            0.0, (pre_near - post_near) / max(0.001, float(PLACE_VERIFY_V2_RADIUS_M))
        )
        effective_delta_min = -1.0
        last_score = score_place_geometry(
            expected_xyz=expected_for_score,
            measured_xyz=(None if measured_xyz is None else np.array(measured_xyz, dtype=float)),
            hits=int(post_obs.get("hits", 0)),
            min_hits=max(1, int(PLACE_VERIFY_V2_MIN_HITS)),
            xy_margin_m=float(score_xy_margin_m),
            z_margin_m=float(score_z_margin_m),
            delta_score=float(delta_score),
            delta_min=float(effective_delta_min),
            cube_edge_m=float(PLACE_VERIFY_V2_CUBE_EDGE_M),
            min_overlap=float(PLACE_VERIFY_V2_MIN_OVERLAP),
        )
        verify_measurement_fallback_source = str(source)
        if bool(last_score.get("confirmed", False)):
            loop_exit_reason = "confirmed"
        elif str(loop_exit_reason).strip() in {
            "timeout",
            "timeout_or_uncertain",
            "no_track_candidate_timeout",
            "reject_cap",
            "slot_scan_primary_unconfirmed",
            "filtered_center_unconfirmed",
        }:
            loop_exit_reason = str(source)
        return bool(last_score.get("confirmed", False))

    def _observe_track_candidate(track_id: int) -> tuple[SceneObservation | None, dict | None]:
        obs_now = observe_scene_frame(
            det=det,
            arm=arm,
            per=per,
            draw=False,
            projected_min_conf=float(PLACE_VERIFY_MIN_CONF),
            state=state,
            update_tracks=True,
        )
        if obs_now is None:
            return None, None
        row = _find_track_candidate(
            obs_now.candidates,
            int(track_id),
            min_conf=float(PLACE_VERIFY_MIN_CONF),
        )
        return obs_now, row

    def _score_locked_track_id(track_id: int | None, source: str) -> bool:
        if track_id is None:
            return False
        try:
            tid_i = int(track_id)
        except Exception:
            return False
        obs_now, row = _observe_track_candidate(int(tid_i))
        if obs_now is None or row is None:
            _ladder_log(
                f"[PlaceVerifyRetry] source={source} track_missing track_id={int(tid_i)}"
            )
            return False
        return _score_track_measurement(
            track_id=int(tid_i),
            first_obs=obs_now,
            first_candidate=row,
            source=str(source),
        )

    def _side_pixel_ok(u_px: int, img_cx: int) -> bool:
        section_name = str(expected_section or "").strip().lower()
        if not bool(PLACE_VERIFY_V2_SECTION_PIXEL_GATE):
            return True
        margin_px = max(0, int(PLACE_VERIFY_V2_SECTION_PIXEL_MARGIN_PX))
        if section_name == str(SECTION_LEFT_NAME).strip().lower():
            return int(u_px) <= int(img_cx + margin_px)
        if section_name == str(SECTION_RIGHT_NAME).strip().lower():
            return int(u_px) >= int(img_cx - margin_px)
        return True

    def _linked_track_id_for_candidate(candidate: dict) -> int | None:
        tid = _candidate_track_id(candidate)
        if tid is not None:
            return int(tid)
        if not bool(TRACK_ENABLE):
            return None
        try:
            linked = nearest_visible_track_by_uv(
                state,
                u=int(candidate.get("u", 0)),
                v=int(candidate.get("v", 0)),
                max_dist_px=120.0,
            )
        except Exception:
            linked = None
        return None if linked is None else int(linked)

    def _collect_top_side_track_candidates(limit: int) -> list[dict]:
        limit_i = max(0, int(limit))
        if limit_i <= 0:
            return []
        obs_now = observe_scene_frame(
            det=det,
            arm=arm,
            per=per,
            draw=False,
            projected_min_conf=float(PLACE_VERIFY_MIN_CONF),
            state=state,
            update_tracks=True,
        )
        if obs_now is None:
            return []
        cx_now, cy_now = obs_now.image_center_uv
        rows: list[dict] = []
        seen_tids: set[int] = set()
        for candidate in list(obs_now.candidates or []):
            conf = float(candidate.get("conf", 0.0))
            if conf < float(PLACE_VERIFY_MIN_CONF):
                continue
            u = int(candidate.get("u", 0))
            v = int(candidate.get("v", 0))
            if not _side_pixel_ok(u, int(cx_now)):
                continue
            tid = _linked_track_id_for_candidate(candidate)
            if tid is None or int(tid) in seen_tids or int(tid) in verify_blocked_track_ids:
                continue
            seen_tids.add(int(tid))
            color_name = "unknown"
            color_conf = 0.0
            if bool(PLACE_VERIFY_V2_LADDER_LOGS):
                try:
                    color_name_raw, color_conf_raw = classify_cube_color_patch(
                        obs_now.image_bgr,
                        bbox_xyxy=candidate.get("bbox_xyxy", None),
                        center_uv=(u, v),
                    )
                    color_name = str(color_name_raw).strip().lower()
                    color_conf = float(color_conf_raw)
                except Exception:
                    color_name = "unknown"
                    color_conf = 0.0
            row = dict(candidate)
            row["u"] = int(u)
            row["v"] = int(v)
            row["conf"] = float(conf)
            row["track_id"] = int(tid)
            row["d2_px"] = float((int(u) - int(cx_now)) ** 2 + (int(v) - int(cy_now)) ** 2)
            row["color_name"] = str(color_name)
            row["color_conf"] = float(color_conf)
            rows.append(row)
        rows.sort(
            key=lambda row: (
                int(row.get("v", 0)) if bool(PLACE_VERIFY_V2_RECENTER_PIXEL_TOP) else float(row.get("d2_px", float("inf"))),
                float(row.get("d2_px", float("inf"))),
                -float(row.get("conf", 0.0)),
            )
        )
        return rows[:limit_i]

    def _run_expected_recenter_attempt(attempt_i: int, total_attempts: int) -> bool:
        nonlocal active_center_used, selected_track_id
        source = "filtered_center" if int(attempt_i) <= 0 else f"filtered_center_retry_{int(attempt_i)}"
        _ladder_log(
            f"[PlaceVerifyRetry] attempt={int(attempt_i) + 1}/{int(total_attempts)} "
            f"stage=expected_slot_recenter section={expected_section or 'unknown'} "
            f"stack_level={int(stack_level)} source={source}"
        )
        center_min_conf = float(min(DETECT_CONF, PLACE_VERIFY_MIN_CONF))
        centered_uv = center_object_on_expected_slot(
            det=det,
            arm=arm,
            per=per,
            expected_xyz=expected_for_score,
            timeout_s=float(PLACE_VERIFY_V2_ACTIVE_CENTER_TIMEOUT_S),
            required_centered_frames=3,
            min_conf=float(center_min_conf),
            radius_m=max(float(PLACE_VERIFY_V2_RADIUS_M), float(PLACE_VERIFY_RADIUS_M)),
            show_window=bool(SHOW_WINDOW),
            stack_level=int(stack_level),
            min_z_m=stack_min_z_m,
            expected_section=expected_section,
            expected_color=expected_color,
            state=state,
            blocked_track_ids_seed=verify_blocked_track_ids,
            use_pixel_blacklist=False,
            target_mode_override="filtered_first",
            apply_lock_wrong_xy_gate=False,
            use_projected_xyz_for_filter=False,
            no_target_timeout_s=float(PLACE_VERIFY_V2_ACTIVE_CENTER_TIMEOUT_S),
            reset_timeout_on_first_candidate=True,
        )
        attempt_row = {
            "attempt": int(attempt_i) + 1,
            "source": str(source),
            "centered_uv": None if centered_uv is None else [int(centered_uv[0]), int(centered_uv[1])],
            "track_id": None,
            "status": "target_not_found",
        }
        if centered_uv is None:
            expected_recenter_attempts.append(dict(attempt_row))
            if int(attempt_i) <= 0 or bool(PLACE_VERIFY_V2_LADDER_LOGS):
                print(
                    f"[PlaceVerifyPrimary] source={source} target_not_found "
                    f"section={expected_section or 'unknown'} color={expected_color or 'unknown'}"
                )
            return False
        active_center_used = True
        try:
            selected_track_id = (
                None
                if state.last_verify_lock_track_id is None
                else int(state.last_verify_lock_track_id)
            )
        except Exception:
            selected_track_id = None
        attempt_row["track_id"] = selected_track_id
        if selected_track_id is not None:
            confirmed = _score_locked_track_id(
                selected_track_id,
                source=f"{source}_track_measure",
            )
            attempt_row["status"] = str(last_score.get("status", "unknown"))
            expected_recenter_attempts.append(dict(attempt_row))
            return bool(confirmed)

        centered_post_obs = collect_slot_observations(
            det=det,
            arm=arm,
            per=per,
            expected_xyz=expected_for_score,
            samples=max(1, int(PLACE_VERIFY_V2_SAMPLES_POST)),
            radius_m=max(0.0, float(PLACE_VERIFY_V2_RADIUS_M)),
            min_conf=float(PLACE_VERIFY_MIN_CONF),
            max_abs_z_error_m=max(0.01, float(score_z_margin_m) * 2.0),
            min_z_m=stack_min_z_m,
            prefer_higher_z=bool(prefer_top),
        )
        confirmed = _score_slot_observation(
            centered_post_obs,
            source=f"{source}_slot_scan",
            confirmed_exit_reason=f"{source}_confirmed",
            unconfirmed_exit_reason=f"{source}_unconfirmed",
        )
        attempt_row["status"] = str(last_score.get("status", "unknown"))
        expected_recenter_attempts.append(dict(attempt_row))
        return bool(confirmed)

    def _run_top_side_candidate_checks() -> bool:
        nonlocal selected_track_id
        candidates = _collect_top_side_track_candidates(int(PLACE_VERIFY_V2_TOP_CANDIDATE_CHECKS))
        if not candidates:
            _ladder_log(
                f"[PlaceVerifyTop2] no_side_candidates section={expected_section or 'unknown'} "
                f"limit={int(PLACE_VERIFY_V2_TOP_CANDIDATE_CHECKS)}"
            )
            return False
        total = len(candidates)
        for idx, candidate in enumerate(candidates, start=1):
            tid = int(candidate.get("track_id"))
            _ladder_log(
                f"[PlaceVerifyTop2] candidate_i={idx}/{total} track={tid} "
                f"uv=({int(candidate.get('u', 0))},{int(candidate.get('v', 0))}) "
                f"conf={float(candidate.get('conf', 0.0)):.3f} "
                f"color={candidate.get('color_name', 'unknown')} "
                f"color_conf={float(candidate.get('color_conf', 0.0)):.3f}"
            )
            centered_uv = center_object_on_expected_slot(
                det=det,
                arm=arm,
                per=per,
                expected_xyz=expected_for_score,
                timeout_s=float(PLACE_VERIFY_V2_ACTIVE_CENTER_TIMEOUT_S),
                required_centered_frames=3,
                min_conf=float(min(DETECT_CONF, PLACE_VERIFY_MIN_CONF)),
                radius_m=max(float(PLACE_VERIFY_V2_RADIUS_M), float(PLACE_VERIFY_RADIUS_M)),
                show_window=bool(SHOW_WINDOW),
                stack_level=int(stack_level),
                min_z_m=stack_min_z_m,
                expected_section=expected_section,
                expected_color=expected_color,
                state=state,
                blocked_track_ids_seed=verify_blocked_track_ids,
                use_pixel_blacklist=False,
                target_mode_override="top_first",
                apply_lock_wrong_xy_gate=False,
                use_projected_xyz_for_filter=False,
                no_target_timeout_s=float(PLACE_VERIFY_V2_ACTIVE_CENTER_TIMEOUT_S),
                reset_timeout_on_first_candidate=True,
                preferred_track_id=int(tid),
                strict_track_lock=True,
            )
            row = {
                "candidate_i": int(idx),
                "track_id": int(tid),
                "uv": [int(candidate.get("u", 0)), int(candidate.get("v", 0))],
                "centered_uv": None if centered_uv is None else [int(centered_uv[0]), int(centered_uv[1])],
                "status": "target_not_found",
            }
            if centered_uv is None:
                top_candidate_attempts.append(dict(row))
                continue
            selected_track_id = int(tid)
            confirmed = _score_locked_track_id(
                int(tid),
                source=f"top_side_candidate_{int(idx)}_track_measure",
            )
            row["status"] = str(last_score.get("status", "unknown"))
            row["measured_xyz"] = _finite_xyz_or_none(measured_xyz)
            row["xy_error_m"] = float(last_score.get("xy_error_m", float("inf")))
            row["z_error_m"] = float(last_score.get("z_error_m", float("inf")))
            top_candidate_attempts.append(dict(row))
            _ladder_log(
                f"[PlaceVerifyTop2] candidate_i={idx}/{total} track={tid} "
                f"status={row['status']} measured={row.get('measured_xyz')} "
                f"err_xy={float(row['xy_error_m']):.3f} err_z={float(row['z_error_m']):.3f}"
            )
            if bool(confirmed):
                return True
            verify_blocked_track_ids.add(int(tid))
        return False

    def _score_slot_observation(
        slot_post_obs: dict,
        *,
        source: str,
        confirmed_exit_reason: str,
        unconfirmed_exit_reason: str,
    ) -> bool:
        nonlocal post_obs, assoc, measured_xyz, delta_score, effective_delta_min, last_score
        nonlocal loop_exit_reason, verify_measurement_fallback_source
        slot_assoc = associate_newest_placement(placement, slot_post_obs)
        slot_xyz = slot_assoc.get("measured_xyz", None)
        pre_hits_ratio = float(pre_obs.get("hits_ratio", 0.0)) if pre_obs else 0.0
        post_hits_ratio = float(slot_post_obs.get("hits_ratio", 0.0))
        pre_near = float(pre_obs.get("min_xy_error_m", float("inf"))) if pre_obs else float("inf")
        post_near = float(slot_post_obs.get("min_xy_error_m", float("inf")))
        if not np.isfinite(pre_near):
            pre_near = float(max(0.001, PLACE_VERIFY_V2_RADIUS_M))
        if not np.isfinite(post_near):
            post_near = float(max(0.001, PLACE_VERIFY_V2_RADIUS_M))
        delta_score = (post_hits_ratio - pre_hits_ratio) + max(
            0.0, (pre_near - post_near) / max(0.001, float(PLACE_VERIFY_V2_RADIUS_M))
        )
        effective_delta_min = -1.0
        post_obs = dict(slot_post_obs)
        assoc = dict(slot_assoc)
        measured_xyz = _finite_xyz_or_none(slot_xyz)
        last_score = score_place_geometry(
            expected_xyz=expected_for_score,
            measured_xyz=(None if measured_xyz is None else np.array(measured_xyz, dtype=float)),
            hits=int(post_obs.get("hits", 0)),
            min_hits=max(1, int(PLACE_VERIFY_V2_MIN_HITS)),
            xy_margin_m=float(score_xy_margin_m),
            z_margin_m=float(score_z_margin_m),
            delta_score=float(delta_score),
            delta_min=float(effective_delta_min),
            cube_edge_m=float(PLACE_VERIFY_V2_CUBE_EDGE_M),
            min_overlap=float(PLACE_VERIFY_V2_MIN_OVERLAP),
        )
        verify_measurement_fallback_source = str(source)
        if bool(last_score.get("confirmed", False)):
            loop_exit_reason = str(confirmed_exit_reason)
        elif str(loop_exit_reason).strip() in {
            "timeout",
            "timeout_or_uncertain",
            "no_track_candidate_timeout",
            "reject_cap",
        }:
            loop_exit_reason = str(unconfirmed_exit_reason)
        meas = _finite_xyz_or_none(measured_xyz)
        meas_text = "n/a" if meas is None else f"({meas[0]:.3f},{meas[1]:.3f},{meas[2]:.3f})"
        print(
            f"[PlaceVerifyPrimary] source={str(source)} hits={int(post_obs.get('hits', 0))}/"
            f"{int(post_obs.get('samples', 0))} measured={meas_text} "
            f"err_xy={float(last_score.get('xy_error_m', float('inf'))):.3f} "
            f"err_z={float(last_score.get('z_error_m', float('inf'))):.3f} "
            f"status={last_score.get('status')} confirmed={bool(last_score.get('confirmed', False))}"
        )
        return bool(last_score.get("confirmed", False))

    if det is None or per is None:
        result = {
            "status": "placed_uncertain_no_valid_depth",
            "confirmed": False,
            "object_id": placement.get("object_id"),
            "slot_index": placement.get("slot_index"),
            "expected_section": expected_section,
            "expected_color": expected_color,
            "expected_color_source": expected_color_source,
            "expected_xyz": _finite_xyz_or_none(expected),
            "expected_xyz_eval": _finite_xyz_or_none(expected_for_score),
            "expected_x_offset_m": float(expected_x_offset_m),
            "expected_y_offset_m": float(expected_y_offset_m),
            "expected_z_offset_m": float(expected_z_offset_m),
            "measured_xyz": None,
            "measured_color": "unknown",
            "measured_color_conf": 0.0,
            "measured_color_hits": 0,
            "measured_color_samples": int(max(1, int(PLACE_VERIFY_V2_COLOR_SAMPLES))),
            "measured_color_counts": {"orange": 0, "blue": 0, "unknown": 0},
            "measured_color_source": "disabled",
            "xy_error_m": float("inf"),
            "z_error_m": float("inf"),
            "dx_m": float("inf"),
            "dy_m": float("inf"),
            "overlap_ratio": 0.0,
            "hits": 0,
            "samples": int(PLACE_VERIFY_V2_SAMPLES_POST),
            "delta_score": 0.0,
            "effective_delta_min": -1.0,
            "effective_xy_margin_m": float(score_xy_margin_m),
            "effective_z_margin_m": float(score_z_margin_m),
            "stack_level": int(stack_level),
            "active_center_used": False,
            "verify_recenter_attempted": bool(recenter_attempted),
            "verify_look_pose": _finite_xyz_or_none(verify_look_pose[:3]),
            "verify_look_status": "verify_dependencies_missing",
            "verify_target_mode": "expected_slot_first",
            "verify_stack_min_z_m": (None if stack_min_z_m is None else float(stack_min_z_m)),
            "verify_stack_prefer_top": bool(prefer_top),
            "verify_blocked_track_ids": [],
            "verify_reject_count": 0,
            "verify_max_rejects": int(max_rejects),
            "verify_selected_track_id": None,
            "verify_exit_reason": "dependencies_missing",
            "verify_last_lock_uv": None,
            "verify_last_lock_xyz": None,
            "verify_last_lock_track_id": None,
            "verify_last_lock_source": "none",
            "verify_measurement_fallback_source": "disabled",
            "verify_mismatch_recenter_used": False,
            "verify_mismatch_blocked_track": None,
            "verify_mismatch_blocked_uv": None,
            "pre_observation": pre_obs,
            "post_observation": post_obs,
        }
        state.last_place_verification_v2 = dict(result)
        state.last_place_verification = dict(result)
        placement["verify_result"] = dict(result)
        return result

    arm.goto_task_space(
        verify_look_pose,
        duration=max(0.25, float(PLACE_VERIFY_V2_LOOK_MOVE_S)),
        label="verify_v2_look",
    )
    if PLACE_VERIFY_V2_SETTLE_S > 0:
        time.sleep(max(0.0, float(PLACE_VERIFY_V2_SETTLE_S)))

    if bool(PLACE_VERIFY_V2_SLOT_SCAN_FIRST):
        primary_post_obs = collect_slot_observations(
            det=det,
            arm=arm,
            per=per,
            expected_xyz=expected_for_score,
            samples=max(1, int(PLACE_VERIFY_V2_SAMPLES_POST)),
            radius_m=max(0.0, float(PLACE_VERIFY_V2_RADIUS_M)),
            min_conf=float(PLACE_VERIFY_MIN_CONF),
            max_abs_z_error_m=max(0.01, float(score_z_margin_m) * 2.0),
            min_z_m=stack_min_z_m,
            prefer_higher_z=bool(prefer_top),
        )
        _score_slot_observation(
            primary_post_obs,
            source="slot_scan_primary",
            confirmed_exit_reason="slot_scan_primary_confirmed",
            unconfirmed_exit_reason="slot_scan_primary_unconfirmed",
        )

    if (
        not bool(last_score.get("confirmed", False))
        and bool(PLACE_VERIFY_V2_ACTIVE_CENTER_ON_WEAK or PLACE_VERIFY_V2_ALWAYS_RECENTER)
    ):
        recenter_total = 1 + max(0, int(PLACE_VERIFY_V2_EXPECTED_SLOT_RETRIES))
        for attempt_i in range(int(recenter_total)):
            if _run_expected_recenter_attempt(int(attempt_i), int(recenter_total)):
                break

    if (
        not bool(last_score.get("confirmed", False))
        and int(PLACE_VERIFY_V2_TOP_CANDIDATE_CHECKS) > 0
    ):
        _run_top_side_candidate_checks()

    def _verify_on_locked_candidate(
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
        _ = (collect_track_measurement, distance_to_blocked_xyz, blocked_xyzs, xy_margin_m, z_margin_m)
        _score_track_measurement(
            track_id=int(track_id),
            first_obs=obs,
            first_candidate=selected_row,
            source="generic_track_handoff",
        )
        if bool(last_score.get("confirmed", False)):
            return {
                "decision": "accept",
                "reason": "placed_confirmed_geometry",
                "selected_xyz": _finite_xyz_or_none(measured_xyz),
                "exit_reason": "confirmed",
            }
        if str(last_score.get("status", "")) == "placed_mismatch_out_of_margin":
            return {
                "decision": "reject",
                "reason": "placed_mismatch_out_of_margin",
                # Keep this reject track-scoped only. Using a full XYZ blacklist here can
                # mask the truly correct nearby top cube when two candidates are close.
                "blocked_xyz": None,
                "selected_xyz": _finite_xyz_or_none(measured_xyz),
            }
        return {
            "decision": "continue",
            "reason": str(last_score.get("status", "placed_uncertain_no_valid_depth")),
            "selected_xyz": _finite_xyz_or_none(measured_xyz),
        }

    should_defer_generic_to_hydrate = bool(
        pending_stack_level is not None
        and PLACE_VERIFY_V2_HYDRATE_FALLBACK_ENABLED
        and PLACE_VERIFY_V2_DEFER_GENERIC_HANDOFF_TO_HYDRATE
    )
    if not bool(last_score.get("confirmed", False)) and bool(should_defer_generic_to_hydrate):
        generic_handoff_deferred = True
        loop_exit_reason = "defer_hydrate_fallback"
        verify_measurement_fallback_source = "expected_slot_ladder"
        _ladder_log(
            f"[PlaceVerifyGenericHandoff] deferred_to_hydrate section={expected_section or 'unknown'} "
            f"pending_level={pending_stack_level} status={last_score.get('status')}"
        )

    if not bool(last_score.get("confirmed", False)) and not bool(generic_handoff_deferred):
        _ladder_log(
            f"[PlaceVerifyGenericHandoff] begin reason=expected_ladder_failed "
            f"section={expected_section or 'unknown'} status={last_score.get('status')}"
        )
        session = run_track_handoff_session(
            state=state,
            arm=arm,
            per=per,
            det=det,
            reject_cap=int(max_rejects),
            hard_timeout_s=float(hard_timeout_s),
            xy_margin_m=float(xy_margin_m),
            z_margin_m=float(z_margin_m),
            blocked_track_id=None,
            blocked_xyz=None,
            blocked_uv=None,
            status_prefix="verify",
            log_prefix="[PlaceVerifyV2]",
            disable_stable_gate=bool(PLACE_VERIFY_V2_DISABLE_STABLE_GATE),
            track_stable_frames_override=1,
            on_locked_candidate=_verify_on_locked_candidate,
            reject_below_base_y_m=(
                float(PLACE_VERIFY_V2_MIN_TRACK_Y_M)
                if bool(PLACE_VERIFY_V2_AVOID_NEGATIVE_Y)
                else None
            ),
        )
        reject_count = int(session.get("reject_count", 0))
        selected_track_id = session.get("selected_track_id", None)
        try:
            verify_blocked_track_ids = {int(x) for x in list(session.get("blocked_track_ids", []))}
        except Exception:
            verify_blocked_track_ids = set()
        loop_exit_reason = str(session.get("exit_reason", loop_exit_reason))
        if loop_exit_reason == "timeout":
            loop_exit_reason = "timeout_or_uncertain"
        verify_measurement_fallback_source = "disabled_track_id_only"

    result = {
        "status": str(last_score.get("status", "placed_uncertain_no_valid_depth")),
        "confirmed": bool(last_score.get("confirmed", False)),
        "object_id": placement.get("object_id"),
        "slot_index": placement.get("slot_index"),
        "expected_section": expected_section,
        "expected_color": expected_color,
        "expected_color_source": expected_color_source,
        "expected_xyz": _finite_xyz_or_none(assoc.get("expected_xyz")),
        "expected_xyz_eval": _finite_xyz_or_none(expected_for_score),
        "expected_x_offset_m": float(expected_x_offset_m),
        "expected_y_offset_m": float(expected_y_offset_m),
        "expected_z_offset_m": float(expected_z_offset_m),
        "measured_xyz": _finite_xyz_or_none(measured_xyz),
        "measured_color": str(verify_color_result.get("color", "unknown")).strip().lower(),
        "measured_color_conf": float(verify_color_result.get("conf", 0.0)),
        "measured_color_hits": int(verify_color_result.get("hits", 0)),
        "measured_color_samples": int(verify_color_result.get("samples", max(1, int(PLACE_VERIFY_V2_COLOR_SAMPLES)))),
        "measured_color_counts": {
            "orange": int((verify_color_result.get("counts", {}) or {}).get("orange", 0)),
            "blue": int((verify_color_result.get("counts", {}) or {}).get("blue", 0)),
            "unknown": int((verify_color_result.get("counts", {}) or {}).get("unknown", 0)),
        },
        "measured_color_source": str(verify_color_result.get("source", "verify_track_samples")),
        "xy_error_m": float(last_score.get("xy_error_m", float("inf"))),
        "z_error_m": float(last_score.get("z_error_m", float("inf"))),
        "dx_m": float(last_score.get("dx_m", float("inf"))),
        "dy_m": float(last_score.get("dy_m", float("inf"))),
        "overlap_ratio": float(last_score.get("overlap_ratio", 0.0)),
        "hits": int(post_obs.get("hits", 0)),
        "samples": int(post_obs.get("samples", 0)),
        "delta_score": float(delta_score),
        "effective_delta_min": float(effective_delta_min),
        "effective_xy_margin_m": float(score_xy_margin_m),
        "effective_z_margin_m": float(score_z_margin_m),
        "stack_level": int(stack_level),
        "active_center_used": bool(active_center_used),
        "verify_recenter_attempted": bool(recenter_attempted),
        "verify_look_pose": _finite_xyz_or_none(verify_look_pose[:3]),
        "verify_look_status": str(verify_look_status),
        "verify_target_mode": (
            "expected_slot_first"
            if bool(PLACE_VERIFY_V2_SLOT_SCAN_FIRST)
            else "track_handoff_section_filtered"
        ),
        "verify_stack_min_z_m": (None if stack_min_z_m is None else float(stack_min_z_m)),
        "verify_stack_prefer_top": bool(prefer_top),
        "verify_blocked_track_ids": sorted([int(tid) for tid in verify_blocked_track_ids]),
        "verify_reject_count": int(reject_count),
        "verify_max_rejects": int(max_rejects),
        "verify_expected_recenter_attempts": list(expected_recenter_attempts),
        "verify_top_candidate_attempts": list(top_candidate_attempts),
        "verify_generic_handoff_deferred": bool(generic_handoff_deferred),
        "verify_selected_track_id": (None if selected_track_id is None else int(selected_track_id)),
        "verify_exit_reason": str(loop_exit_reason),
        "verify_last_lock_uv": (None if state.last_verify_lock_uv is None else list(state.last_verify_lock_uv)),
        "verify_last_lock_xyz": (None if state.last_verify_lock_xyz is None else list(state.last_verify_lock_xyz)),
        "verify_last_lock_track_id": (None if state.last_verify_lock_track_id is None else int(state.last_verify_lock_track_id)),
        "verify_last_lock_source": str(state.last_verify_lock_source or "none"),
        "verify_measurement_fallback_source": str(verify_measurement_fallback_source),
        "verify_mismatch_recenter_used": False,
        "verify_mismatch_blocked_track": None,
        "verify_mismatch_blocked_uv": None,
        "pre_observation": pre_obs,
        "post_observation": post_obs,
    }

    state.last_place_verification_v2 = dict(result)
    state.last_place_verification = dict(result)
    placement["verify_result"] = dict(result)
    verified_color_name = str(result.get("measured_color", "unknown")).strip().lower()
    try:
        verified_color_conf = float(result.get("measured_color_conf", 0.0))
    except Exception:
        verified_color_conf = 0.0
    try:
        verified_color_hits = int(result.get("measured_color_hits", 0))
    except Exception:
        verified_color_hits = 0
    color_geometry_ok = _color_geometry_ok_for_commit(
        result,
        fallback_xy_margin_m=float(score_xy_margin_m),
        fallback_z_margin_m=float(score_z_margin_m),
    )
    if (
        verified_color_name in {"orange", "blue"}
        and verified_color_hits >= max(1, int(PLACE_VERIFY_V2_COLOR_MIN_HITS))
        and verified_color_conf >= float(PLACE_VERIFY_V2_COLOR_COMMIT_CONF)
        and bool(color_geometry_ok)
    ):
        prev_color_name = str(placement.get("cube_color", "unknown")).strip().lower()
        placement["cube_color"] = str(verified_color_name)
        placement["cube_color_verified"] = str(verified_color_name)
        placement["cube_color_verified_conf"] = float(verified_color_conf)
        if prev_color_name != verified_color_name:
            print(
                f"[PlaceVerifyV2Color] corrected cube_color {prev_color_name} -> {verified_color_name} "
                f"(conf={verified_color_conf:.3f}, hits={verified_color_hits})"
            )
    else:
        if (
            verified_color_name in {"orange", "blue"}
            and verified_color_hits >= max(1, int(PLACE_VERIFY_V2_COLOR_MIN_HITS))
            and verified_color_conf >= float(PLACE_VERIFY_V2_COLOR_COMMIT_CONF)
            and not bool(color_geometry_ok)
        ):
            print(
                f"[PlaceVerifyV2Color] skipped_color_commit color={verified_color_name} "
                f"conf={verified_color_conf:.3f} hits={verified_color_hits} "
                f"status={result.get('status')} geometry_ok=False"
            )
        placement["cube_color_verified"] = "unknown"
        placement["cube_color_verified_conf"] = float(verified_color_conf)
    if count_in_stats and not bool(placement.get("verify_counted", False)):
        if bool(result.get("confirmed", False)):
            state.place_verify_confirmed_count += 1
        elif str(result.get("status", "")).startswith("placed_uncertain") or str(result.get("status", "")).startswith(
            "placed_mismatch"
        ):
            state.place_verify_uncertain_count += 1
        placement["verify_counted"] = True

    exp = result.get("expected_xyz")
    exp_eval = result.get("expected_xyz_eval")
    meas = result.get("measured_xyz")
    look_xyz = result.get("verify_look_pose")
    exp_text = "n/a" if exp is None else f"({exp[0]:.3f},{exp[1]:.3f},{exp[2]:.3f})"
    exp_eval_text = "n/a" if exp_eval is None else f"({exp_eval[0]:.3f},{exp_eval[1]:.3f},{exp_eval[2]:.3f})"
    meas_text = "n/a" if meas is None else f"({meas[0]:.3f},{meas[1]:.3f},{meas[2]:.3f})"
    look_text = "n/a" if look_xyz is None else f"({look_xyz[0]:.3f},{look_xyz[1]:.3f},{look_xyz[2]:.3f})"
    print(
        f"[PlaceVerifyV2] obj_id={result.get('object_id')} slot={result.get('slot_index')} "
        f"section={result.get('expected_section')} color={result.get('expected_color')} "
        f"color_src={result.get('expected_color_source')} "
        f"expected={exp_text} expected_eval={exp_eval_text} measured={meas_text} "
        f"measured_color={result.get('measured_color')} "
        f"(conf={float(result.get('measured_color_conf', 0.0)):.2f},"
        f"hits={int(result.get('measured_color_hits', 0))}/{int(result.get('measured_color_samples', 0))}) "
        f"x_ref_offset={float(result.get('expected_x_offset_m', 0.0)):.3f} "
        f"y_ref_offset={float(result.get('expected_y_offset_m', 0.0)):.3f} "
        f"z_ref_offset={float(result.get('expected_z_offset_m', 0.0)):.3f} "
        f"look={look_text} "
        f"stack_min_z={result.get('verify_stack_min_z_m')} prefer_top={result.get('verify_stack_prefer_top')} "
        f"look_status={result.get('verify_look_status')} "
        f"target_mode={result.get('verify_target_mode')} "
        f"err_xy={float(result.get('xy_error_m', float('inf'))):.3f} "
        f"err_z={float(result.get('z_error_m', float('inf'))):.3f} "
        f"margins=(xy:{float(result.get('effective_xy_margin_m', float('nan'))):.3f},"
        f"z:{float(result.get('effective_z_margin_m', float('nan'))):.3f}) "
        f"overlap={float(result.get('overlap_ratio', 0.0)):.2f} "
        f"hits={int(result.get('hits', 0))}/{int(result.get('samples', 0))} "
        f"delta={float(result.get('delta_score', 0.0)):.3f} "
        f"blocked_tracks={result.get('verify_blocked_track_ids')} "
        f"reject_count={int(result.get('verify_reject_count', 0))}/{int(result.get('verify_max_rejects', 0))} "
        f"retry_attempts={len(result.get('verify_expected_recenter_attempts', []) or [])} "
        f"top_checks={len(result.get('verify_top_candidate_attempts', []) or [])} "
        f"generic_deferred={bool(result.get('verify_generic_handoff_deferred', False))} "
        f"selected_track_id={result.get('verify_selected_track_id')} "
        f"lock_track={result.get('verify_last_lock_track_id')} "
        f"exit_reason={result.get('verify_exit_reason')} "
        f"measurement_src={result.get('verify_measurement_fallback_source')} "
        f"status={result.get('status')} active_center={bool(result.get('active_center_used', False))}"
    )
    return result



__all__ = [
    "compute_verify_stack_min_z",
    "compute_verify_z_margin",
    "collect_slot_observations",
    "associate_newest_placement",
    "build_verify_expected_for_score",
    "score_place_geometry",
    "verify_last_place_reliability",
]
