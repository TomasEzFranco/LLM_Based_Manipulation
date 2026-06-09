#!/usr/bin/env python3
from __future__ import annotations

import cv2
import math
import numpy as np
import time

_CORE_BIND_READY = False


def _bind_core_globals() -> None:
    global _CORE_BIND_READY
    import runtime_core as core
    protected = {
        '_bind_core_globals', '_CORE_BIND_READY',
        'run_track_handoff_session', 'run_return_verify_stage',
        'run_return_handoff_stage', 'run_return_verify_and_handoff_session',
        'execute_pick_misplaced_cube_action', 'execute_return_placed_cube_correction',
    }
    for name, value in core.__dict__.items():
        if name.startswith('__') or name in protected:
            continue
        globals()[name] = value
    _CORE_BIND_READY = True


def _misplaced_pick_height_band(
    z_m: float,
    *,
    step_m: float | None = None,
    tol_m: float | None = None,
    max_level: int | None = None,
) -> dict:
    _bind_core_globals()
    step = float(MISPLACED_PICK_HEIGHT_STEP_M if step_m is None else step_m)
    tol = float(MISPLACED_PICK_HEIGHT_TOL_M if tol_m is None else tol_m)
    max_level_i = int(MAX_STACK_LEVELS_PER_SECTION if max_level is None else max_level)
    max_level_i = max(1, int(max_level_i))
    step = max(1e-6, float(step))
    tol = max(0.0, float(tol))
    try:
        z_val = float(z_m)
    except Exception:
        z_val = float("nan")
    if not np.isfinite(z_val):
        return {
            "valid": False,
            "reason": "height_out_of_band",
            "z_m": float("nan"),
            "level": None,
            "nearest_z_m": None,
            "error_m": None,
            "step_m": float(step),
            "tol_m": float(tol),
            "max_level": int(max_level_i),
        }
    level_i = int(round(float(z_val) / float(step)))
    nearest_z = float(level_i) * float(step)
    err = float(abs(float(z_val) - float(nearest_z)))
    in_range = bool(1 <= int(level_i) <= int(max_level_i))
    in_band = bool(float(err) <= float(tol))
    valid = bool(in_range and in_band)
    if not bool(in_range):
        reason = "height_level_out_of_range"
    elif not bool(in_band):
        reason = "height_out_of_band"
    else:
        reason = "ok"
    return {
        "valid": bool(valid),
        "reason": str(reason),
        "z_m": float(z_val),
        "level": int(level_i),
        "nearest_z_m": float(nearest_z),
        "error_m": float(err),
        "step_m": float(step),
        "tol_m": float(tol),
        "max_level": int(max_level_i),
    }


def _handoff_candidate_on_target_pixel_half(
    u: int,
    cx: int,
    preferred_section_norm: str,
) -> bool:
    """True when detection UV lies on the target stack column (image half)."""
    _bind_core_globals()
    side_norm = str(preferred_section_norm).strip().lower()
    if side_norm not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        return True
    margin_px = max(0, int(PLACE_VERIFY_V2_SECTION_PIXEL_MARGIN_PX))
    if side_norm == str(SECTION_LEFT_NAME).strip().lower():
        return int(u) <= int(int(cx) + margin_px)
    return int(u) >= int(int(cx) - margin_px)


def rank_misplaced_handoff_track_candidate(
    candidates: list[dict],
    projected_rows: list[dict],
    *,
    cx: int,
    cy: int,
    min_conf: float,
    preferred_section_norm: str,
    enforce_preferred_section_hard_filter: bool = False,
    section_centers_xy: dict[str, tuple[float, float]] | None = None,
    blocked_track_ids: set[int] | None = None,
) -> dict | None:
    """Rank handoff picks for pick_placed: pixel column + top-v, or legacy xyz filter when enabled."""
    _bind_core_globals()
    _ = int(cy)
    preferred_norm = str(preferred_section_norm).strip().lower()
    if preferred_norm not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        preferred_norm = ""
    centers_xy: dict[str, tuple[float, float]] = dict(section_centers_xy or {})
    blocked = set(int(t) for t in (blocked_track_ids or set()))
    pixel_only = bool(preferred_norm) and not bool(enforce_preferred_section_hard_filter)
    rows: list[dict] = []
    for c in candidates:
        conf = float(c.get("conf", 0.0))
        if conf < float(min_conf):
            continue
        raw_tid = c.get("track_id", None)
        if raw_tid is None:
            continue
        try:
            tid = int(raw_tid)
        except (TypeError, ValueError):
            continue
        if int(tid) in blocked:
            continue
        u = int(c.get("u", 0))
        v = int(c.get("v", 0))
        if pixel_only and not _handoff_candidate_on_target_pixel_half(int(u), int(cx), preferred_norm):
            continue
        xyz_proj = None
        if projected_rows:
            proj = _match_projected_row_by_uv(projected_rows, u=u, v=v)
            if proj is not None:
                xyz_proj = _finite_xyz_or_none(proj.get("xyz", None))
        if bool(enforce_preferred_section_hard_filter) and preferred_norm and centers_xy:
            if isinstance(xyz_proj, (list, tuple)) and len(xyz_proj) >= 2:
                inferred_side_pref, assign_info = _infer_section_for_place_xy(
                    float(xyz_proj[0]),
                    float(xyz_proj[1]),
                    centers_xy,
                    band_min=None,
                    band_max=None,
                    max_center_dist_m=float(MISPLACED_PICK_SECTION_MAX_DIST_M),
                )
                inferred_side_norm = (
                    "" if inferred_side_pref is None else str(inferred_side_pref).strip().lower()
                )
                if inferred_side_norm in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
                    if inferred_side_norm != preferred_norm:
                        continue
        side_pref_rank = 1
        side_pref_source = "none"
        if preferred_norm:
            if pixel_only:
                side_pref_rank = 0
                side_pref_source = "pixel"
            elif isinstance(xyz_proj, (list, tuple)) and len(xyz_proj) >= 2:
                inferred_side_pref, _assign_info_pref = _infer_section_for_place_xy(
                    float(xyz_proj[0]),
                    float(xyz_proj[1]),
                    centers_xy,
                    band_min=None,
                    band_max=None,
                    max_center_dist_m=float(MISPLACED_PICK_SECTION_MAX_DIST_M),
                )
                inferred_side_norm = (
                    "" if inferred_side_pref is None else str(inferred_side_pref).strip().lower()
                )
                if inferred_side_norm == preferred_norm:
                    side_pref_rank = 0
                    side_pref_source = "xyz"
                elif _handoff_candidate_on_target_pixel_half(int(u), int(cx), preferred_norm):
                    side_pref_rank = 0
                    side_pref_source = "pixel"
            elif _handoff_candidate_on_target_pixel_half(int(u), int(cx), preferred_norm):
                side_pref_rank = 0
                side_pref_source = "pixel"
        z_m = None
        if isinstance(xyz_proj, (list, tuple)) and len(xyz_proj) >= 3:
            try:
                z_candidate = float(xyz_proj[2])
                if np.isfinite(z_candidate):
                    z_m = float(z_candidate)
            except (TypeError, ValueError):
                z_m = None
        prefer_max_z = bool(PICK_PLACED_HANDOFF_PREFER_MAX_Z) and (z_m is not None)
        row = dict(c)
        row["u"] = int(u)
        row["v"] = int(v)
        row["conf"] = float(conf)
        row["track_id"] = int(tid)
        row["d2_px"] = float((int(u) - int(cx)) ** 2 + (int(v) - int(cy)) ** 2)
        row["side_pref_rank"] = int(side_pref_rank)
        row["side_pref_source"] = str(side_pref_source)
        row["z_m"] = z_m
        row["handoff_rank"] = "max_z" if bool(prefer_max_z) else "pixel_top"
        rows.append(row)
    if not rows:
        return None
    use_max_z = bool(PICK_PLACED_HANDOFF_PREFER_MAX_Z) and any(
        r.get("z_m", None) is not None for r in rows
    )

    def _handoff_sort_key(r: dict) -> tuple:
        z_key = float(r.get("z_m")) if r.get("z_m", None) is not None else float("-inf")
        if bool(use_max_z):
            return (
                int(r.get("side_pref_rank", 1)),
                -float(z_key),
                int(r.get("v", 0)),
                float(r.get("d2_px", float("inf"))),
                -float(r.get("conf", 0.0)),
            )
        return (
            int(r.get("side_pref_rank", 1)),
            int(r.get("v", 0)),
            float(r.get("d2_px", float("inf"))),
            -float(r.get("conf", 0.0)),
        )

    rows.sort(key=_handoff_sort_key)
    return rows[0]


def _try_sole_visible_track_handoff_retry(
    *,
    state,
    arm,
    det,
    per,
    failed_track_ids: list[int],
    sole_track_retries_used: int,
    max_sole_retries: int,
    target_section_norm: str,
    section_centers_xy: dict[str, tuple[float, float]],
    enforce_hard_filter: bool,
    min_conf: float,
) -> tuple[bool, int]:
    """When every visible target-side track is blocked, clear block list once and retry."""
    _bind_core_globals()
    if int(sole_track_retries_used) >= int(max(0, max_sole_retries)):
        return False, int(sole_track_retries_used)
    if not failed_track_ids:
        return False, int(sole_track_retries_used)
    obs_snap = observe_scene_frame(
        det=det,
        arm=arm,
        per=per,
        draw=False,
        projected_min_conf=float(min_conf),
        state=state,
        update_tracks=True,
    )
    if obs_snap is None:
        return False, int(sole_track_retries_used)
    center_uv = getattr(obs_snap, "image_center_uv", None)
    if isinstance(center_uv, (list, tuple, np.ndarray)) and len(center_uv) >= 2:
        cx = int(center_uv[0])
        cy = int(center_uv[1])
    else:
        cx = 320
        cy = 240
    blocked_set = set(int(tid) for tid in list(failed_track_ids))
    blocked_pick = rank_misplaced_handoff_track_candidate(
        list(obs_snap.candidates or []),
        list(obs_snap.projected_rows or []),
        cx=int(cx),
        cy=int(cy),
        min_conf=float(min_conf),
        preferred_section_norm=str(target_section_norm),
        enforce_preferred_section_hard_filter=bool(enforce_hard_filter),
        section_centers_xy=dict(section_centers_xy),
        blocked_track_ids=blocked_set,
    )
    if blocked_pick is not None:
        return False, int(sole_track_retries_used)
    open_pick = rank_misplaced_handoff_track_candidate(
        list(obs_snap.candidates or []),
        list(obs_snap.projected_rows or []),
        cx=int(cx),
        cy=int(cy),
        min_conf=float(min_conf),
        preferred_section_norm=str(target_section_norm),
        enforce_preferred_section_hard_filter=bool(enforce_hard_filter),
        section_centers_xy=dict(section_centers_xy),
        blocked_track_ids=set(),
    )
    if open_pick is None:
        return False, int(sole_track_retries_used)
    sole_tid = int(open_pick.get("track_id"))
    if int(sole_tid) not in blocked_set:
        return False, int(sole_track_retries_used)
    failed_track_ids.clear()
    print(
        f"[PickMisplacedRetry] sole_visible_track tid={int(sole_tid)} "
        f"retry={int(sole_track_retries_used) + 1}/{int(max_sole_retries)} "
        f"side={str(target_section_norm)}"
    )
    return True, int(sole_track_retries_used) + 1


def _misplaced_xyz_log_text(xyz: list[float] | tuple[float, ...] | np.ndarray | None) -> str:
    if xyz is None:
        return "none"
    try:
        arr = np.array(xyz, dtype=float).reshape(-1)
    except Exception:
        return "none"
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return "none"
    return f"({float(arr[0]):.3f},{float(arr[1]):.3f},{float(arr[2]):.3f})"


def _misplaced_pick_top_height_eval(
    selected_z: float,
    *,
    expected_level: int,
    step_m: float | None = None,
    top_tol_m: float | None = None,
    max_level: int | None = None,
    require_top_level_match: bool | None = None,
) -> dict:
    """Validate measured Z for pick_placed: must match authoritative top stack_level."""
    _bind_core_globals()
    step = float(MISPLACED_PICK_HEIGHT_STEP_M if step_m is None else step_m)
    top_tol = float(MISPLACED_PICK_TOP_HEIGHT_TOL_M if top_tol_m is None else top_tol_m)
    max_level_i = int(MAX_STACK_LEVELS_PER_SECTION if max_level is None else max_level)
    max_level_i = max(1, int(max_level_i))
    step = max(1e-6, float(step))
    top_tol = max(0.0, float(top_tol))
    require_top = bool(MISPLACED_PICK_REQUIRE_TOP_LEVEL_MATCH if require_top_level_match is None else require_top_level_match)
    try:
        expected_level_i = int(max(0, min(int(max_level_i), int(expected_level))))
    except Exception:
        expected_level_i = 0
    try:
        z_val = float(selected_z)
    except Exception:
        z_val = float("nan")
    level_probe = _misplaced_pick_height_band(
        z_val,
        step_m=float(step),
        tol_m=float(top_tol),
        max_level=int(max_level_i),
    )
    selected_level_i = level_probe.get("level", None)
    expected_z = (
        None
        if int(expected_level_i) < 1
        else float(int(expected_level_i)) * float(step)
    )
    err_top = (
        float("nan")
        if (expected_z is None) or (not np.isfinite(z_val))
        else float(abs(float(z_val) - float(expected_z)))
    )
    out = {
        "valid": False,
        "reason": "height_out_of_band",
        "selected_z_m": z_val if np.isfinite(z_val) else None,
        "selected_level": selected_level_i,
        "selected_nearest_z_m": expected_z,
        "selected_error_m": err_top if np.isfinite(err_top) else None,
        "height_step_m": float(step),
        "height_tol_m": float(top_tol),
        "max_level": int(max_level_i),
        "expected_state_level": int(expected_level_i),
        "expected_state_z_m": expected_z,
        "require_top_level_match": bool(require_top),
        "state_level_delta": None,
        "state_reconcile_action": "none",
    }
    if int(expected_level_i) < 1:
        out["reason"] = "height_no_stack_to_pick"
        return out
    if not np.isfinite(z_val):
        out["reason"] = "height_out_of_band"
        return out
    if bool(require_top):
        if selected_level_i is None:
            out["reason"] = "height_out_of_band"
            return out
        allow_higher = bool(MISPLACED_PICK_ALLOW_HIGHER_THAN_EXPECTED)
        delta_i = int(selected_level_i) - int(expected_level_i)
        if int(delta_i) < 0:
            out["reason"] = "height_not_top_level"
            out["state_level_delta"] = int(delta_i)
            out["state_reconcile_action"] = "lower_state_to_measured"
            return out
        if int(delta_i) > 0:
            if not bool(allow_higher):
                out["reason"] = "height_not_top_level"
                out["state_level_delta"] = int(delta_i)
                out["state_reconcile_action"] = "raise_state_to_measured"
                return out
            selected_band = _misplaced_pick_height_band(
                z_val,
                step_m=float(step),
                tol_m=float(top_tol),
                max_level=int(max_level_i),
            )
            if not bool(selected_band.get("valid", False)):
                out["reason"] = str(selected_band.get("reason", "height_out_of_band"))
                return out
            out["valid"] = True
            out["reason"] = "ok_higher_than_expected"
            out["state_level_delta"] = int(delta_i)
            out["state_reconcile_action"] = "raise_state_to_measured"
            return out
        if (not np.isfinite(err_top)) or float(err_top) > float(top_tol):
            out["reason"] = "height_out_of_band"
            return out
        out["valid"] = True
        out["reason"] = "ok"
        out["state_level_delta"] = 0
        out["state_reconcile_action"] = "none"
        return out
    selected_band = level_probe
    if not bool(selected_band.get("valid", False)):
        out["reason"] = str(selected_band.get("reason", "height_out_of_band"))
        return out
    out["valid"] = True
    out["reason"] = "ok"
    if int(selected_level_i) != int(expected_level_i):
        out["state_level_delta"] = int(selected_level_i) - int(expected_level_i)
        if int(selected_level_i) > int(expected_level_i):
            out["state_reconcile_action"] = "raise_state_to_measured"
        elif int(selected_level_i) < int(expected_level_i):
            out["state_reconcile_action"] = "lower_state_to_measured"
    else:
        out["state_level_delta"] = 0
        out["state_reconcile_action"] = "none"
    return out


def _build_misplaced_pick_height_gate(
    *,
    obs,
    selected_xyz: list[float] | tuple[float, float, float] | np.ndarray,
    target_section_norm: str,
    infer_section_for_xyz,
    correction_pick_min_y_m: float,
    expected_level: int | None = None,
    step_m: float | None = None,
    tol_m: float | None = None,
    max_level: int | None = None,
) -> dict:
    _bind_core_globals()
    _ = (obs, target_section_norm, infer_section_for_xyz, correction_pick_min_y_m)
    max_level_i = int(MAX_STACK_LEVELS_PER_SECTION if max_level is None else max_level)
    max_level_i = max(1, int(max_level_i))
    expected_level_i = None
    if expected_level is not None:
        try:
            expected_level_i = int(max(0, min(int(max_level_i), int(expected_level))))
        except Exception:
            expected_level_i = None
    selected_arr = np.array(selected_xyz, dtype=float).reshape(-1)
    selected_z = float(selected_arr[2]) if selected_arr.size >= 3 else float("nan")
    top_eval = _misplaced_pick_top_height_eval(
        selected_z,
        expected_level=0 if expected_level_i is None else int(expected_level_i),
        step_m=step_m,
        top_tol_m=tol_m,
        max_level=int(max_level_i),
    )
    gate = {
        "valid": False,
        "reason": str(top_eval.get("reason", "height_out_of_band")),
        "selected_z_m": top_eval.get("selected_z_m", None),
        "selected_level": top_eval.get("selected_level", None),
        "selected_nearest_z_m": top_eval.get("selected_nearest_z_m", None),
        "selected_error_m": top_eval.get("selected_error_m", None),
        "height_step_m": top_eval.get("height_step_m", None),
        "height_tol_m": top_eval.get("height_tol_m", None),
        "max_level": int(max_level_i),
        "expected_state_level": top_eval.get("expected_state_level", expected_level_i),
        "expected_state_z_m": top_eval.get("expected_state_z_m", None),
        "state_level_delta": top_eval.get("state_level_delta", None),
        "state_reconcile_action": top_eval.get("state_reconcile_action", "none"),
        "target_side_top_level": top_eval.get("selected_level", None),
        "target_side_top_z_m": top_eval.get("selected_z_m", None),
        "reference_source": "centered_measurement",
        "reference_candidate_count": 0,
        "target_side_xyz_count": 0,
        "require_top_level_match": top_eval.get("require_top_level_match", None),
    }
    if bool(top_eval.get("valid", False)):
        gate["valid"] = True
        gate["reason"] = "ok"
    return gate


def _reconcile_authoritative_stack_level_to_measured(
    state,
    section_name: str,
    measured_level: int | None,
) -> dict:
    _bind_core_globals()
    section_norm = str(section_name).strip().lower()
    if section_norm not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        return {"changed": False, "reason": "invalid_section", "section": section_norm}
    if measured_level is None:
        return {"changed": False, "reason": "missing_measured_level", "section": section_norm}
    measured_level_i = int(max(0, min(int(MAX_STACK_LEVELS_PER_SECTION), int(measured_level))))
    row = get_startup_hydrated_section_row(state, section_norm)
    old_level = int(max(0, row.get("stack_level", 0) or 0))
    seq = list(row.get("color_sequence_bottom_to_top", []))
    if len(seq) < int(old_level):
        seq.extend(["unknown"] * int(old_level - len(seq)))
    seq = list(seq[: int(max(old_level, measured_level_i))])
    if int(measured_level_i) > len(seq):
        seq.extend(["unknown"] * int(measured_level_i - len(seq)))
    seq = list(seq[: int(measured_level_i)])
    if int(measured_level_i) == int(old_level):
        return {
            "changed": False,
            "reason": "already_matching",
            "section": section_norm,
            "old_level": int(old_level),
            "measured_level": int(measured_level_i),
            "sequence": list(seq),
        }
    updated = _set_authoritative_section_sequence(state, section_norm, seq)
    return {
        "changed": True,
        "reason": "measured_height_reconcile",
        "section": section_norm,
        "old_level": int(old_level),
        "measured_level": int(measured_level_i),
        "new_level": int(updated.get("stack_level", 0) or 0),
        "sequence": list(updated.get("color_sequence_bottom_to_top", [])),
    }


def _misplaced_return_grid_slot(drop_index: int) -> dict:
    _bind_core_globals()
    try:
        raw_i = int(drop_index)
    except Exception:
        raw_i = 0
    raw_i = int(max(0, raw_i))
    cols_i = max(1, int(MISPLACED_RETURN_GRID_COLS))
    max_slots_i = max(1, int(MISPLACED_RETURN_GRID_MAX_SLOTS))
    base_xyz = np.array(
        [float(MISPLACED_RETURN_DROP_X_M), float(MISPLACED_RETURN_DROP_Y_M), float(MISPLACED_RETURN_DROP_Z_M)],
        dtype=float,
    ).reshape(-1)
    if int(raw_i) >= int(max_slots_i):
        return {
            "ok": False,
            "reason": "misplaced_return_slots_exhausted",
            "raw": int(raw_i),
            "row": None,
            "col": None,
            "max_slots": int(max_slots_i),
            "drop_xyz": None,
            "base_xyz": [float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2])],
            "grid_dx_m": float(MISPLACED_RETURN_GRID_DX_M),
            "grid_dy_m": float(MISPLACED_RETURN_GRID_DY_M),
            "reach_m": None,
        }
    col_i = int(raw_i % int(cols_i))
    row_i = int(raw_i // int(cols_i))
    drop_xyz = base_xyz.copy()
    drop_xyz[0] = float(drop_xyz[0]) + float(col_i) * float(MISPLACED_RETURN_GRID_DX_M)
    drop_xyz[1] = float(drop_xyz[1]) + float(row_i) * float(MISPLACED_RETURN_GRID_DY_M)
    drop_xyz[2] = max(float(drop_xyz[2]), float(TABLE_Z_SAT_M), float(PLACE_RELEASE_Z_M))
    reach = float(np.linalg.norm(drop_xyz[:3]))
    reason = "ok"
    ok = True
    if reach > float(MAX_REACH_M):
        ok = False
        reason = "misplaced_return_slot_too_far_from_base"
    elif reach < float(MIN_PLACE_REACH_M):
        ok = False
        reason = "misplaced_return_slot_too_close_to_base"
    return {
        "ok": bool(ok),
        "reason": str(reason),
        "raw": int(raw_i),
        "row": int(row_i),
        "col": int(col_i),
        "max_slots": int(max_slots_i),
        "drop_xyz": [float(drop_xyz[0]), float(drop_xyz[1]), float(drop_xyz[2])],
        "base_xyz": [float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2])],
        "grid_dx_m": float(MISPLACED_RETURN_GRID_DX_M),
        "grid_dy_m": float(MISPLACED_RETURN_GRID_DY_M),
        "reach_m": float(reach),
    }


def _classify_misplaced_return_place_failure(place_reason: object, arm: object | None = None) -> dict:
    reason = str(place_reason or "").strip().lower()
    diag_src = getattr(arm, "last_motion_diag", {}) if arm is not None else {}
    diag = dict(diag_src) if isinstance(diag_src, dict) else {}
    label = str(diag.get("label", "") or "").strip().lower()
    motion_reason = str(
        diag.get("last_motion_reason", getattr(arm, "last_motion_reason", ""))
        if arm is not None
        else diag.get("last_motion_reason", "")
    ).strip().lower()
    post_release_reason = reason in {
        "place_release_clearance_failed",
        "place_retreat_vertical_failed",
    }
    post_release_motion = bool(
        reason == "move_overcurrent_unrecoverable"
        and any(token in label for token in ("release_clearance", "retreat_vertical"))
    )
    released = bool(post_release_reason or post_release_motion)
    return {
        "reason": str(reason or "unknown"),
        "released": bool(released),
        "holding_after_failure": bool(not released),
        "post_release_phase": bool(released),
        "unrecoverable": bool(reason == "move_overcurrent_unrecoverable"),
        "motion_label": str(label),
        "motion_reason": str(motion_reason),
    }


def run_track_handoff_session(
    state: CycleState,
    arm: Arm,
    per: Perception | None,
    det: YOLODetector | None,
    *,
    reject_cap: int,
    hard_timeout_s: float,
    xy_margin_m: float,
    z_margin_m: float,
    blocked_track_id: int | None = None,
    blocked_xyz: list[float] | None = None,
    blocked_track_ids_extra: set[int] | list[int] | None = None,
    blocked_xyzs_extra: list[list[float]] | None = None,
    blocked_uv: list[int] | None = None,
    status_prefix: str = "pick_other",
    log_prefix: str = "[PickOther]",
    disable_stable_gate: bool = False,
    track_stable_frames_override: int | None = None,
    commit_conf_override: float | None = None,
    post_lock_refresh_s: float = 0.0,
    post_lock_refresh_min_frames: int = 0,
    on_locked_candidate=None,
    required_track_id: int | None = None,
    centered_frames_required: int | None = None,
    close_window_on_exit: bool = True,
    max_no_candidate_frames: int | None = None,
    terminal_callback_decision: bool = False,
    wrong_section_soft_skip: bool = True,
    preferred_section_name: str | None = None,
    preferred_section_centers_xy: dict[str, tuple[float, float]] | None = None,
    enforce_preferred_section_hard_filter: bool = False,
    use_projected_xyz_for_handoff: bool = True,
    center_ey_scale: float = 1.0,
    enforce_pick_workspace_only: bool = False,
    reject_below_base_y_m: float | None = None,
    refresh_required_track_timeout_on_visible: bool = False,
    refresh_required_track_max_s: float | None = None,
) -> dict:
    _bind_core_globals()
    reject_cap = max(1, int(reject_cap))
    hard_timeout_s = max(0.5, float(hard_timeout_s))
    no_candidate_timeout_s = float(
        max(0.4, min(float(hard_timeout_s), float(TRACK_HANDOFF_NO_CANDIDATE_TIMEOUT_S)))
    )
    detect_conf_local = float(DETECT_CONF)
    commit_conf_local = float(COMMIT_CONF)
    if commit_conf_override is not None:
        try:
            commit_conf_local = float(commit_conf_override)
        except (TypeError, ValueError):
            commit_conf_local = float(COMMIT_CONF)
    commit_conf_local = float(max(0.0, min(1.0, commit_conf_local)))
    post_lock_refresh_s = float(max(0.0, float(post_lock_refresh_s)))
    post_lock_refresh_min_frames = int(max(0, int(post_lock_refresh_min_frames)))
    max_no_candidate_frames_i = 0 if max_no_candidate_frames is None else max(0, int(max_no_candidate_frames))
    terminal_callback_decision = bool(terminal_callback_decision)
    wrong_section_soft_skip = bool(wrong_section_soft_skip)
    preferred_section_norm = ""
    if preferred_section_name is not None:
        preferred_section_norm = str(preferred_section_name).strip().lower()
    if preferred_section_norm not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        preferred_section_norm = ""
    section_centers_pref: dict[str, tuple[float, float]] = {}
    if isinstance(preferred_section_centers_xy, dict) and preferred_section_centers_xy:
        for name_raw, center_raw in preferred_section_centers_xy.items():
            side_norm = str(name_raw).strip().lower()
            if side_norm not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
                continue
            try:
                cx_v = float(center_raw[0])
                cy_v = float(center_raw[1])
            except Exception:
                continue
            if np.isfinite(cx_v) and np.isfinite(cy_v):
                section_centers_pref[str(side_norm)] = (float(cx_v), float(cy_v))
    elif preferred_section_norm:
        section_centers_pref = dict(_verify_section_xy_centers())
    enforce_preferred_section_hard_filter = bool(enforce_preferred_section_hard_filter)
    use_projected_xyz_for_handoff = bool(use_projected_xyz_for_handoff)
    handoff_pixel_only_side = bool(preferred_section_norm) and not bool(
        enforce_preferred_section_hard_filter
    )
    center_ey_scale = float(np.clip(float(center_ey_scale), 0.20, 1.20))
    enforce_pick_workspace_only = bool(enforce_pick_workspace_only)
    if reject_below_base_y_m is None:
        reject_below_base_y = None
    else:
        try:
            reject_below_base_y = float(reject_below_base_y_m)
        except Exception:
            reject_below_base_y = None
        if reject_below_base_y is not None and not np.isfinite(float(reject_below_base_y)):
            reject_below_base_y = None
    if centered_frames_required is None:
        centered_frames_req = max(1, int(CENTERED_FRAMES_REQUIRED))
    else:
        centered_frames_req = max(1, int(centered_frames_required))
    if det is None or per is None:
        return {
            "status": "observe_retry",
            "centered_pos": None,
            "selected_track_id": None,
            "selected_xyz": None,
            "reject_count": 0,
            "reject_cap": int(reject_cap),
            "blocked_track_ids": [],
            "exit_reason": "dependencies_missing",
            "accept_payload": None,
            "last_decision": "none",
            "last_reason": "",
            "last_candidate_track_id": None,
            "no_candidate_frames": 0,
        }
    required_track_id_i: int | None = None
    if required_track_id is not None:
        try:
            required_track_id_i = int(required_track_id)
        except (TypeError, ValueError):
            required_track_id_i = None
    refresh_required_track_timeout_on_visible = bool(refresh_required_track_timeout_on_visible)
    if refresh_required_track_max_s is None:
        refresh_required_track_max_s = float(hard_timeout_s)
    try:
        refresh_required_track_max_s = float(max(float(hard_timeout_s), float(refresh_required_track_max_s)))
    except Exception:
        refresh_required_track_max_s = float(hard_timeout_s)

    blocked_track_ids: set[int] = set()
    blocked_xyzs: list[np.ndarray] = []
    if blocked_track_id is not None:
        try:
            blocked_track_ids.add(int(blocked_track_id))
        except (TypeError, ValueError):
            pass
    if blocked_track_ids_extra:
        for tid_raw in blocked_track_ids_extra:
            try:
                blocked_track_ids.add(int(tid_raw))
            except Exception:
                continue
    if isinstance(blocked_xyz, (list, tuple)) and len(blocked_xyz) >= 3:
        try:
            bxyz = np.array([float(blocked_xyz[0]), float(blocked_xyz[1]), float(blocked_xyz[2])], dtype=float).reshape(-1)
        except (TypeError, ValueError):
            bxyz = np.array([np.nan, np.nan, np.nan], dtype=float)
        if bxyz.size >= 3 and np.all(np.isfinite(bxyz[:3])):
            blocked_xyzs.append(np.array([float(bxyz[0]), float(bxyz[1]), float(bxyz[2])], dtype=float))
    if blocked_xyzs_extra:
        for xyz_row in blocked_xyzs_extra:
            try:
                extra = np.array([float(xyz_row[0]), float(xyz_row[1]), float(xyz_row[2])], dtype=float).reshape(-1)
            except Exception:
                continue
            if extra.size >= 3 and np.all(np.isfinite(extra[:3])):
                blocked_xyzs.append(np.array([float(extra[0]), float(extra[1]), float(extra[2])], dtype=float))
    def _candidate_track_id(row: dict) -> int | None:
        raw_tid = row.get("track_id", None)
        if raw_tid is None:
            return None
        try:
            return int(raw_tid)
        except (TypeError, ValueError):
            return None

    def _is_uv_blocked(u: int, v: int) -> bool:
        _ = (u, v)
        # Intentionally disabled: handoff rejection is track-id + XYZ based only.
        return False

    def _append_block_uv(u: int, v: int, radius_px: float | None = None) -> None:
        _ = (u, v, radius_px)
        # Intentionally disabled: no pixel-zone blacklist persistence.
        return

    def _append_block_xyz(xyz_row: list[float] | tuple[float, float, float] | np.ndarray | None) -> None:
        if xyz_row is None:
            return
        try:
            arr = np.array([float(xyz_row[0]), float(xyz_row[1]), float(xyz_row[2])], dtype=float).reshape(-1)
        except Exception:
            return
        if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
            return
        blocked_xyzs.append(np.array([float(arr[0]), float(arr[1]), float(arr[2])], dtype=float))

    def _blocked_xyzs_for_result() -> list[list[float]]:
        rows: list[list[float]] = []
        for xyz_row in blocked_xyzs:
            try:
                arr = np.array(xyz_row, dtype=float).reshape(-1)
            except Exception:
                continue
            if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                rows.append([float(arr[0]), float(arr[1]), float(arr[2])])
        return rows

    def _is_track_xyz_blocked(track_id: int) -> bool:
        if not blocked_xyzs:
            return False
        row = state.track_memory.get(int(track_id), None)
        if row is None:
            return False
        xyz = np.array(row.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
        if xyz.size < 3 or not np.all(np.isfinite(xyz[:3])):
            return False
        for bxyz in blocked_xyzs:
            if bxyz.size < 3 or not np.all(np.isfinite(bxyz[:3])):
                continue
            d_xy = float(math.hypot(float(xyz[0]) - float(bxyz[0]), float(xyz[1]) - float(bxyz[1])))
            d_z = float(abs(float(xyz[2]) - float(bxyz[2])))
            if d_xy <= float(xy_margin_m) and d_z <= float(z_margin_m):
                return True
        return False

    def _is_xyz_blocked_direct(xyz_row: list[float] | tuple[float, float, float] | np.ndarray | None) -> bool:
        if not blocked_xyzs:
            return False
        if xyz_row is None:
            return False
        try:
            arr = np.array([float(xyz_row[0]), float(xyz_row[1]), float(xyz_row[2])], dtype=float).reshape(-1)
        except Exception:
            return False
        if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
            return False
        for bxyz in blocked_xyzs:
            if bxyz.size < 3 or not np.all(np.isfinite(bxyz[:3])):
                continue
            d_xy = float(math.hypot(float(arr[0]) - float(bxyz[0]), float(arr[1]) - float(bxyz[1])))
            d_z = float(abs(float(arr[2]) - float(bxyz[2])))
            if d_xy <= float(xy_margin_m) and d_z <= float(z_margin_m):
                return True
        return False

    def _candidate_xyz_from_projected_rows(cand_row: dict, projected_rows: list[dict]) -> list[float] | None:
        if not bool(use_projected_xyz_for_handoff):
            return None
        if not isinstance(cand_row, dict):
            return None
        try:
            u = int(cand_row.get("u", 0))
            v = int(cand_row.get("v", 0))
        except Exception:
            return None
        proj = _match_projected_row_by_uv(projected_rows, u=u, v=v)
        if proj is None:
            return None
        return _finite_xyz_or_none(proj.get("xyz", None))

    def _candidate_outside_pick_workspace(
        cand_row: dict,
        projected_rows: list[dict],
    ) -> bool:
        if not bool(enforce_pick_workspace_only):
            return False
        xyz_proj = _candidate_xyz_from_projected_rows(cand_row, projected_rows)
        if xyz_proj is None:
            return False
        reason = pick_workspace_reject_reason(xyz_proj)
        if reason is None:
            return False
        tid = _candidate_track_id(cand_row)
        if tid is not None:
            if int(tid) not in blocked_track_ids:
                blocked_track_ids.add(int(tid))
                _append_block_xyz(xyz_proj)
                print(
                    f"{log_prefix} skip track_id={int(tid)} "
                    f"xyz={_misplaced_xyz_log_text(xyz_proj)} reason={reason} structural_skip=yes"
                )
        return True

    negative_y_skip_logged_track_ids: set[int] = set()

    def _candidate_below_min_base_y(
        cand_row: dict,
        projected_rows: list[dict],
    ) -> bool:
        if reject_below_base_y is None:
            return False
        xyz_proj = _candidate_xyz_from_projected_rows(cand_row, projected_rows)
        if xyz_proj is None:
            return False
        try:
            y_val = float(xyz_proj[1])
        except Exception:
            return False
        if not np.isfinite(y_val) or y_val >= float(reject_below_base_y):
            return False
        tid = _candidate_track_id(cand_row)
        if tid is not None and int(tid) not in negative_y_skip_logged_track_ids:
            negative_y_skip_logged_track_ids.add(int(tid))
            print(
                f"{log_prefix} skip track_id={int(tid)} "
                f"reason=negative_y_pick_area y={float(y_val):+.3f} "
                f"min_y={float(reject_below_base_y):+.3f}"
            )
        return True

    def _candidate_infers_opposite_section(
        xyz_proj: list[float] | None,
        *,
        centers_xy: dict[str, tuple[float, float]],
    ) -> tuple[bool, str]:
        if not bool(enforce_preferred_section_hard_filter) or not preferred_section_norm:
            return False, ""
        if not isinstance(xyz_proj, (list, tuple)) or len(xyz_proj) < 2:
            return False, ""
        if not centers_xy:
            return False, ""
        inferred_side_pref, assign_info = _infer_section_for_place_xy(
            float(xyz_proj[0]),
            float(xyz_proj[1]),
            centers_xy,
            band_min=None,
            band_max=None,
            max_center_dist_m=float(MISPLACED_PICK_SECTION_MAX_DIST_M),
        )
        inferred_side_norm = (
            "" if inferred_side_pref is None else str(inferred_side_pref).strip().lower()
        )
        if inferred_side_norm not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
            return False, ""
        if inferred_side_norm == preferred_section_norm:
            return False, ""
        via = str(assign_info.get("reason", "ok") or "ok")
        return True, f"got={inferred_side_norm} via={via}"

    def _select_top_track_candidate(
        candidates: list[dict],
        projected_rows: list[dict],
        cx: int,
        cy: int,
        min_conf: float,
    ) -> dict | None:
        prefiltered: list[dict] = []
        for c in candidates:
            conf = float(c.get("conf", 0.0))
            if conf < float(min_conf):
                continue
            tid = _candidate_track_id(c)
            if tid is None or int(tid) in blocked_track_ids:
                continue
            if _is_track_xyz_blocked(int(tid)):
                continue
            u = int(c.get("u", 0))
            v = int(c.get("v", 0))
            if _is_uv_blocked(u, v):
                continue
            xyz_proj = _candidate_xyz_from_projected_rows(c, projected_rows)
            if _is_xyz_blocked_direct(xyz_proj):
                continue
            if _candidate_below_min_base_y(c, projected_rows):
                continue
            if _candidate_outside_pick_workspace(c, projected_rows):
                continue
            if bool(enforce_preferred_section_hard_filter):
                wrong_section, wrong_detail = _candidate_infers_opposite_section(
                    xyz_proj,
                    centers_xy=section_centers_pref,
                )
                if bool(wrong_section):
                    print(
                        f"[PickMisplacedHandoff] skip tid={int(tid)} uv=({int(u)},{int(v)}) "
                        f"xyz={_misplaced_xyz_log_text(xyz_proj)} expected={preferred_section_norm} "
                        f"{wrong_detail}"
                    )
                    continue
            prefiltered.append(dict(c))
        picked = rank_misplaced_handoff_track_candidate(
            prefiltered,
            (projected_rows if bool(use_projected_xyz_for_handoff) else []),
            cx=int(cx),
            cy=int(cy),
            min_conf=float(min_conf),
            preferred_section_norm=str(preferred_section_norm),
            enforce_preferred_section_hard_filter=bool(enforce_preferred_section_hard_filter),
            section_centers_xy=dict(section_centers_pref),
            blocked_track_ids=set(),
        )
        return picked

    def _find_track_candidate(
        candidates: list[dict],
        tid: int,
        min_conf: float,
        projected_rows: list[dict] | None = None,
    ) -> dict | None:
        rows: list[dict] = []
        proj_rows = list(projected_rows or [])
        for c in candidates:
            conf = float(c.get("conf", 0.0))
            if conf < float(min_conf):
                continue
            c_tid = _candidate_track_id(c)
            if c_tid is None or int(c_tid) != int(tid):
                continue
            if _is_track_xyz_blocked(int(c_tid)):
                continue
            u = int(c.get("u", 0))
            v = int(c.get("v", 0))
            if _is_uv_blocked(u, v):
                continue
            if _candidate_below_min_base_y(c, proj_rows):
                continue
            if _candidate_outside_pick_workspace(c, proj_rows):
                continue
            row = dict(c)
            row["u"] = int(u)
            row["v"] = int(v)
            row["conf"] = float(conf)
            row["track_id"] = int(c_tid)
            rows.append(row)
        if not rows:
            return None
        rows.sort(key=lambda r: (-float(r.get("conf", 0.0)), int(r.get("v", 0))))
        return rows[0]

    def _has_any_track_candidate(
        candidates: list[dict],
        projected_rows: list[dict],
        min_conf: float,
    ) -> bool:
        for c in list(candidates):
            try:
                conf = float(c.get("conf", 0.0))
            except Exception:
                conf = 0.0
            if conf < float(min_conf):
                continue
            tid = _candidate_track_id(c)
            if tid is None:
                continue
            if int(tid) in blocked_track_ids:
                continue
            if _is_track_xyz_blocked(int(tid)):
                continue
            xyz_proj = _candidate_xyz_from_projected_rows(c, projected_rows)
            if _is_xyz_blocked_direct(xyz_proj):
                continue
            if _candidate_below_min_base_y(c, projected_rows):
                continue
            if _candidate_outside_pick_workspace(c, projected_rows):
                continue
            return True
        return False

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

    def _collect_track_measurement(
        track_id: int,
        first_obs: SceneObservation,
        first_candidate: dict,
        sample_count_override: int | None = None,
    ) -> dict:
        samples = (
            max(1, int(sample_count_override))
            if sample_count_override is not None
            else max(1, int(PICK_OTHER_VALIDATE_SAMPLES))
        )
        timeout_s = max(0.4, float(PICK_OTHER_VALIDATE_TIMEOUT_S))
        xyz_rows: list[np.ndarray] = []
        obs_local: SceneObservation | None = first_obs
        candidate_local: dict | None = dict(first_candidate)
        t0_local = time.time()
        while len(xyz_rows) < samples and (time.time() - t0_local) < timeout_s:
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
                    projected_rows=obs_local.projected_rows,
                )
            if candidate_local is not None:
                xyz = _resolve_candidate_xyz(obs_local, candidate_local)
                if xyz is not None:
                    arr = np.array(xyz, dtype=float).reshape(-1)
                    if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                        xyz_rows.append(np.array([float(arr[0]), float(arr[1]), float(arr[2])], dtype=float))
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
                    f"{status_prefix} measure id={int(track_id)} hits={len(xyz_rows)}/{samples}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                if _show_center_frame(SHOW_WINDOW, disp):
                    break
            obs_local = None
            candidate_local = None
            time.sleep(max(0.0, float(arm.sample_time)))
        median_xyz = None
        if xyz_rows:
            med = np.median(np.array(xyz_rows, dtype=float), axis=0)
            median_xyz = [float(med[0]), float(med[1]), float(med[2])]
        return {
            "samples": int(samples),
            "hits": int(len(xyz_rows)),
            "median_xyz": median_xyz,
        }

    def _distance_to_blocked_xyz(xyz: list[float] | None) -> tuple[float, float]:
        if xyz is None:
            return float("inf"), float("inf")
        arr = np.array(xyz, dtype=float).reshape(-1)
        if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
            return float("inf"), float("inf")
        best_xy = float("inf")
        best_z = float("inf")
        for bxyz in blocked_xyzs:
            if bxyz.size < 3 or not np.all(np.isfinite(bxyz[:3])):
                continue
            d_xy = float(math.hypot(float(arr[0]) - float(bxyz[0]), float(arr[1]) - float(bxyz[1])))
            d_z = float(abs(float(arr[2]) - float(bxyz[2])))
            if d_xy < best_xy:
                best_xy = d_xy
                best_z = d_z
        return best_xy, best_z

    centered_frames = deque(maxlen=int(centered_frames_req))
    target_track_id: int | None = (None if required_track_id_i is None else int(required_track_id_i))
    selected_track_id: int | None = None
    selected_xyz: list[float] | None = None
    last_decision = "none"
    last_reason = ""
    last_candidate_track_id: int | None = None
    no_candidate_frames = 0
    reject_count = 0
    loop_exit_reason = "timeout"
    frame_idx = 0
    last_seen_track_id: int | None = None
    last_seen_uv: tuple[int, int] | None = None
    track_stable_count = 0
    stabilize_until_t = 0.0
    if track_stable_frames_override is None:
        track_stable_frames_req = max(1, int(PLACE_VERIFY_V2_TRACK_STABLE_FRAMES))
    else:
        track_stable_frames_req = max(1, int(track_stable_frames_override))
    track_max_jump_px = max(0.0, float(PLACE_VERIFY_V2_TRACK_MAX_JUMP_PX))
    track_shift_pause_s = max(0.0, float(PLACE_VERIFY_V2_TRACK_SHIFT_PAUSE_S))
    accept_payload = None
    structural_reject_count = 0
    last_side_pref_source = "none"
    _reset_centering_integrator()

    if SHOW_WINDOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 720)

    try:
        t0 = time.time()
        session_deadline = float(t0 + hard_timeout_s)
        refresh_deadline = float(t0 + refresh_required_track_max_s)
        no_candidate_deadline = float(min(t0 + hard_timeout_s, t0 + no_candidate_timeout_s))
        while time.time() < float(session_deadline):
            if int(reject_count) >= int(reject_cap):
                loop_exit_reason = "reject_cap"
                break
            obs = observe_scene_frame(
                det=det,
                arm=arm,
                per=per,
                draw=bool(SHOW_WINDOW and ((UI_MODE == "debug") or UI_DRAW_ALL_BOXES)),
                projected_min_conf=float(detect_conf_local),
                state=state,
                update_tracks=True,
            )
            if obs is None:
                loop_exit_reason = "no_observation"
                break
            cx, cy = obs.image_center_uv
            img_display = obs.image_display
            elapsed = float(time.time() - t0)
            _draw_center_reference_overlay(
                img_display=img_display,
                cx=int(cx),
                cy=int(cy),
                elapsed_s=elapsed,
                timeout_s=float(hard_timeout_s),
            )

            selected = None
            if target_track_id is not None:
                if int(target_track_id) in blocked_track_ids:
                    if required_track_id_i is None:
                        target_track_id = None
                        last_seen_track_id = None
                        last_seen_uv = None
                        track_stable_count = 0
                        centered_frames.clear()
                    else:
                        blocked_track_ids.discard(int(target_track_id))
                else:
                    selected = _find_track_candidate(
                        obs.candidates,
                        int(target_track_id),
                        min_conf=float(detect_conf_local),
                        projected_rows=obs.projected_rows,
                    )
                    if selected is None:
                        if required_track_id_i is None:
                            target_track_id = None
                            last_seen_track_id = None
                            last_seen_uv = None
                            track_stable_count = 0
                            centered_frames.clear()

            if target_track_id is None:
                if _has_any_track_candidate(obs.candidates, obs.projected_rows, float(detect_conf_local)):
                    no_candidate_deadline = float(
                        min(float(t0 + hard_timeout_s), float(time.time() + no_candidate_timeout_s))
                    )
                selected = _select_top_track_candidate(
                    obs.candidates,
                    obs.projected_rows,
                    cx=int(cx),
                    cy=int(cy),
                    min_conf=float(detect_conf_local),
                )
                if selected is None:
                    no_candidate_frames += 1
                    if max_no_candidate_frames_i > 0 and int(no_candidate_frames) >= int(max_no_candidate_frames_i):
                        loop_exit_reason = "no_track_candidate_frames_exceeded"
                        break
                    if float(time.time()) >= float(no_candidate_deadline):
                        loop_exit_reason = "no_track_candidate_timeout"
                        break
                    selected_track_id = None
                    centered_frames.clear()
                    cv2.putText(
                        img_display,
                        f"{status_prefix} waiting: no track-id candidate",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                    if SHOW_WINDOW:
                        img_display = render_operator_overlay(
                            frame=img_display,
                            state=None,
                            ui_mode=UI_MODE,
                            tracks={},
                            active_track_id=None,
                            cx=int(cx),
                            cy=int(cy),
                            selected_uv=None,
                            status_line=f"{status_prefix}_waiting rejects={reject_count}/{reject_cap}",
                        )
                        if _show_center_frame(SHOW_WINDOW, img_display):
                            loop_exit_reason = "user_abort"
                            break
                    time.sleep(0.03)
                    continue
                target_track_id = int(selected.get("track_id"))
                if bool(handoff_pixel_only_side):
                    rank_tag = str(selected.get("handoff_rank", "pixel_top") or "pixel_top")
                    print(
                        f"[PickMisplacedHandoffPick] tid={int(target_track_id)} "
                        f"uv=({int(selected.get('u', 0))},{int(selected.get('v', 0))}) "
                        f"rank={rank_tag} side={str(preferred_section_norm)}"
                    )
                no_candidate_frames = 0
                centered_frames.clear()
            elif selected is None and required_track_id_i is not None:
                no_candidate_frames += 1
                if max_no_candidate_frames_i > 0 and int(no_candidate_frames) >= int(max_no_candidate_frames_i):
                    loop_exit_reason = "no_track_candidate_frames_exceeded"
                    break
                if float(time.time()) >= float(no_candidate_deadline):
                    loop_exit_reason = "no_track_candidate_timeout"
                    break
                selected_track_id = None
                centered_frames.clear()
                cv2.putText(
                    img_display,
                    f"{status_prefix} waiting: track_id={int(required_track_id_i)}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                if SHOW_WINDOW:
                    img_display = render_operator_overlay(
                        frame=img_display,
                        state=None,
                        ui_mode=UI_MODE,
                        tracks={},
                        active_track_id=None,
                        cx=int(cx),
                        cy=int(cy),
                        selected_uv=None,
                        status_line=f"{status_prefix}_waiting track={int(required_track_id_i)}",
                    )
                    if _show_center_frame(SHOW_WINDOW, img_display):
                        loop_exit_reason = "user_abort"
                        break
                time.sleep(0.03)
                continue

            if selected is None and target_track_id is not None:
                selected = _find_track_candidate(
                    obs.candidates,
                    int(target_track_id),
                    min_conf=float(detect_conf_local),
                    projected_rows=obs.projected_rows,
                )
            if required_track_id_i is not None and selected is not None:
                no_candidate_deadline = float(
                    min(float(t0 + hard_timeout_s), float(time.time() + no_candidate_timeout_s))
                )
                if (
                    bool(refresh_required_track_timeout_on_visible)
                    and int(_candidate_track_id(selected) or -1) == int(required_track_id_i)
                    and float(session_deadline) < float(refresh_deadline)
                    and (float(session_deadline) - float(time.time())) <= 0.25
                ):
                    next_deadline = float(min(float(refresh_deadline), float(time.time() + hard_timeout_s)))
                    if next_deadline > float(session_deadline) + 1e-3:
                        session_deadline = float(next_deadline)
                        print(
                            f"[StartupHydrateCenterRefresh] track={int(required_track_id_i)} "
                            f"reason=still_visible"
                        )
            if selected is None:
                no_candidate_frames += 1
                if max_no_candidate_frames_i > 0 and int(no_candidate_frames) >= int(max_no_candidate_frames_i):
                    loop_exit_reason = "no_track_candidate_frames_exceeded"
                    break
                centered_frames.clear()
                time.sleep(0.03)
                continue

            no_candidate_frames = 0
            selected_track_id = int(selected.get("track_id"))
            last_side_pref_source = str(selected.get("side_pref_source", "none") or "none")
            u = int(selected.get("u", 0))
            v = int(selected.get("v", 0))
            conf = float(selected.get("conf", 0.0))
            ex = int(u - int(cx))
            ey = int(v - int(cy))
            now_t = time.time()
            track_shifted = False
            jump_px = 0.0
            if last_seen_track_id is None or int(selected_track_id) != int(last_seen_track_id):
                track_stable_count = 1
                track_shifted = True
            else:
                if last_seen_uv is not None:
                    jump_px = float(math.hypot(float(u - int(last_seen_uv[0])), float(v - int(last_seen_uv[1]))))
                if jump_px > float(track_max_jump_px):
                    track_stable_count = 1
                    track_shifted = True
                else:
                    track_stable_count += 1
            last_seen_track_id = int(selected_track_id)
            last_seen_uv = (int(u), int(v))
            if track_shifted:
                stabilize_until_t = max(float(stabilize_until_t), float(now_t + track_shift_pause_s))

            cv2.circle(img_display, (u, v), 12, (0, 255, 0), 2)
            cv2.drawMarker(img_display, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
            cv2.line(img_display, (int(cx), int(cy)), (u, v), (255, 0, 255), 2)
            cv2.putText(
                img_display,
                f"{status_prefix} id={selected_track_id} err X={ex}px Y={ey}px "
                f"rejects={reject_count}/{reject_cap} stable={min(int(track_stable_count), int(track_stable_frames_req))}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            waiting_stable = False
            if not bool(disable_stable_gate):
                waiting_stable = bool(
                    now_t < float(stabilize_until_t) or int(track_stable_count) < int(track_stable_frames_req)
                )
            if waiting_stable:
                centered_frames.clear()
                if SHOW_WINDOW:
                    img_display = render_operator_overlay(
                        frame=img_display,
                        state=None,
                        ui_mode=UI_MODE,
                        tracks={},
                        active_track_id=None,
                        cx=int(cx),
                        cy=int(cy),
                        selected_uv=(u, v),
                        status_line=f"{status_prefix}_stabilizing id={selected_track_id}",
                    )
                    if _show_center_frame(SHOW_WINDOW, img_display):
                        loop_exit_reason = "user_abort"
                        break
                time.sleep(max(0.01, min(0.10, float(track_shift_pause_s))))
                continue

            if abs(ex) <= PX_TOL and abs(ey) <= PX_TOL and conf >= commit_conf_local:
                centered_frames.append((u, v))
                if len(centered_frames) >= centered_frames.maxlen:
                    decision_row = {"decision": "accept"}
                    obs_cb = obs
                    selected_cb = dict(selected)
                    if float(post_lock_refresh_s) > 0.0 and selected_track_id is not None:
                        t_refresh0 = float(time.time())
                        refresh_hits = 0
                        while True:
                            elapsed_refresh = float(time.time() - t_refresh0)
                            if elapsed_refresh >= float(post_lock_refresh_s) and refresh_hits >= int(post_lock_refresh_min_frames):
                                break
                            obs_refresh = observe_scene_frame(
                                det=det,
                                arm=arm,
                                per=per,
                                draw=bool(SHOW_WINDOW and ((UI_MODE == "debug") or UI_DRAW_ALL_BOXES)),
                                projected_min_conf=float(detect_conf_local),
                                state=state,
                                update_tracks=True,
                            )
                            if obs_refresh is None:
                                time.sleep(max(0.01, float(arm.sample_time)))
                                continue
                            cand_refresh = _find_track_candidate(
                                obs_refresh.candidates,
                                int(selected_track_id),
                                min_conf=float(detect_conf_local),
                                projected_rows=obs_refresh.projected_rows,
                            )
                            if cand_refresh is not None:
                                obs_cb = obs_refresh
                                selected_cb = dict(cand_refresh)
                                refresh_hits += 1
                            if SHOW_WINDOW:
                                try:
                                    uv_now = (
                                        int(selected_cb.get("u", int(u))),
                                        int(selected_cb.get("v", int(v))),
                                    )
                                    frame_now = render_operator_overlay(
                                        frame=obs_refresh.image_display,
                                        state=None,
                                        ui_mode=UI_MODE,
                                        tracks={},
                                        active_track_id=None,
                                        cx=int(obs_refresh.image_center_uv[0]),
                                        cy=int(obs_refresh.image_center_uv[1]),
                                        selected_uv=uv_now,
                                        status_line=(
                                            f"{status_prefix}_refresh id={int(selected_track_id)} "
                                            f"{refresh_hits}/{int(post_lock_refresh_min_frames)} "
                                            f"{elapsed_refresh:.1f}/{float(post_lock_refresh_s):.1f}s"
                                        ),
                                    )
                                    if _show_center_frame(SHOW_WINDOW, frame_now):
                                        loop_exit_reason = "user_abort"
                                        break
                                except Exception:
                                    pass
                            time.sleep(max(0.01, float(arm.sample_time)))
                        if str(loop_exit_reason).strip().lower() == "user_abort":
                            break
                    if callable(on_locked_candidate):
                        try:
                            decision_row = on_locked_candidate(
                                obs=obs_cb,
                                selected_row=dict(selected_cb),
                                track_id=int(selected_track_id),
                                collect_track_measurement=_collect_track_measurement,
                                distance_to_blocked_xyz=_distance_to_blocked_xyz,
                                blocked_xyzs=blocked_xyzs,
                                xy_margin_m=float(xy_margin_m),
                                z_margin_m=float(z_margin_m),
                            ) or {"decision": "continue"}
                        except Exception as exc:
                            decision_row = {"decision": "continue", "reason": f"callback_error:{exc}"}
                    decision = str(decision_row.get("decision", "continue")).strip().lower()
                    reason = str(decision_row.get("reason", "")).strip()
                    blocked_xyz_row = decision_row.get("blocked_xyz", None)
                    selected_xyz = decision_row.get("selected_xyz", None)
                    accept_payload = decision_row.get("accept_payload", None)
                    last_decision = str(decision or "continue")
                    last_reason = str(reason or "")
                    last_candidate_track_id = (None if selected_track_id is None else int(selected_track_id))
                    if decision == "accept":
                        state.active_target_track_id = int(selected_track_id)
                        loop_exit_reason = str(decision_row.get("exit_reason", "accepted")).strip() or "accepted"
                        if SHOW_WINDOW:
                            cv2.putText(
                                img_display,
                                f"{status_prefix.upper()} LOCKED",
                                (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                            )
                            img_display = render_operator_overlay(
                                frame=img_display,
                                state=None,
                                ui_mode=UI_MODE,
                                tracks={},
                                active_track_id=None,
                                cx=int(cx),
                                cy=int(cy),
                                selected_uv=(u, v),
                                status_line=f"{status_prefix}_locked id={selected_track_id}",
                            )
                            _show_center_frame(SHOW_WINDOW, img_display)
                        return {
                            "status": "ok",
                            "centered_pos": (int(u), int(v)),
                            "selected_track_id": int(selected_track_id),
                            "selected_xyz": selected_xyz,
                            "reject_count": int(reject_count),
                            "reject_cap": int(reject_cap),
                            "blocked_track_ids": sorted([int(tid) for tid in blocked_track_ids]),
                            "blocked_xyzs": _blocked_xyzs_for_result(),
                            "exit_reason": str(loop_exit_reason),
                            "accept_payload": accept_payload,
                            "last_decision": str(last_decision),
                            "last_reason": str(last_reason),
                            "last_candidate_track_id": (
                                None if last_candidate_track_id is None else int(last_candidate_track_id)
                            ),
                            "no_candidate_frames": int(no_candidate_frames),
                            "structural_reject_count": int(structural_reject_count),
                            "last_side_pref_source": str(last_side_pref_source),
                        }
                    if decision == "reject":
                        soft_wrong_section = bool(wrong_section_soft_skip) and str(reason).strip().lower().startswith("wrong_section")
                        structural_skip = str(reason).strip().lower().startswith("candidate_pick_space_rejected")
                        terminal_duplicate = str(reason).strip().lower().startswith("duplicate_measurement")
                        callback_terminal = bool(decision_row.get("terminal", False))
                        callback_exit_reason = str(decision_row.get("exit_reason", "")).strip()
                        if not bool(soft_wrong_section) and not bool(structural_skip):
                            reject_count += 1
                        if bool(structural_skip):
                            structural_reject_count += 1
                        if not (required_track_id_i is not None and int(selected_track_id) == int(required_track_id_i)):
                            blocked_track_ids.add(int(selected_track_id))
                        _append_block_uv(int(u), int(v), radius_px=float(PLACE_VERIFY_V2_RECENTER_DYNAMIC_BLACKLIST_PX))
                        _append_block_xyz(blocked_xyz_row)
                        xyz_text = None
                        if isinstance(selected_xyz, (list, tuple)) and len(selected_xyz) >= 3:
                            try:
                                xyz_text = (
                                    f"({float(selected_xyz[0]):.3f},"
                                    f"{float(selected_xyz[1]):.3f},"
                                    f"{float(selected_xyz[2]):.3f})"
                                )
                            except (TypeError, ValueError):
                                xyz_text = None
                        if bool(soft_wrong_section):
                            print(
                                f"{log_prefix} skip track_id={int(selected_track_id)} uv=({int(u)},{int(v)}) "
                                f"xyz={xyz_text} reason={reason or 'wrong_section'} "
                                f"reject_i={int(reject_count)}/{int(reject_cap)} soft_skip=yes"
                            )
                        else:
                            print(
                                f"{log_prefix} reject track_id={int(selected_track_id)} uv=({int(u)},{int(v)}) "
                                f"xyz={xyz_text} reason={reason or 'rejected'} "
                                f"reject_i={int(reject_count)}/{int(reject_cap)}"
                            )
                        if terminal_duplicate:
                            loop_exit_reason = "duplicate_measurement"
                            break
                        if bool(callback_terminal) and bool(terminal_callback_decision):
                            loop_exit_reason = str(callback_exit_reason or reason or "terminal_reject")
                            break
                        target_track_id = (
                            None if required_track_id_i is None else int(required_track_id_i)
                        )
                        selected_track_id = None
                        centered_frames.clear()
                        continue
                    # continue: neither accepted nor rejected hard
                    if str(reason).strip().lower().startswith("already_committed_cube"):
                        blocked_track_ids.add(int(selected_track_id))
                        _append_block_xyz(selected_xyz)
                        print(
                            f"{log_prefix} skip track_id={int(selected_track_id)} "
                            f"reason={reason or 'already_committed_cube'}"
                        )
                        target_track_id = (
                            None if required_track_id_i is None else int(required_track_id_i)
                        )
                        selected_track_id = None
                        centered_frames.clear()
                        continue
                    if str(reason).strip().lower().startswith("quality_gate_wait"):
                        if SHOW_WINDOW:
                            img_display = render_operator_overlay(
                                frame=img_display,
                                state=None,
                                ui_mode=UI_MODE,
                                tracks={},
                                active_track_id=None,
                                cx=int(cx),
                                cy=int(cy),
                                selected_uv=(u, v),
                                status_line=f"{status_prefix}_quality_wait id={selected_track_id}",
                            )
                            if _show_center_frame(SHOW_WINDOW, img_display):
                                loop_exit_reason = "user_abort"
                                break
                        time.sleep(max(0.02, float(arm.sample_time)))
                        continue
                    print(f"{log_prefix} lock id={int(selected_track_id)} continue reason={reason or 'unspecified'}")
                    target_track_id = (
                        None if required_track_id_i is None else int(required_track_id_i)
                    )
                    selected_track_id = None
                    centered_frames.clear()
                    continue
                if SHOW_WINDOW:
                    img_display = render_operator_overlay(
                        frame=img_display,
                        state=None,
                        ui_mode=UI_MODE,
                        tracks={},
                        active_track_id=None,
                        cx=int(cx),
                        cy=int(cy),
                        selected_uv=(u, v),
                        status_line=f"{status_prefix}_hold id={selected_track_id} {len(centered_frames)}/{centered_frames.maxlen}",
                    )
                    if _show_center_frame(SHOW_WINDOW, img_display):
                        loop_exit_reason = "user_abort"
                        break
                time.sleep(0.03)
                continue

            centered_frames.clear()
            ey_cmd = int(round(float(ey) * float(center_ey_scale)))
            frame_idx = _maybe_apply_centering_nudge(
                arm,
                ex,
                ey_cmd,
                conf,
                frame_idx,
                detect_conf=float(DETECT_CONF),
                center_verbose=bool(CENTER_VERBOSE),
            )
            if SHOW_WINDOW:
                img_display = render_operator_overlay(
                    frame=img_display,
                    state=None,
                    ui_mode=UI_MODE,
                    tracks={},
                    active_track_id=None,
                    cx=int(cx),
                    cy=int(cy),
                    selected_uv=(u, v),
                    status_line=f"{status_prefix}_recentering id={selected_track_id}",
                )
                if _show_center_frame(SHOW_WINDOW, img_display):
                    loop_exit_reason = "user_abort"
                    break
            time.sleep(0.01)
    finally:
        _reset_centering_integrator()
        if SHOW_WINDOW and bool(close_window_on_exit):
            cv2.destroyAllWindows()

    handoff_status = "observe_retry"
    if str(last_decision).strip().lower() == "accept" and selected_track_id is not None:
        handoff_status = "ok"
    exit_reason_out = str(loop_exit_reason if loop_exit_reason != "timeout" else "timeout_or_uncertain")

    return {
        "status": str(handoff_status),
        "centered_pos": None,
        "selected_track_id": (None if selected_track_id is None else int(selected_track_id)),
        "selected_xyz": selected_xyz,
        "reject_count": int(reject_count),
        "reject_cap": int(reject_cap),
        "blocked_track_ids": sorted([int(tid) for tid in blocked_track_ids]),
        "blocked_xyzs": _blocked_xyzs_for_result(),
        "exit_reason": exit_reason_out,
        "accept_payload": accept_payload,
        "last_decision": str(last_decision),
        "last_reason": str(last_reason),
        "last_candidate_track_id": (None if last_candidate_track_id is None else int(last_candidate_track_id)),
        "no_candidate_frames": int(no_candidate_frames),
        "structural_reject_count": int(structural_reject_count),
        "last_side_pref_source": str(last_side_pref_source),
    }

def run_return_verify_stage(
    *,
    state: CycleState,
    arm: Arm,
    per: Perception | None,
    det: YOLODetector | None,
    label_prefix: str,
    return_target_xyz: list[float] | tuple[float, float, float] | np.ndarray | None,
    required_hits: int = 8,
    measurement_samples: int | None = None,
    reject_cap: int | None = None,
    hard_timeout_s: float | None = None,
    verify_look_pose: np.ndarray | list[float] | tuple[float, float, float, float] | None = None,
) -> dict:
    _bind_core_globals()
    reject_cap_i = max(1, int(PICK_OTHER_MAX_REJECTS if reject_cap is None else reject_cap))
    hard_timeout_s_f = max(1.0, float(PICK_OTHER_HARD_TIMEOUT_S if hard_timeout_s is None else hard_timeout_s))
    xy_margin_m = float(PLACE_VERIFY_V2_XY_MARGIN_M)
    z_margin_m = float(PLACE_VERIFY_V2_Z_MARGIN_M)
    required_hits_i = max(1, int(required_hits))
    measurement_samples_i = max(
        int(PICK_OTHER_VALIDATE_SAMPLES),
        int(required_hits_i if measurement_samples is None else measurement_samples),
    )
    target_xyz = None
    if isinstance(return_target_xyz, (list, tuple, np.ndarray)):
        try:
            arr = np.array(return_target_xyz, dtype=float).reshape(-1)
        except Exception:
            arr = np.array([np.nan, np.nan, np.nan], dtype=float)
        if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
            target_xyz = np.array([float(arr[0]), float(arr[1]), float(arr[2])], dtype=float)
    if target_xyz is None:
        return {
            "status": "observe_retry",
            "returned_verified": False,
            "blocked_return_track_id": None,
            "blocked_return_uv": None,
            "blocked_return_xyz": None,
            "return_target_xyz": None,
            "required_hits": int(required_hits_i),
            "measurement_samples": int(measurement_samples_i),
            "return_verify": {
                "status": "invalid_return_target",
                "confirmed": False,
                "hits": 0,
                "samples": int(required_hits_i),
                "xy_error_m": float("inf"),
                "z_error_m": float("inf"),
                "expected_xyz": None,
                "measured_xyz": None,
            },
            "verify_exit_reason": "invalid_return_target",
        }
    print(
        f"[ReturnVerify] start target_xyz=({float(target_xyz[0]):.3f},{float(target_xyz[1]):.3f},{float(target_xyz[2]):.3f}) "
        f"required_hits={int(required_hits_i)} measure_samples={int(measurement_samples_i)}"
    )
    look_pose = PICK_LOOKING if verify_look_pose is None else np.array(verify_look_pose, dtype=float).reshape(-1)
    if look_pose.size < 4 or not np.all(np.isfinite(look_pose[:4])):
        look_pose = PICK_LOOKING
    arm.goto_task_space(look_pose, duration=1.2, label=f"{label_prefix}_return_verify_look")
    time.sleep(0.2)

    def _verify_returned_candidate(
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
        _ = (distance_to_blocked_xyz, blocked_xyzs)
        measure = collect_track_measurement(
            track_id=int(track_id),
            first_obs=obs,
            first_candidate=selected_row,
            sample_count_override=int(measurement_samples_i),
        )
        hits = int(measure.get("hits", 0))
        selected_xyz = measure.get("median_xyz")
        selected_arr = np.array(
            selected_xyz if isinstance(selected_xyz, (list, tuple, np.ndarray)) else [np.nan, np.nan, np.nan],
            dtype=float,
        ).reshape(-1)
        xyz_ok = bool(selected_arr.size >= 3 and np.all(np.isfinite(selected_arr[:3])))
        if not xyz_ok or hits < int(required_hits_i):
            return {
                "decision": "continue",
                "reason": f"quality_gate_wait hits={hits}/{int(required_hits_i)} xyz_ok={bool(xyz_ok)}",
                "selected_xyz": (None if not xyz_ok else [float(selected_arr[0]), float(selected_arr[1]), float(selected_arr[2])]),
            }
        d_xy = float(math.hypot(float(selected_arr[0]) - float(target_xyz[0]), float(selected_arr[1]) - float(target_xyz[1])))
        d_z = float(abs(float(selected_arr[2]) - float(target_xyz[2])))
        near_target = bool(d_xy <= float(xy_margin_m) and d_z <= float(z_margin_m))
        if near_target:
            print(
                f"[ReturnVerify] accept track_id={int(track_id)} hits={hits}/{int(required_hits_i)} "
                f"xyz=({float(selected_arr[0]):.3f},{float(selected_arr[1]):.3f},{float(selected_arr[2]):.3f}) "
                f"d_xy_target={float(d_xy):.3f} d_z_target={float(d_z):.3f}"
            )
            return {
                "decision": "accept",
                "reason": "returned_cube_verified",
                "selected_xyz": [float(selected_arr[0]), float(selected_arr[1]), float(selected_arr[2])],
                "accept_payload": {
                    "hits": int(hits),
                    "required_hits": int(required_hits_i),
                    "d_xy_target": float(d_xy),
                    "d_z_target": float(d_z),
                },
            }
        return {
            "decision": "reject",
            "reason": f"not_returned_cube d_xy={float(d_xy):.3f} d_z={float(d_z):.3f}",
            "blocked_xyz": [float(selected_arr[0]), float(selected_arr[1]), float(selected_arr[2])],
            "selected_xyz": [float(selected_arr[0]), float(selected_arr[1]), float(selected_arr[2])],
        }

    verify_session = run_track_handoff_session(
        state=state,
        arm=arm,
        per=per,
        det=det,
        reject_cap=int(reject_cap_i),
        hard_timeout_s=float(hard_timeout_s_f),
        xy_margin_m=float(xy_margin_m),
        z_margin_m=float(z_margin_m),
        blocked_track_id=None,
        blocked_xyz=None,
        blocked_uv=None,
        status_prefix="return_verify",
        log_prefix="[ReturnVerify]",
        on_locked_candidate=_verify_returned_candidate,
    )
    if str(verify_session.get("status", "")) != "ok":
        print(
            f"[ReturnVerify] summary status=observe_retry rejects={int(verify_session.get('reject_count', 0))}/"
            f"{int(verify_session.get('reject_cap', 0))} exit_reason={verify_session.get('exit_reason')}"
        )
        return {
            "status": "observe_retry",
            "returned_verified": False,
            "blocked_return_track_id": None,
            "blocked_return_uv": None,
            "blocked_return_xyz": None,
            "return_target_xyz": [float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])],
            "required_hits": int(required_hits_i),
            "measurement_samples": int(measurement_samples_i),
            "return_verify": {
                "status": str(verify_session.get("exit_reason", "observe_retry")),
                "confirmed": False,
                "hits": int((verify_session.get("accept_payload") or {}).get("hits", 0)),
                "samples": int(required_hits_i),
                "xy_error_m": float((verify_session.get("accept_payload") or {}).get("d_xy_target", float("inf"))),
                "z_error_m": float((verify_session.get("accept_payload") or {}).get("d_z_target", float("inf"))),
                "expected_xyz": [float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])],
                "measured_xyz": None,
            },
            "verify_exit_reason": str(verify_session.get("exit_reason", "observe_retry")),
        }

    blocked_return_track_id = verify_session.get("selected_track_id", None)
    blocked_return_uv = verify_session.get("centered_pos", None)
    blocked_return_xyz = verify_session.get("selected_xyz", None)
    print(
        f"[ReturnVerify] summary status=ok track_id={blocked_return_track_id} "
        f"uv={blocked_return_uv} xyz={blocked_return_xyz} "
        f"rejects={int(verify_session.get('reject_count', 0))}/{int(verify_session.get('reject_cap', 0))}"
    )
    return {
        "status": "ok",
        "returned_verified": True,
        "blocked_return_track_id": (None if blocked_return_track_id is None else int(blocked_return_track_id)),
        "blocked_return_uv": (
            None
            if not isinstance(blocked_return_uv, (list, tuple)) or len(blocked_return_uv) < 2
            else [int(blocked_return_uv[0]), int(blocked_return_uv[1])]
        ),
        "blocked_return_xyz": (
            None
            if not isinstance(blocked_return_xyz, (list, tuple)) or len(blocked_return_xyz) < 3
            else [float(blocked_return_xyz[0]), float(blocked_return_xyz[1]), float(blocked_return_xyz[2])]
        ),
        "return_target_xyz": [float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])],
        "required_hits": int(required_hits_i),
        "measurement_samples": int(measurement_samples_i),
        "return_verify": {
            "status": "returned_verified",
            "confirmed": True,
            "hits": int((verify_session.get("accept_payload") or {}).get("hits", 0)),
            "samples": int(required_hits_i),
            "xy_error_m": float((verify_session.get("accept_payload") or {}).get("d_xy_target", float("inf"))),
            "z_error_m": float((verify_session.get("accept_payload") or {}).get("d_z_target", float("inf"))),
            "expected_xyz": [float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])],
            "measured_xyz": (
                None
                if not isinstance(blocked_return_xyz, (list, tuple)) or len(blocked_return_xyz) < 3
                else [float(blocked_return_xyz[0]), float(blocked_return_xyz[1]), float(blocked_return_xyz[2])]
            ),
        },
        "verify_exit_reason": str(verify_session.get("exit_reason", "confirmed")),
    }


def run_return_handoff_stage(
    *,
    state: CycleState,
    arm: Arm,
    per: Perception | None,
    det: YOLODetector | None,
    blocked_return_track_id: int | None,
    blocked_return_uv: list[int] | tuple[int, int] | None,
    blocked_return_xyz: list[float] | tuple[float, float, float] | np.ndarray | None,
    required_hits: int,
    measurement_samples: int,
    reject_cap: int,
    hard_timeout_s: float,
    existing_blocked_track_ids: set[int] | list[int] | None = None,
    existing_blocked_xyzs: list[list[float]] | None = None,
) -> dict:
    _bind_core_globals()
    xy_margin_m = float(PLACE_VERIFY_V2_XY_MARGIN_M)
    z_margin_m = float(PLACE_VERIFY_V2_Z_MARGIN_M)

    def _handoff_next_candidate(
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
        _ = (distance_to_blocked_xyz, blocked_xyzs)
        measure = collect_track_measurement(
            track_id=int(track_id),
            first_obs=obs,
            first_candidate=selected_row,
            sample_count_override=int(measurement_samples),
        )
        hits = int(measure.get("hits", 0))
        selected_xyz = measure.get("median_xyz")
        selected_arr = np.array(
            selected_xyz if isinstance(selected_xyz, (list, tuple, np.ndarray)) else [np.nan, np.nan, np.nan],
            dtype=float,
        ).reshape(-1)
        xyz_ok = bool(selected_arr.size >= 3 and np.all(np.isfinite(selected_arr[:3])))
        if not xyz_ok or hits < int(required_hits):
            return {
                "decision": "continue",
                "reason": f"quality_gate_wait hits={hits}/{int(required_hits)} xyz_ok={bool(xyz_ok)}",
                "selected_xyz": (None if not xyz_ok else [float(selected_arr[0]), float(selected_arr[1]), float(selected_arr[2])]),
            }
        d_xy_ret = float("inf")
        d_z_ret = float("inf")
        if isinstance(blocked_return_xyz, (list, tuple, np.ndarray)) and len(blocked_return_xyz) >= 3:
            b = np.array(blocked_return_xyz, dtype=float).reshape(-1)
            if b.size >= 3 and np.all(np.isfinite(b[:3])):
                d_xy_ret = float(math.hypot(float(selected_arr[0]) - float(b[0]), float(selected_arr[1]) - float(b[1])))
                d_z_ret = float(abs(float(selected_arr[2]) - float(b[2])))
        if np.isfinite(d_xy_ret) and np.isfinite(d_z_ret) and d_xy_ret <= float(xy_margin_m) and d_z_ret <= float(z_margin_m):
            return {
                "decision": "reject",
                "reason": f"returned_cube_reacquire d_xy={float(d_xy_ret):.3f} d_z={float(d_z_ret):.3f}",
                "blocked_xyz": [float(selected_arr[0]), float(selected_arr[1]), float(selected_arr[2])],
                "selected_xyz": [float(selected_arr[0]), float(selected_arr[1]), float(selected_arr[2])],
            }
        print(
            f"[ReturnHandoff] accept next track_id={int(track_id)} hits={hits}/{int(required_hits)} "
            f"xyz=({float(selected_arr[0]):.3f},{float(selected_arr[1]):.3f},{float(selected_arr[2]):.3f}) "
            f"d_xy_returned={float(d_xy_ret):.3f} d_z_returned={float(d_z_ret):.3f}"
        )
        return {
            "decision": "accept",
            "reason": "next_target_locked",
            "selected_xyz": [float(selected_arr[0]), float(selected_arr[1]), float(selected_arr[2])],
            "accept_payload": {
                "hits": int(hits),
                "required_hits": int(required_hits),
                "d_xy_returned": float(d_xy_ret),
                "d_z_returned": float(d_z_ret),
            },
        }

    handoff_session = run_track_handoff_session(
        state=state,
        arm=arm,
        per=per,
        det=det,
        reject_cap=int(max(1, int(reject_cap))),
        hard_timeout_s=float(max(1.0, float(hard_timeout_s))),
        xy_margin_m=float(xy_margin_m),
        z_margin_m=float(z_margin_m),
        blocked_track_id=(None if blocked_return_track_id is None else int(blocked_return_track_id)),
        blocked_xyz=(None if blocked_return_xyz is None else list(blocked_return_xyz)),
        blocked_track_ids_extra=existing_blocked_track_ids,
        blocked_xyzs_extra=existing_blocked_xyzs,
        blocked_uv=(None if blocked_return_uv is None else list(blocked_return_uv)),
        status_prefix="return_handoff",
        log_prefix="[ReturnHandoff]",
        on_locked_candidate=_handoff_next_candidate,
    )
    if str(handoff_session.get("status", "")) != "ok":
        print(
            f"[ReturnHandoff] summary status=observe_retry rejects={int(handoff_session.get('reject_count', 0))}/"
            f"{int(handoff_session.get('reject_cap', 0))} exit_reason={handoff_session.get('exit_reason')}"
        )
        return {
            "status": "observe_retry",
            "next_centered_uv": None,
            "next_track_id": None,
            "next_xyz": None,
            "next_color": "unknown",
            "next_color_conf": 0.0,
            "handoff": {
                "status": str(handoff_session.get("exit_reason", "observe_retry")),
                "hits": int((handoff_session.get("accept_payload") or {}).get("hits", 0)),
                "samples": int(required_hits),
            },
        }

    next_centered_uv = handoff_session.get("centered_pos", None)
    next_track_id = handoff_session.get("selected_track_id", None)
    next_xyz = handoff_session.get("selected_xyz", None)
    next_color = "unknown"
    next_color_conf = 0.0
    if isinstance(next_centered_uv, (list, tuple)) and len(next_centered_uv) >= 2 and per is not None and det is not None:
        try:
            color_frame, _depth = per.get_frames()
            img_now = np.asanyarray(color_frame.get_data())
            _img_ann, candidates = det.detect_candidates_and_draw(img_now, draw=False)
            target = choose_track_candidate_near_uv(
                candidates,
                (None if next_track_id is None else int(next_track_id)),
                int(next_centered_uv[0]),
                int(next_centered_uv[1]),
                min_conf=0.0,
            )
            if target is not None:
                next_color, next_color_conf = classify_cube_color_patch(
                    img_now,
                    bbox_xyxy=target.get("bbox_xyxy", None),
                    center_uv=None,
                    bbox_core_ratio=0.55,
                )
            else:
                print(
                    "[ReturnHandoff] track_bbox_missing "
                    f"track_id={None if next_track_id is None else int(next_track_id)} "
                    f"uv=({int(next_centered_uv[0])},{int(next_centered_uv[1])})"
                )
                next_color, next_color_conf = "unknown", 0.0
        except Exception as exc:
            print(f"[ReturnHandoff] auto-classify failed: {exc}")
            next_color, next_color_conf = "unknown", 0.0
    print(
        f"[ReturnHandoff] summary status=ok blocked_return_track={blocked_return_track_id} "
        f"next_track={next_track_id} next_uv={next_centered_uv} "
        f"next_color={next_color} conf={float(next_color_conf):.3f}"
    )
    return {
        "status": "ok",
        "next_centered_uv": (
            None if not isinstance(next_centered_uv, (list, tuple)) or len(next_centered_uv) < 2
            else [int(next_centered_uv[0]), int(next_centered_uv[1])]
        ),
        "next_track_id": (None if next_track_id is None else int(next_track_id)),
        "next_xyz": (
            None if not isinstance(next_xyz, (list, tuple)) or len(next_xyz) < 3
            else [float(next_xyz[0]), float(next_xyz[1]), float(next_xyz[2])]
        ),
        "next_color": str(next_color),
        "next_color_conf": float(next_color_conf),
        "handoff": {
            "status": "next_target_locked",
            "hits": int((handoff_session.get("accept_payload") or {}).get("hits", 0)),
            "samples": int(required_hits),
        },
    }


def run_return_verify_and_handoff_session(
    *,
    state: CycleState,
    arm: Arm,
    per: Perception | None,
    det: YOLODetector | None,
    label_prefix: str,
    return_target_xyz: list[float] | tuple[float, float, float] | np.ndarray | None,
    existing_blocked_track_ids: set[int] | list[int] | None = None,
    existing_blocked_xyzs: list[list[float]] | None = None,
) -> dict:
    _bind_core_globals()
    verify_stage = run_return_verify_stage(
        state=state,
        arm=arm,
        per=per,
        det=det,
        label_prefix=label_prefix,
        return_target_xyz=return_target_xyz,
        required_hits=8,
        measurement_samples=max(int(PICK_OTHER_VALIDATE_SAMPLES), 8),
        reject_cap=max(1, int(PICK_OTHER_MAX_REJECTS)),
        hard_timeout_s=max(1.0, float(PICK_OTHER_HARD_TIMEOUT_S)),
    )
    if not bool(verify_stage.get("returned_verified", False)):
        return {
            "status": str(verify_stage.get("status", "observe_retry")),
            "returned_verified": False,
            "blocked_return_track_id": None,
            "blocked_return_uv": None,
            "blocked_return_xyz": None,
            "next_centered_uv": None,
            "next_track_id": None,
            "next_xyz": None,
            "next_color": "unknown",
            "next_color_conf": 0.0,
            "return_verify": dict(verify_stage.get("return_verify", {})),
            "handoff": {"status": "not_run"},
        }
    handoff_stage = run_return_handoff_stage(
        state=state,
        arm=arm,
        per=per,
        det=det,
        blocked_return_track_id=verify_stage.get("blocked_return_track_id", None),
        blocked_return_uv=verify_stage.get("blocked_return_uv", None),
        blocked_return_xyz=verify_stage.get("blocked_return_xyz", None),
        required_hits=int(verify_stage.get("required_hits", 8)),
        measurement_samples=int(verify_stage.get("measurement_samples", max(int(PICK_OTHER_VALIDATE_SAMPLES), 8))),
        reject_cap=max(1, int(PICK_OTHER_MAX_REJECTS)),
        hard_timeout_s=max(1.0, float(PICK_OTHER_HARD_TIMEOUT_S)),
        existing_blocked_track_ids=existing_blocked_track_ids,
        existing_blocked_xyzs=existing_blocked_xyzs,
    )
    return {
        "status": str(handoff_stage.get("status", "observe_retry")),
        "returned_verified": True,
        "blocked_return_track_id": verify_stage.get("blocked_return_track_id", None),
        "blocked_return_uv": verify_stage.get("blocked_return_uv", None),
        "blocked_return_xyz": verify_stage.get("blocked_return_xyz", None),
        "next_centered_uv": handoff_stage.get("next_centered_uv", None),
        "next_track_id": handoff_stage.get("next_track_id", None),
        "next_xyz": handoff_stage.get("next_xyz", None),
        "next_color": str(handoff_stage.get("next_color", "unknown")),
        "next_color_conf": float(handoff_stage.get("next_color_conf", 0.0)),
        "return_verify": dict(verify_stage.get("return_verify", {})),
        "handoff": dict(handoff_stage.get("handoff", {})),
    }

def execute_pick_misplaced_cube_action(
    *,
    state: CycleState,
    arm: Arm,
    det: YOLODetector,
    per: Perception,
    label_prefix: str,
    preferred_section: str | None = None,
    safe_pick_reach_m: float | None = None,
) -> tuple[bool, str, dict | None]:
    _bind_core_globals()
    if bool(state.holding_object):
        return False, "holding_object", None
    side_pref = None if preferred_section is None else str(preferred_section).strip().lower()
    if side_pref not in {None, SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        return False, "invalid_preferred_section", {"preferred_section": preferred_section}
    if side_pref is None:
        if not can_pick_misplaced_cube_now(state):
            return False, "no_pick_misplaced_available", None
        candidate_sides = [
            side_name
            for side_name in [SECTION_LEFT_NAME, SECTION_RIGHT_NAME]
            if can_pick_misplaced_on_side_now(state, side_name)
        ]
    else:
        if not can_pick_misplaced_on_side_now(state, side_pref):
            return False, "requested_side_not_available", {"preferred_section": side_pref}
        candidate_sides = [str(side_pref)]
    if not candidate_sides:
        return False, "no_pick_misplaced_available", None

    context_base: dict[str, object] = {
        "preferred_section": side_pref,
        "candidate_sides": list(candidate_sides),
        "target_section": None,
        "target_xyz": None,
        "top_ref_xyz": None,
        "top_ref_z": None,
        "top_ref_missing": False,
        "drop_xyz": None,
        "lock_track_id": None,
        "lock_uv": None,
        "lock_xyz": None,
        "candidate_order": [],
        "attempted_track_ids": [],
        "failed_track_ids": [],
        "attempt_exit_reasons": [],
        "misplaced_seed": {},
        "misplaced_seed_attempts": [],
        "lock_attempt_details": [],
        "swap_used": False,
        "mismatch_strike_threshold": int(MISPLACED_PICK_MISMATCH_STRIKES_REQUIRED),
    }
    is_pick_placed_target = side_pref in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}

    def _ctx(extra: dict | None = None) -> dict:
        row = dict(context_base)
        if isinstance(extra, dict):
            row.update(extra)
        return row

    # Deterministic pre-scan: move to correction place-looking before selecting misplaced target.
    correction_place_look = np.array(PLACE_LOOKING, dtype=float).copy()
    if correction_place_look.size >= 2:
        correction_place_look[1] = float(LOOKING[1]) + float(MISPLACED_PLACE_LOOK_Y_OFFSET)
    if not arm.goto_task_space(
        correction_place_look,
        duration=1.0,
        label=f"{label_prefix}_misplaced_place_look",
    ):
        return False, "misplaced_place_look_failed", _ctx(
            {"motion_reason": str(getattr(arm, "last_motion_reason", "") or "")}
        )

    # Keep pick_placed targeting lock-first: do not hard-gate on coarse pre-scan seed rows.
    # We still validate side and top-ness inside the lock callback.
    target_reconcile: dict = {
        "status": "skipped_pre_scan",
        "mode": "pre_pick_misplaced_target",
    }
    section_centers_xy = dict(_verify_section_xy_centers())
    section_centers_source: dict[str, str] = {
        str(k).strip().lower(): "slot_geometry"
        for k in list(section_centers_xy.keys())
    }
    for _side_name in [SECTION_LEFT_NAME, SECTION_RIGHT_NAME]:
        anchor_xyz, anchor_source = get_latest_side_stack_anchor_xyz(state, str(_side_name))
        if not isinstance(anchor_xyz, (list, tuple)) or len(anchor_xyz) < 2:
            continue
        try:
            ax = float(anchor_xyz[0])
            ay = float(anchor_xyz[1])
        except Exception:
            continue
        if (not np.isfinite(ax)) or (not np.isfinite(ay)):
            continue
        side_norm = str(_side_name).strip().lower()
        section_centers_xy[str(side_norm)] = (float(ax), float(ay))
        section_centers_source[str(side_norm)] = str(anchor_source or "stack_anchor")
    target_section = str(candidate_sides[0])
    if side_pref in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        target_section = str(side_pref)
    elif len(candidate_sides) > 1:
        best_side = str(target_section)
        best_level = -1
        for side in list(candidate_sides):
            side_row = _planner_section_row_unified(state, str(side), stack_level_hint=None)
            try:
                side_level = int(side_row.get("stack_level", 0) or 0)
            except Exception:
                side_level = 0
            if side_level > best_level:
                best_level = int(side_level)
                best_side = str(side)
        target_section = str(best_side)
    target_xyz = None
    context_base["target_section"] = str(target_section)
    context_base["target_xyz"] = None
    context_base["section_centers_xy"] = {
        str(name): [float(center_xy[0]), float(center_xy[1])]
        for name, center_xy in dict(section_centers_xy).items()
    }
    context_base["section_center_sources"] = {
        str(name): str(src)
        for name, src in dict(section_centers_source).items()
    }
    print(
        f"[PickMisplacedCenters] target_side={str(target_section).strip().lower()} "
        f"centers_xy={context_base['section_centers_xy']} "
        f"sources={context_base['section_center_sources']}"
    )

    required_hits_i = max(1, int(MISPLACED_PICK_REQUIRED_HITS))
    measurement_samples_i = max(int(PICK_OTHER_VALIDATE_SAMPLES), int(MISPLACED_PICK_MEASURE_SAMPLES), int(required_hits_i))
    if bool(is_pick_placed_target):
        primary_timeout_s = max(0.5, float(PICK_PLACED_LOCK_TIMEOUT_S))
        pick_placed_max_total_lock_time_s = float(PICK_PLACED_MAX_TOTAL_LOCK_TIME_S)
    else:
        primary_timeout_s = max(0.5, float(MISPLACED_PICK_HARD_TIMEOUT_S))
        pick_placed_max_total_lock_time_s = float(MISPLACED_PICK_MAX_TOTAL_LOCK_TIME_S)
    target_section_norm = str(target_section).strip().lower()
    track_id_only_lock_enabled = bool(MISPLACED_PICK_TRACK_ID_ONLY_LOCK)
    if bool(track_id_only_lock_enabled):
        # Side-correction picks must validate section membership before grasp.
        track_id_only_lock_enabled = False
    context_base["top_ref_xyz"] = None
    context_base["top_ref_z"] = None
    context_base["top_ref_missing"] = True
    try:
        target_side_height_row = _planner_section_row_unified(state, str(target_section_norm), stack_level_hint=None)
        expected_target_level = int(target_side_height_row.get("stack_level", 0) or 0)
    except Exception:
        expected_target_level = 0
    expected_target_level = int(max(0, min(int(MAX_STACK_LEVELS_PER_SECTION), int(expected_target_level))))
    context_base["height_expected_level"] = int(expected_target_level)

    def _infer_section_for_misplaced_lock(base_x: float, base_y: float) -> tuple[str, str]:
        inferred_section, assign_xy = _infer_section_for_place_xy(
            float(base_x),
            float(base_y),
            section_centers_xy,
            # Do not hard-gate by place Y-band here; slight drift/out-of-band placements
            # should still map to nearest stack side for pick_placed recovery.
            band_min=None,
            band_max=None,
            max_center_dist_m=float(MISPLACED_PICK_SECTION_MAX_DIST_M),
        )
        inferred_norm = "" if inferred_section is None else str(inferred_section).strip().lower()
        if inferred_norm in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
            return inferred_norm, str(assign_xy.get("reason", "ok_xy"))
        return "", str(assign_xy.get("reason", "unknown"))

    mismatch_strikes_required = int(
        max(1, PICK_PLACED_VERIFY_STRIKES if bool(is_pick_placed_target) else MISPLACED_PICK_MISMATCH_STRIKES_REQUIRED)
    )
    context_base["mismatch_strike_threshold"] = int(mismatch_strikes_required)
    mismatch_strikes = {
        "wrong_section_fast": 0,
        "wrong_section": 0,
        "height_out_of_band": 0,
        "height_level_out_of_range": 0,
        "height_not_top_level": 0,
        "height_no_stack_to_pick": 0,
    }
    mismatch_track_strikes: dict[tuple[int, str], int] = {}
    quality_wait_by_track: dict[int, int] = {}
    attempt_no_progress_count = 0
    last_lock_callback_track_id: int | None = None

    def _mismatch_track_counts_snapshot() -> dict[str, int]:
        out: dict[str, int] = {}
        for (tid_k, tag_k), val_k in list(mismatch_track_strikes.items()):
            out[f"{int(tid_k)}:{str(tag_k)}"] = int(val_k)
        return out

    def _emit_pick_misplaced_lock_verify(
        *,
        track_id: int,
        median_xyz_row: list[float] | None,
        section_tag: str,
        height_tag: str,
        decision: str,
        reason: str,
    ) -> None:
        median_text = _misplaced_xyz_log_text(median_xyz_row)
        print(
            f"[PickMisplacedLockVerify] track={int(track_id)} median={median_text} "
            f"section={str(target_section_norm)}:{str(section_tag)} "
            f"height=top:{str(height_tag)} decision={str(decision)} reason={str(reason)}"
        )

    def _mismatch_probe_or_terminal_reject(
        *,
        tag: str,
        track_id_for_strike: int | None,
        reason_detail: str,
        selected_xyz_row: list[float] | None,
    ) -> dict:
        nonlocal attempt_no_progress_count
        tag_norm = str(tag)
        strikes_required_local = int(mismatch_strikes_required)
        strikes_now = 1
        if track_id_for_strike is not None:
            key = (int(track_id_for_strike), tag_norm)
            strikes_now = int(mismatch_track_strikes.get(key, 0)) + 1
            mismatch_track_strikes[key] = int(strikes_now)
        else:
            strikes_now = int(mismatch_strikes.get(tag_norm, 0)) + 1
            mismatch_strikes[tag_norm] = int(strikes_now)
        if int(strikes_now) < int(strikes_required_local):
            attempt_no_progress_count += 1
            if int(attempt_no_progress_count) >= int(MISPLACED_PICK_ATTEMPT_NO_PROGRESS_CAP):
                return {
                    "decision": "reject",
                    "reason": (
                        f"attempt_no_progress mismatch_probe:{str(tag_norm)} "
                        f"count={int(attempt_no_progress_count)}/{int(MISPLACED_PICK_ATTEMPT_NO_PROGRESS_CAP)}"
                    ),
                    "terminal": True,
                    "exit_reason": "attempt_no_progress",
                    "selected_xyz": selected_xyz_row,
                }
            return {
                "decision": "continue",
                "reason": (
                    f"mismatch_probe:{str(tag_norm)} strike={int(strikes_now)}/{int(strikes_required_local)} "
                    f"{reason_detail}"
                ),
                "selected_xyz": selected_xyz_row,
            }
        return {
            "decision": "reject",
            "reason": (
                f"{str(tag_norm)} expected_side={target_section_norm} "
                f"strike={int(strikes_now)}/{int(strikes_required_local)} {reason_detail}"
            ),
            "blocked_xyz": selected_xyz_row,
            "selected_xyz": selected_xyz_row,
            "terminal": True,
            "exit_reason": f"mismatch_terminal:{str(tag_norm)}",
        }

    def _lock_requested_side_candidate(
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
        _ = (obs, selected_row, distance_to_blocked_xyz, blocked_xyzs, xy_margin_m, z_margin_m)
        nonlocal attempt_no_progress_count, last_lock_callback_track_id
        if last_lock_callback_track_id is None or int(last_lock_callback_track_id) != int(track_id):
            attempt_no_progress_count = 0
        last_lock_callback_track_id = int(track_id)
        track_id_only_lock = bool(track_id_only_lock_enabled)
        selected_u = int(selected_row.get("u", 0))
        selected_v = int(selected_row.get("v", 0))
        img_cx = int(obs.image_center_uv[0]) if isinstance(getattr(obs, "image_center_uv", None), (list, tuple, np.ndarray)) and len(obs.image_center_uv) >= 1 else int(selected_u)
        side_margin_px = max(0, int(PLACE_VERIFY_V2_SECTION_PIXEL_MARGIN_PX))
        # Side / height / pick-space gates use post-centering lock samples only (median_xyz),
        # not a one-frame projected row at lock UV (can mis-assign side vs visual lock).
        measure = collect_track_measurement(
            track_id=int(track_id),
            first_obs=obs,
            first_candidate=selected_row,
            sample_count_override=int(measurement_samples_i),
        )
        hits = int(measure.get("hits", 0))
        selected_xyz = measure.get("median_xyz")
        selected_arr = np.array(
            selected_xyz if isinstance(selected_xyz, (list, tuple, np.ndarray)) else [np.nan, np.nan, np.nan],
            dtype=float,
        ).reshape(-1)
        xyz_ok = bool(selected_arr.size >= 3 and np.all(np.isfinite(selected_arr[:3])))
        median_xyz_row = (
            None
            if not bool(xyz_ok)
            else [float(selected_arr[0]), float(selected_arr[1]), float(selected_arr[2])]
        )

        if bool(xyz_ok) and float(selected_arr[1]) <= float(correction_pick_min_y_m):
            if bool(is_pick_placed_target):
                _emit_pick_misplaced_lock_verify(
                    track_id=int(track_id),
                    median_xyz_row=median_xyz_row,
                    section_tag="fail",
                    height_tag="fail",
                    decision="reject",
                    reason="candidate_pick_space_rejected",
                )
            return {
                "decision": "reject",
                "reason": (
                    f"candidate_pick_space_rejected y={float(selected_arr[1]):.3f}<=correction_min={float(correction_pick_min_y_m):.3f}"
                ),
                "blocked_xyz": [float(selected_arr[0]), float(selected_arr[1]), float(selected_arr[2])],
                "selected_xyz": [float(selected_arr[0]), float(selected_arr[1]), float(selected_arr[2])],
            }
        if (not xyz_ok) or (hits < int(required_hits_i)):
            qcount = int(quality_wait_by_track.get(int(track_id), 0)) + 1
            quality_wait_by_track[int(track_id)] = int(qcount)
            if int(qcount) >= int(MISPLACED_PICK_QUALITY_WAIT_PER_TRACK_CAP):
                return {
                    "decision": "reject",
                    "reason": (
                        f"track_quality_stuck track_id={int(track_id)} "
                        f"count={int(qcount)}/{int(MISPLACED_PICK_QUALITY_WAIT_PER_TRACK_CAP)} "
                        f"hits={hits}/{int(required_hits_i)} xyz_ok={bool(xyz_ok)}"
                    ),
                    "selected_xyz": (
                        None
                        if (not xyz_ok)
                        else [float(selected_arr[0]), float(selected_arr[1]), float(selected_arr[2])]
                    ),
                }
            attempt_no_progress_count += 1
            if int(attempt_no_progress_count) >= int(MISPLACED_PICK_ATTEMPT_NO_PROGRESS_CAP):
                return {
                    "decision": "reject",
                    "reason": (
                        f"attempt_no_progress quality_gate_wait "
                        f"count={int(attempt_no_progress_count)}/{int(MISPLACED_PICK_ATTEMPT_NO_PROGRESS_CAP)}"
                    ),
                    "terminal": True,
                    "exit_reason": "attempt_no_progress",
                }
            return {
                "decision": "continue",
                "reason": f"quality_gate_wait hits={hits}/{int(required_hits_i)} xyz_ok={bool(xyz_ok)}",
                "selected_xyz": (
                    None
                    if (not xyz_ok)
                    else [float(selected_arr[0]), float(selected_arr[1]), float(selected_arr[2])]
                ),
            }

        inferred_norm = str(target_section_norm)
        inferred_reason = "track_id_only"
        section_verify_tag = "ok"
        if not bool(track_id_only_lock):
            inferred_norm, inferred_reason = _infer_section_for_misplaced_lock(
                float(selected_arr[0]),
                float(selected_arr[1]),
            )
            if (
                (not bool(is_pick_placed_target))
                and (not inferred_norm)
                and str(inferred_reason) == "too_far_from_section_center_xy"
            ):
                # pick_misplaced only: pixel rescue when median misses center-radius.
                if (
                    str(target_section_norm) == str(SECTION_LEFT_NAME).strip().lower()
                    and int(selected_u) <= int(img_cx + side_margin_px)
                ):
                    inferred_norm = str(SECTION_LEFT_NAME).strip().lower()
                    inferred_reason = "too_far_from_section_center_xy+pixel_fallback"
                elif (
                    str(target_section_norm) == str(SECTION_RIGHT_NAME).strip().lower()
                    and int(selected_u) >= int(img_cx - side_margin_px)
                ):
                    inferred_norm = str(SECTION_RIGHT_NAME).strip().lower()
                    inferred_reason = "too_far_from_section_center_xy+pixel_fallback"
            if inferred_norm != str(target_section_norm):
                section_verify_tag = "fail"
                if bool(is_pick_placed_target):
                    _emit_pick_misplaced_lock_verify(
                        track_id=int(track_id),
                        median_xyz_row=median_xyz_row,
                        section_tag=str(section_verify_tag),
                        height_tag="pending",
                        decision="reject",
                        reason=f"wrong_section got={inferred_norm or 'unknown'} via={inferred_reason}",
                    )
                return _mismatch_probe_or_terminal_reject(
                    tag="wrong_section",
                    track_id_for_strike=int(track_id),
                    reason_detail=f"got={inferred_norm or 'unknown'} via={inferred_reason}",
                    selected_xyz_row=median_xyz_row,
                )

        height_gate = _build_misplaced_pick_height_gate(
            obs=obs,
            selected_xyz=[float(selected_arr[0]), float(selected_arr[1]), float(selected_arr[2])],
            target_section_norm=str(target_section_norm),
            infer_section_for_xyz=_infer_section_for_misplaced_lock,
            correction_pick_min_y_m=float(correction_pick_min_y_m),
            expected_level=int(expected_target_level),
        )
        if not bool(height_gate.get("valid", False)):
            reason_tag = str(height_gate.get("reason", "height_out_of_band") or "height_out_of_band")
            detail = (
                f"selected_z={float(height_gate.get('selected_z_m', float('nan'))):.3f} "
                f"selected_level={height_gate.get('selected_level', None)} "
                f"nearest_z={height_gate.get('selected_nearest_z_m', None)} "
                f"err={height_gate.get('selected_error_m', None)} "
                f"tol={height_gate.get('height_tol_m', None)} "
                f"expected_level={height_gate.get('expected_state_level', None)} "
                f"state_delta={height_gate.get('state_level_delta', None)} "
                f"state_action={height_gate.get('state_reconcile_action', None)} "
                f"height_gate={height_gate}"
            )
            if bool(is_pick_placed_target):
                _emit_pick_misplaced_lock_verify(
                    track_id=int(track_id),
                    median_xyz_row=median_xyz_row,
                    section_tag=str(section_verify_tag),
                    height_tag="fail",
                    decision="reject",
                    reason=str(reason_tag),
                )
            return _mismatch_probe_or_terminal_reject(
                tag=str(reason_tag),
                track_id_for_strike=int(track_id),
                reason_detail=str(detail),
                selected_xyz_row=median_xyz_row,
            )

        attempt_no_progress_count = 0
        quality_wait_by_track[int(track_id)] = 0
        if bool(is_pick_placed_target):
            _emit_pick_misplaced_lock_verify(
                track_id=int(track_id),
                median_xyz_row=median_xyz_row,
                section_tag=str(section_verify_tag),
                height_tag="ok",
                decision="accept",
                reason="misplaced_side_target_locked",
            )
        return {
            "decision": "accept",
            "reason": "misplaced_side_target_locked",
            "selected_xyz": [float(selected_arr[0]), float(selected_arr[1]), float(selected_arr[2])],
            "accept_payload": {
                "hits": int(hits),
                "required_hits": int(required_hits_i),
                "section": str(inferred_norm),
                "selected_v": int(selected_row.get("v", 0)),
                "top_v_target_side": None,
                "height_gate": dict(height_gate),
            },
        }

    lock_attempt_rows: list[dict] = []
    seed_attempt_rows: list[dict] = []
    attempted_track_ids: list[int] = []
    failed_track_ids: list[int] = []
    attempt_exit_reasons: list[dict] = []
    lock_session: dict = {}
    seed_status = "global_top_two"
    seed_attempt_rows.append(
        {
            "status": str(seed_status),
            "target_section": str(target_section_norm),
            "required_track_id": None,
            "notes": "always_enter_handoff",
        }
    )
    context_base["misplaced_seed"] = dict(seed_attempt_rows[0])
    max_lock_attempts_i = max(1, int(MISPLACED_PICK_MAX_LOCK_ATTEMPTS))
    max_total_lock_time_s = float(
        max(float(primary_timeout_s), float(pick_placed_max_total_lock_time_s))
    )
    lock_stage_t0 = float(time.time())
    lock_loop_exit_reason = "attempts_exhausted"
    attempt_idx = 0
    sole_track_retries_used = 0
    correction_pick_min_y_m = 0.0
    while int(attempt_idx) < int(max_lock_attempts_i):
        lock_elapsed_s = float(time.time() - lock_stage_t0)
        remaining_budget_s = float(max_total_lock_time_s - lock_elapsed_s)
        if remaining_budget_s <= 0.0:
            lock_loop_exit_reason = "lock_budget_exhausted"
            break
        attempt_idx += 1
        swap_used_now = bool(attempt_idx > 1)
        for key in list(mismatch_strikes.keys()):
            mismatch_strikes[key] = 0
        mismatch_track_strikes.clear()
        quality_wait_by_track.clear()
        attempt_no_progress_count = 0
        blocked_for_attempt = set(int(tid) for tid in list(failed_track_ids))
        blocked_extra = None
        if bool(swap_used_now) and bool(blocked_for_attempt):
            blocked_extra = set(int(tid) for tid in list(blocked_for_attempt))
        attempt_timeout_s = float(max(0.5, min(float(primary_timeout_s), float(remaining_budget_s))))
        print(
            f"[PickMisplacedTop2] side={target_section_norm} "
            f"attempt={int(attempt_idx)}/{int(max_lock_attempts_i)} "
            f"timeout_s={attempt_timeout_s:.2f} budget_left_s={remaining_budget_s:.2f} "
            f"required_track_id=None swap_used={bool(swap_used_now)} "
            f"track_id_only_lock={bool(track_id_only_lock_enabled)} "
            f"pick_space_gate={bool(MISPLACED_PICK_ENFORCE_PICK_SPACE_GATE)} "
            f"pick_min_y={float(correction_pick_min_y_m):+.3f} "
            f"blocked_track_ids={sorted([int(tid) for tid in list(blocked_for_attempt)])}"
        )
        lock_session = run_track_handoff_session(
            state=state,
            arm=arm,
            per=per,
            det=det,
            reject_cap=max(1, int(PICK_OTHER_MAX_REJECTS)),
            hard_timeout_s=float(attempt_timeout_s),
            xy_margin_m=float(PLACE_VERIFY_V2_XY_MARGIN_M),
            z_margin_m=float(PLACE_VERIFY_V2_Z_MARGIN_M),
            blocked_track_id=None,
            blocked_xyz=None,
            blocked_track_ids_extra=blocked_extra,
            blocked_xyzs_extra=None,
            blocked_uv=None,
            status_prefix=f"pick_misplaced_a{int(attempt_idx)}",
            log_prefix=f"[PickMisplaced#{int(attempt_idx)}]",
            commit_conf_override=float(MISPLACED_PICK_LOCK_COMMIT_CONF),
            post_lock_refresh_s=float(max(0.0, MISPLACED_PICK_POST_CENTER_REFRESH_S)),
            post_lock_refresh_min_frames=int(max(0, MISPLACED_PICK_POST_CENTER_REFRESH_MIN_FRAMES)),
            on_locked_candidate=_lock_requested_side_candidate,
            required_track_id=None,
            max_no_candidate_frames=int(MISPLACED_PICK_NO_VALID_DETECTIONS_FRAMES),
            terminal_callback_decision=True,
            wrong_section_soft_skip=False,
            preferred_section_name=str(target_section_norm),
            preferred_section_centers_xy=dict(section_centers_xy),
            enforce_preferred_section_hard_filter=bool(
                PICK_PLACED_HANDOFF_SECTION_HARD_FILTER
            ),
            reject_below_base_y_m=(
                float(correction_pick_min_y_m)
                if bool(MISPLACED_PICK_ENFORCE_PICK_SPACE_GATE)
                else None
            ),
        )
        lock_status = str(lock_session.get("status", "observe_retry"))
        lock_exit_reason = str(lock_session.get("exit_reason", "observe_retry"))
        selected_track_id_i = (
            None
            if lock_session.get("selected_track_id", None) is None
            else int(lock_session.get("selected_track_id"))
        )
        last_candidate_track_id_i = (
            None
            if lock_session.get("last_candidate_track_id", None) is None
            else int(lock_session.get("last_candidate_track_id"))
        )
        attempt_track_id_i = (
            int(selected_track_id_i)
            if selected_track_id_i is not None
            else (None if last_candidate_track_id_i is None else int(last_candidate_track_id_i))
        )
        if attempt_track_id_i is not None:
            attempted_track_ids.append(int(attempt_track_id_i))
        attempt_row = {
            "attempt_index": int(attempt_idx),
            "swap_used": bool(swap_used_now),
            "seed_status": str(seed_status),
            "seed_track_id": None,
            "selected_track_id": selected_track_id_i,
            "candidate_track_id": last_candidate_track_id_i,
            "status": str(lock_status),
            "exit_reason": str(lock_exit_reason),
            "mismatch_counts": dict(mismatch_strikes),
            "mismatch_track_counts": _mismatch_track_counts_snapshot(),
            "reject_count": int(lock_session.get("reject_count", 0) or 0),
            "structural_reject_count": int(lock_session.get("structural_reject_count", 0) or 0),
            "no_candidate_frames": int(lock_session.get("no_candidate_frames", 0) or 0),
            "last_decision": str(lock_session.get("last_decision", "none")),
            "last_reason": str(lock_session.get("last_reason", "")),
            "last_side_pref_source": str(lock_session.get("last_side_pref_source", "none") or "none"),
        }
        lock_attempt_rows.append(dict(attempt_row))
        attempt_exit_reasons.append(
            {
                "attempt_index": int(attempt_idx),
                "track_id": attempt_track_id_i,
                "selected_track_id": selected_track_id_i,
                "status": str(lock_status),
                "exit_reason": str(lock_exit_reason),
            }
        )
        if str(lock_status) == "ok":
            lock_loop_exit_reason = "accepted"
            break
        if attempt_track_id_i is not None:
            failed_track_ids.append(int(attempt_track_id_i))
        retried_sole, sole_track_retries_used = _try_sole_visible_track_handoff_retry(
            state=state,
            arm=arm,
            det=det,
            per=per,
            failed_track_ids=failed_track_ids,
            sole_track_retries_used=int(sole_track_retries_used),
            max_sole_retries=int(PICK_PLACED_SOLE_TRACK_RETRIES),
            target_section_norm=str(target_section_norm),
            section_centers_xy=dict(section_centers_xy),
            enforce_hard_filter=bool(PICK_PLACED_HANDOFF_SECTION_HARD_FILTER),
            min_conf=float(MISPLACED_PICK_LOCK_COMMIT_CONF),
        )
        if bool(retried_sole):
            attempt_idx -= 1
            continue
    if str(lock_loop_exit_reason) != "accepted":
        if str(lock_loop_exit_reason) == "lock_budget_exhausted":
            if not isinstance(lock_session, dict):
                lock_session = {}
            lock_session["status"] = str(lock_session.get("status", "observe_retry") or "observe_retry")
            lock_session["exit_reason"] = "lock_budget_exhausted"
            lock_session["last_decision"] = str(lock_session.get("last_decision", "none") or "none")
            lock_session["last_reason"] = str(lock_session.get("last_reason", "lock_budget_exhausted") or "lock_budget_exhausted")
        elif int(attempt_idx) >= int(max_lock_attempts_i) and str(lock_session.get("status", "")) != "ok":
            lock_session["status"] = str(lock_session.get("status", "observe_retry") or "observe_retry")
            lock_session["exit_reason"] = "attempts_exhausted"
            lock_session["last_decision"] = str(lock_session.get("last_decision", "none") or "none")
            lock_session["last_reason"] = str(lock_session.get("last_reason", "attempts_exhausted") or "attempts_exhausted")

    # Keep order stable while deduplicating.
    attempted_track_ids = list(dict.fromkeys([int(tid) for tid in list(attempted_track_ids)]))
    failed_track_ids = list(dict.fromkeys([int(tid) for tid in list(failed_track_ids)]))
    context_base["misplaced_seed_attempts"] = list(seed_attempt_rows)
    context_base["lock_attempt_details"] = list(lock_attempt_rows)
    context_base["attempted_track_ids"] = list(attempted_track_ids)
    context_base["failed_track_ids"] = list(failed_track_ids)
    context_base["attempt_exit_reasons"] = list(attempt_exit_reasons)
    context_base["candidate_order"] = list(attempted_track_ids)
    context_base["swap_used"] = bool(len(lock_attempt_rows) > 1)
    lock_elapsed_total_s = float(max(0.0, time.time() - lock_stage_t0))

    lock_info = {
        "status": str(lock_session.get("status", "observe_retry")),
        "hits": int((lock_session.get("accept_payload") or {}).get("hits", 0)),
        "required_hits": int(required_hits_i),
        "section": str((lock_session.get("accept_payload") or {}).get("section", target_section_norm)),
        "selected_v": (lock_session.get("accept_payload") or {}).get("selected_v", None),
        "top_v_target_side": (lock_session.get("accept_payload") or {}).get("top_v_target_side", None),
        "reject_count": int(lock_session.get("reject_count", 0)),
        "reject_cap": int(lock_session.get("reject_cap", max(1, int(PICK_OTHER_MAX_REJECTS)))),
        "lock_attempts": int(len(lock_attempt_rows)),
        "max_lock_attempts": int(max_lock_attempts_i),
        "swap_used": bool(context_base.get("swap_used", False)),
        "exit_reason": str(lock_session.get("exit_reason", "observe_retry")),
        "last_decision": str(lock_session.get("last_decision", "none")),
        "last_reason": str(lock_session.get("last_reason", "")),
        "last_candidate_track_id": lock_session.get("last_candidate_track_id", None),
        "last_side_pref_source": str(lock_session.get("last_side_pref_source", "none") or "none"),
        "no_candidate_frames": int(lock_session.get("no_candidate_frames", 0) or 0),
        "structural_reject_count": int(lock_session.get("structural_reject_count", 0) or 0),
        "primary_timeout_s": float(primary_timeout_s),
        "max_total_lock_time_s": float(max_total_lock_time_s),
        "attempts_used": int(len(lock_attempt_rows)),
        "lock_elapsed_s": float(lock_elapsed_total_s),
        "lock_loop_exit_reason": str(lock_loop_exit_reason),
        "required_track_id": (
            None
            if not attempted_track_ids
            else int(attempted_track_ids[0])
        ),
        "attempted_track_ids": [int(tid) for tid in list(attempted_track_ids)],
        "failed_track_ids": [int(tid) for tid in list(failed_track_ids)],
        "attempt_exit_reasons": list(attempt_exit_reasons),
        "lock_attempt_details": list(lock_attempt_rows),
        "seed_attempts": list(seed_attempt_rows),
        "height_gate": (
            (lock_session.get("accept_payload") or {}).get("height_gate", None)
            if isinstance(lock_session, dict)
            else None
        ),
    }
    if not isinstance(lock_info.get("height_gate", None), dict):
        lock_info["height_gate"] = {}
    if str(lock_session.get("status", "")) != "ok":
        exit_reason = str(lock_session.get("exit_reason", "observe_retry")).strip().lower()
        timeout_like_exit = exit_reason in {"timeout_or_uncertain", "dependencies_missing"}
        no_lock_evidence = (
            int(lock_session.get("reject_count", 0) or 0) <= 0
            and lock_session.get("selected_track_id", None) is None
        )
        common_ctx = _ctx(
            {
                "misplaced_lock": dict(lock_info),
                "verify_status": str(lock_session.get("status", "observe_retry")),
                "reconcile_status": str(target_reconcile.get("status", "unknown")),
                "reconcile_scene_revision": int(target_reconcile.get("scene_revision", state.scene_revision)),
            }
        )
        if exit_reason in {
            "target_side_no_valid_detections_5f",
            "no_track_candidate_frames_exceeded",
            "no_track_candidate_timeout",
            "no_observation",
        }:
            return False, "reacquire_failed_no_valid_detections_5f", common_ctx
        if exit_reason == "attempts_exhausted":
            return False, "reacquire_failed_attempts_exhausted", common_ctx
        if exit_reason == "lock_budget_exhausted":
            return False, "reacquire_failed_lock_budget_exhausted", common_ctx
        if exit_reason == "attempt_no_progress":
            return False, "reacquire_failed_attempt_no_progress", common_ctx
        if exit_reason == "track_quality_stuck":
            return False, "reacquire_failed_track_quality_stuck", common_ctx
        if exit_reason.startswith("mismatch_terminal:"):
            if int(len(lock_attempt_rows)) < int(max_lock_attempts_i) and float(lock_elapsed_total_s) < float(max_total_lock_time_s):
                return False, "reacquire_failed_mismatch_terminal", common_ctx
            if float(lock_elapsed_total_s) >= float(max_total_lock_time_s):
                return False, "reacquire_failed_lock_budget_exhausted", common_ctx
            return False, "reacquire_failed_attempts_exhausted", common_ctx
        if timeout_like_exit and no_lock_evidence:
            return False, "reacquire_failed_transient", common_ctx
        if timeout_like_exit and (not no_lock_evidence):
            return False, "reacquire_failed_timeout_with_evidence", common_ctx
        return False, "reacquire_failed_explicit_exit", common_ctx

    lock_uv = lock_session.get("centered_pos", None)
    lock_track_id = lock_session.get("selected_track_id", None)
    lock_xyz = lock_session.get("selected_xyz", None)
    context_base["lock_track_id"] = (None if lock_track_id is None else int(lock_track_id))
    context_base["lock_uv"] = (
        None
        if not isinstance(lock_uv, (list, tuple)) or len(lock_uv) < 2
        else [int(lock_uv[0]), int(lock_uv[1])]
    )
    context_base["lock_xyz"] = _finite_xyz_or_none(lock_xyz)
    if not isinstance(lock_uv, (list, tuple)) or len(lock_uv) < 2:
        return False, "reacquire_missing_uv", _ctx({"misplaced_lock": dict(lock_info)})
    centered_pos = (int(lock_uv[0]), int(lock_uv[1]))
    state.active_target_track_id = (None if lock_track_id is None else int(lock_track_id))
    drop_level_raw = int(max(0, int(getattr(state, "misplaced_drop_count", 0))))
    drop_plan = _misplaced_return_grid_slot(int(drop_level_raw))
    context_base["drop_level"] = int(drop_level_raw)
    context_base["drop_level_raw"] = int(drop_level_raw)
    context_base["drop_grid"] = dict(drop_plan)
    context_base["drop_xyz"] = _finite_xyz_or_none(drop_plan.get("drop_xyz", None))
    if not bool(drop_plan.get("ok", False)):
        return False, str(drop_plan.get("reason", "misplaced_return_slot_invalid")), _ctx(
            {
                "drop_grid": dict(drop_plan),
                "misplaced_lock": dict(lock_info),
            }
        )
    drop_xyz = np.array(drop_plan["drop_xyz"], dtype=float).reshape(-1)
    print(
        f"[PickMisplacedDrop] raw={int(drop_plan['raw'])} "
        f"row={int(drop_plan['row'])} col={int(drop_plan['col'])} "
        f"max_slots={int(drop_plan['max_slots'])} "
        f"base=({float(drop_plan['base_xyz'][0]):.3f},{float(drop_plan['base_xyz'][1]):.3f},{float(drop_plan['base_xyz'][2]):.3f}) "
        f"grid_step=(dx={float(drop_plan['grid_dx_m']):+.3f},dy={float(drop_plan['grid_dy_m']):+.3f}) "
        f"target=({float(drop_xyz[0]):.3f},{float(drop_xyz[1]):.3f},{float(drop_xyz[2]):.3f}) "
        f"reach={float(drop_plan['reach_m']):.3f}"
    )
    misplaced_grip_step_override = None
    base_grip_step = float(GRIP_CURRENT_LIMITS.grip_step)
    if bool(GRASP_STACK_FORWARD_ENABLE):
        misplaced_grip_step_override = float(base_grip_step * 1.40)
    context_base["grip_step_base"] = float(base_grip_step)
    context_base["grip_step_override"] = (
        None if misplaced_grip_step_override is None else float(misplaced_grip_step_override)
    )
    print(
        f"[PickMisplacedGripStep] enabled={bool(misplaced_grip_step_override is not None)} "
        f"base={float(base_grip_step):.4f} "
        f"override={float(misplaced_grip_step_override) if misplaced_grip_step_override is not None else float(base_grip_step):.4f} "
        f"mult=1.40 trigger=GRASP_STACK_FORWARD_ENABLE "
        f"trigger_enabled={bool(GRASP_STACK_FORWARD_ENABLE)}"
    )
    height_gate_for_offset = lock_info.get("height_gate", {}) if isinstance(lock_info, dict) else {}
    try:
        correction_level_for_offset = int(height_gate_for_offset.get("selected_level", 0) or 0)
    except Exception:
        correction_level_for_offset = 0
    correction_level_for_offset = max(0, min(int(MAX_STACK_LEVELS_PER_SECTION), int(correction_level_for_offset)))
    x_level_steps = max(0, int(correction_level_for_offset) - 1)
    x_dynamic = float(x_level_steps) * float(PICK_MISPLACED_GRASP_X_PER_LEVEL_M)
    x_cap_abs = max(0.0, float(PICK_MISPLACED_GRASP_X_MAX_ABS_M))
    x_dynamic = float(max(-float(x_cap_abs), min(float(x_cap_abs), float(x_dynamic))))
    correction_extra_x = float(PICK_MISPLACED_GRASP_X_OFFSET_M) + float(x_dynamic)
    z_high_extra = float(PICK_MISPLACED_GRASP_HIGH_Z_EXTRA_M) if int(correction_level_for_offset) > 0 else 0.0
    correction_extra_z = float(PICK_MISPLACED_GRASP_Z_OFFSET_M) + float(z_high_extra)
    y_dynamic = float(correction_level_for_offset) * float(PICK_MISPLACED_GRASP_Y_PER_LEVEL_M)
    y_cap = max(0.0, float(PICK_MISPLACED_GRASP_Y_MAX_M))
    y_dynamic = float(max(0.0, min(float(y_cap), float(y_dynamic))))
    correction_extra_y = float(PICK_MISPLACED_GRASP_Y_OFFSET_M) + float(y_dynamic)
    context_base["correction_grasp_extra_x_m"] = float(correction_extra_x)
    context_base["correction_grasp_x_level_delta_m"] = float(x_dynamic)
    context_base["correction_grasp_x_level"] = int(correction_level_for_offset)
    context_base["correction_grasp_x_level_steps"] = int(x_level_steps)
    context_base["correction_grasp_extra_y_m"] = float(correction_extra_y)
    context_base["correction_grasp_y_level"] = int(correction_level_for_offset)
    context_base["correction_grasp_extra_z_m"] = float(correction_extra_z)
    print(
        f"[PickMisplacedGraspOffset] extra_x={float(correction_extra_x):+.3f} "
        f"(base={float(PICK_MISPLACED_GRASP_X_OFFSET_M):+.3f}, "
        f"per_level={float(PICK_MISPLACED_GRASP_X_PER_LEVEL_M):+.3f}, "
        f"level={int(correction_level_for_offset)}, steps={int(x_level_steps)}, "
        f"cap_abs={float(x_cap_abs):.3f}) "
        f"extra_y={float(correction_extra_y):+.3f} "
        f"(base={float(PICK_MISPLACED_GRASP_Y_OFFSET_M):+.3f}, "
        f"per_level={float(PICK_MISPLACED_GRASP_Y_PER_LEVEL_M):+.3f}, "
        f"level={int(correction_level_for_offset)}, cap={float(y_cap):+.3f}) "
        f"extra_z={float(correction_extra_z):+.3f} "
        f"(base={float(PICK_MISPLACED_GRASP_Z_OFFSET_M):+.3f}, "
        f"high_extra={float(z_high_extra):+.3f})"
    )

    carry_status, hold_grip, carry_supervisor = run_grasp_and_carry_common(
        state=state,
        arm=arm,
        per=per,
        centered_pos=centered_pos,
        label_prefix=label_prefix,
        safe_pick_reach_m=safe_pick_reach_m,
        correction_abort_vertical_retreat=True,
        grip_step_override=misplaced_grip_step_override,
        extra_x_offset_m=float(correction_extra_x),
        extra_y_offset_m=float(correction_extra_y),
        extra_z_offset_m=float(correction_extra_z),
    )
    if carry_status != "ok":
        return False, f"grasp_{carry_status}", _ctx({"misplaced_lock": dict(lock_info)})
    drop_reconcile = reconcile_scene(
        state=state,
        arm=arm,
        per=per,
        det=det,
        side="all",
        mode="pre_pick_misplaced_drop",
        target_xyz=drop_xyz,
        include_pick_rows=True,
    )
    if bool(drop_reconcile.get("collision_risk", False)):
        return False, "misplaced_drop_occupied", _ctx(
            {
                "drop_reconcile": drop_reconcile,
                "misplaced_lock": dict(lock_info),
                "released_after_failure": False,
                "holding_after_failure": bool(getattr(state, "holding_object", False)),
                "return_outcome": "holding_return_unavailable",
            }
        )
    if bool(CORRECTION_DROP_TRANSIT_ENABLED):
        if not goto_correction_drop_transit(
            arm,
            float(hold_grip),
            carry_supervisor,
            str(label_prefix),
        ):
            transit_reason = str(getattr(arm, "last_motion_reason", "") or "correction_drop_transit_failed")
            if transit_reason == "move_overcurrent_unrecoverable":
                state.stop_reason = transit_reason
                state.skip_final_motion = True
            return False, f"correction_drop_transit_{transit_reason}", _ctx(
                {"misplaced_lock": dict(lock_info)}
            )
    place_ok, place_reason = safe_place(
        arm=arm,
        slot_index=-1,
        grip=float(hold_grip),
        placed_targets=state.placed_targets,
        blocked_slots=state.blocked_slots,
        det=det,
        per=per,
        motion_supervisor=carry_supervisor,
        custom_target_xyz=drop_xyz,
        allow_stacked_target=True,
    )
    return_place_failure = None
    return_outcome = "returned"
    if not bool(place_ok):
        return_place_failure = _classify_misplaced_return_place_failure(place_reason, arm)
        if not bool(return_place_failure.get("released", False)):
            return False, f"misplaced_return_place_failed:{place_reason}", _ctx(
                {
                    "drop_reconcile": drop_reconcile,
                    "misplaced_lock": dict(lock_info),
                    "return_place_failure": dict(return_place_failure),
                    "released_after_failure": False,
                    "holding_after_failure": bool(getattr(state, "holding_object", False)),
                    "return_outcome": "holding_return_unavailable",
                }
            )
        return_outcome = "released_after_place_soft_fail"
        print(
            f"[PickMisplacedReturnSoftSuccess] reason={str(place_reason)} "
            f"label={return_place_failure.get('motion_label', '') or 'unknown'} "
            f"drop_xyz={_finite_xyz_or_none(drop_xyz)}"
        )

    correction_retreat = retreat_after_correction_drop(
        arm,
        str(label_prefix),
        drop_xyz=drop_xyz,
    )
    state.holding_object = False
    state.current_hold_grip = 0.0
    state.last_pick_return_xyz = None
    state.last_pick_measured_xyz = None
    state.returned_count += 1
    state.misplaced_drop_count = int(max(0, int(getattr(state, "misplaced_drop_count", 0))) + 1)
    state.pick_other_block_track_id = None
    state.pick_other_block_uv = None
    state.pick_other_block_xyz = None
    state.pick_other_block_track_ids = []
    state.pick_other_block_xyzs = []
    state.pick_other_block_uvs = []
    state.pick_other_block_source = "none"
    state_height_reconcile = None
    measured_lock_level = None
    if isinstance(lock_info.get("height_gate", None), dict):
        measured_lock_level = lock_info["height_gate"].get("selected_level", None)
    if target_section_norm in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        state_height_reconcile = _reconcile_authoritative_stack_level_to_measured(
            state,
            target_section_norm,
            measured_lock_level,
        )
    authoritative_pop = None
    if target_section_norm in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        authoritative_pop = pop_authoritative_stack_top(
            state,
            target_section_norm,
            removed_xyz=context_base.get("target_xyz", None),
        )

    return True, (
        "misplaced_return_released_after_place_soft_fail"
        if return_place_failure is not None
        else "ok"
    ), {
        "target_section": context_base.get("target_section"),
        "target_xyz": context_base.get("target_xyz"),
        "drop_level": context_base.get("drop_level"),
        "drop_level_raw": context_base.get("drop_level_raw"),
        "drop_grid": context_base.get("drop_grid"),
        "drop_xyz": context_base.get("drop_xyz"),
        "lock_track_id": context_base.get("lock_track_id"),
        "lock_uv": context_base.get("lock_uv"),
        "lock_xyz": context_base.get("lock_xyz"),
        "preferred_section": context_base.get("preferred_section"),
        "candidate_sides": context_base.get("candidate_sides"),
        "target_reconcile": target_reconcile,
        "drop_reconcile": drop_reconcile,
        "misplaced_lock": dict(lock_info),
        "return_outcome": str(return_outcome),
        "return_place_failure": (
            None if return_place_failure is None else dict(return_place_failure)
        ),
        "released_after_failure": bool(return_place_failure is not None),
        "holding_after_failure": False,
        "state_height_reconcile": state_height_reconcile,
        "authoritative_pop": authoritative_pop,
        "correction_retreat": dict(correction_retreat),
    }

def execute_return_placed_cube_correction(
    *,
    state: CycleState,
    arm: Arm,
    det: YOLODetector,
    per: Perception,
    stack_levels: dict[str, int],
    label_prefix: str,
    safe_pick_reach_m: float | None = None,
) -> tuple[bool, str, dict | None]:
    _bind_core_globals()
    target_entry = get_latest_confirmed_active_stack_placement(state)
    if target_entry is None:
        return False, "no_confirmed_active_placement", None

    target_object_id = int(target_entry.get("object_id", -1))
    target_section = str(target_entry.get("section", "")).strip().lower()
    pick_origin_xyz = _finite_xyz_or_none(target_entry.get("pick_origin_xyz"))
    if pick_origin_xyz is None:
        return False, "missing_pick_origin_xyz", {"object_id": target_object_id}

    verify_row = target_entry.get("verify_result", None)
    target_xyz = None
    if isinstance(verify_row, dict):
        target_xyz = _finite_xyz_or_none(verify_row.get("measured_xyz"))
    if target_xyz is None:
        target_xyz = _finite_xyz_or_none(target_entry.get("expected_xyz"))
    if target_xyz is None:
        return False, "missing_target_xyz", {"object_id": target_object_id}

    target_side = target_section if target_section in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME} else "all"
    target_reconcile = reconcile_scene(
        state=state,
        arm=arm,
        per=per,
        det=det,
        side=str(target_side),
        mode="pre_return_placed_target",
        target_xyz=target_xyz,
        include_pick_rows=False,
    )
    target_min_dxy = float(target_reconcile.get("collision_min_dxy_m", float("inf")))
    target_min_dz = float(target_reconcile.get("collision_min_dz_m", float("inf")))
    target_visible = bool(
        np.isfinite(target_min_dxy)
        and np.isfinite(target_min_dz)
        and target_min_dxy <= (1.25 * float(PLACE_VERIFY_V2_XY_MARGIN_M))
        and target_min_dz <= (1.25 * float(PLACE_VERIFY_V2_Z_MARGIN_M))
    )
    if not target_visible:
        return False, "target_not_visible_in_reconcile", {
            "object_id": target_object_id,
            "target_section": target_section,
            "target_xyz": target_xyz,
            "pick_origin_xyz": pick_origin_xyz,
            "target_reconcile": target_reconcile,
        }

    verify_stage = run_return_verify_stage(
        state=state,
        arm=arm,
        per=per,
        det=det,
        label_prefix=label_prefix,
        return_target_xyz=target_xyz,
        required_hits=8,
        measurement_samples=max(int(PICK_OTHER_VALIDATE_SAMPLES), 8),
        reject_cap=max(1, int(PICK_OTHER_MAX_REJECTS)),
        hard_timeout_s=max(1.0, float(PICK_OTHER_HARD_TIMEOUT_S)),
        verify_look_pose=PLACE_LOOKING,
    )
    return_verify = dict(verify_stage.get("return_verify", {}))
    if not bool(verify_stage.get("returned_verified", False)):
        return False, "reacquire_failed", {
            "object_id": target_object_id,
            "target_section": target_section,
            "target_xyz": target_xyz,
            "pick_origin_xyz": pick_origin_xyz,
            "return_verify": return_verify,
            "verify_status": str(verify_stage.get("status", "observe_retry")),
        }

    lock_uv = verify_stage.get("blocked_return_uv", None)
    lock_track_id = verify_stage.get("blocked_return_track_id", None)
    lock_xyz = verify_stage.get("blocked_return_xyz", None)
    if not isinstance(lock_uv, (list, tuple)) or len(lock_uv) < 2:
        return False, "reacquire_missing_uv", {
            "object_id": target_object_id,
            "target_section": target_section,
            "target_xyz": target_xyz,
            "pick_origin_xyz": pick_origin_xyz,
            "return_verify": return_verify,
        }
    centered_pos = (int(lock_uv[0]), int(lock_uv[1]))
    state.active_target_track_id = (None if lock_track_id is None else int(lock_track_id))

    carry_status, hold_grip, carry_supervisor = run_grasp_and_carry_common(
        state=state,
        arm=arm,
        per=per,
        centered_pos=centered_pos,
        label_prefix=label_prefix,
        safe_pick_reach_m=safe_pick_reach_m,
    )
    if carry_status != "ok":
        return False, f"grasp_{carry_status}", {
            "object_id": target_object_id,
            "target_section": target_section,
            "target_xyz": target_xyz,
            "pick_origin_xyz": pick_origin_xyz,
            "lock_track_id": (None if lock_track_id is None else int(lock_track_id)),
            "lock_uv": [int(centered_pos[0]), int(centered_pos[1])],
            "lock_xyz": _finite_xyz_or_none(lock_xyz),
            "return_verify": return_verify,
        }

    drop_xyz = np.array([float(pick_origin_xyz[0]), float(pick_origin_xyz[1]), float(pick_origin_xyz[2])], dtype=float).reshape(-1)
    drop_xyz[2] = max(float(drop_xyz[2]), float(PLACE_RELEASE_Z_M))
    drop_reconcile = reconcile_scene(
        state=state,
        arm=arm,
        per=per,
        det=det,
        side="all",
        mode="pre_return_placed_drop",
        target_xyz=drop_xyz,
        include_pick_rows=True,
    )
    if bool(drop_reconcile.get("collision_risk", False)):
        return False, "return_drop_occupied", {
            "object_id": target_object_id,
            "target_section": target_section,
            "target_xyz": target_xyz,
            "pick_origin_xyz": _finite_xyz_or_none(drop_xyz),
            "drop_reconcile": drop_reconcile,
            "return_verify": return_verify,
        }
    place_ok, place_reason = safe_place(
        arm=arm,
        slot_index=-1,
        grip=float(hold_grip),
        placed_targets=state.placed_targets,
        blocked_slots=state.blocked_slots,
        det=det,
        per=per,
        motion_supervisor=carry_supervisor,
        custom_target_xyz=drop_xyz,
        allow_stacked_target=True,
    )
    if not bool(place_ok):
        return False, f"return_place_failed:{place_reason}", {
            "object_id": target_object_id,
            "target_section": target_section,
            "target_xyz": target_xyz,
            "pick_origin_xyz": _finite_xyz_or_none(drop_xyz),
            "lock_track_id": (None if lock_track_id is None else int(lock_track_id)),
            "lock_uv": [int(centered_pos[0]), int(centered_pos[1])],
            "lock_xyz": _finite_xyz_or_none(lock_xyz),
            "drop_reconcile": drop_reconcile,
            "return_verify": return_verify,
        }

    target_entry["removed_by_return"] = True
    target_entry["removed_timestamp_ms"] = int(time.time() * 1000)
    target_entry["removed_reason"] = "mission_correction"
    target_entry["removed_command"] = "return_placed_cube"
    if isinstance(target_entry.get("verify_result", None), dict):
        target_entry["verify_result"]["corrected_by_return"] = True

    state.holding_object = False
    state.current_hold_grip = 0.0
    state.last_pick_return_xyz = None
    state.last_pick_measured_xyz = None
    state.returned_count += 1
    state.placed_count = max(0, int(state.placed_count) - 1)
    auth_pop = None
    if target_section in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        auth_pop = pop_authoritative_stack_top(
            state,
            target_section,
            removed_xyz=target_xyz,
        )
        state.placed_counts_by_section[target_section] = max(
            0, int(state.placed_counts_by_section.get(target_section, 0)) - 1
        )
        levels_auth = get_authoritative_stack_levels(state)
        stack_levels[target_section] = int(levels_auth.get(target_section, 0) or 0)
    _remove_nearest_placed_target(state.placed_targets, target_xyz)
    state.pick_other_block_track_id = None
    state.pick_other_block_uv = None
    state.pick_other_block_xyz = None
    state.pick_other_block_track_ids = []
    state.pick_other_block_xyzs = []
    state.pick_other_block_uvs = []
    state.pick_other_block_source = "none"

    return True, "ok", {
        "object_id": target_object_id,
        "target_section": target_section,
        "target_xyz": _finite_xyz_or_none(target_xyz),
        "pick_origin_xyz": _finite_xyz_or_none(drop_xyz),
        "lock_track_id": (None if lock_track_id is None else int(lock_track_id)),
        "lock_uv": [int(centered_pos[0]), int(centered_pos[1])],
        "lock_xyz": _finite_xyz_or_none(lock_xyz),
        "target_reconcile": target_reconcile,
        "drop_reconcile": drop_reconcile,
        "return_verify": return_verify,
        "removed": True,
        "authoritative_pop": auth_pop,
    }
