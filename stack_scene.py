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
    # Keep moved function bindings stable while resolving runtime constants/helpers lazily.
    protected = {
        '_bind_core_globals', '_CORE_BIND_READY',
        '_verify_section_y_centers', '_verify_section_xy_centers', '_place_space_y_band_bounds',
        '_infer_section_for_place_xy', '_infer_section_for_place_y', '_infer_section_for_base_y',
        '_resolve_verify_expected_color', 'build_place_grid_slots', 'get_place_slots',
        'section_slot_groups', 'next_slot_in_section', '_is_confirmed_active_placement',
        '_is_stack_confirmed_active_placement', 'get_latest_confirmed_active_stack_placement',
        '_get_locked_stack_anchor_xyz', '_set_locked_stack_anchor_xyz', '_clear_locked_stack_anchor_xyz',
        '_get_last_popped_xy', '_set_last_popped_xy', '_clear_last_popped_xy',
        'get_latest_side_stack_anchor_xyz', '_compact_section_truth_row', '_ledger_section_truth_row',
        'run_place_space_truth_pass', '_startup_default_hydrated_section_row',
        '_startup_hydrate_sides_at_cap', '_startup_hydrate_should_exit_when_sides_full',
        '_startup_side_full_rescan_top_to_bottom', '_startup_vote_hits_for_layer_slot',
        '_normalize_hydrated_section_row', '_merge_hydrated_section_row_keep_known',
        'run_startup_stack_identity_pass', 'apply_startup_stack_hydration',
        'get_startup_hydrated_section_row', '_sync_last_begin_hydrated_stacks',
        '_set_authoritative_section_sequence', 'get_authoritative_stack_levels',
        'append_authoritative_stack_cube', 'pop_authoritative_stack_top',
        '_scene_sections_for_side', '_row_color_for_reconcile', '_append_unique_reconcile_row',
        '_section_snapshot_signature', 'reconcile_scene',
        '_extract_valid_z', 'remeasure_stack_xyz_after_center',
        'remeasure_stack_xyz_until_stable', 'infer_stack_layers_from_measurement',
    }
    for name, value in core.__dict__.items():
        if name.startswith('__') or name in protected:
            continue
        globals()[name] = value
    _CORE_BIND_READY = True


def _format_xyz_log_3(xyz: list[float] | tuple[float, ...] | np.ndarray | None) -> str:
    if xyz is None:
        return "none"
    try:
        arr = np.array(xyz, dtype=float).reshape(-1)
    except Exception:
        return str(xyz)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return str(xyz)
    return f"[{float(arr[0]):.3f},{float(arr[1]):.3f},{float(arr[2]):.3f}]"


def _verify_section_y_centers() -> dict[str, float]:
    _bind_core_globals()
    slots = get_place_slots()
    groups = section_slot_groups(slots)
    centers: dict[str, float] = {}
    for section_name, indices in groups.items():
        ys = [float(slots[i][1]) for i in indices if 0 <= int(i) < len(slots)]
        if ys:
            centers[str(section_name).strip().lower()] = float(np.mean(ys))
    return centers


def _verify_section_xy_centers() -> dict[str, tuple[float, float]]:
    _bind_core_globals()
    slots = get_place_slots()
    groups = section_slot_groups(slots)
    centers: dict[str, tuple[float, float]] = {}
    for section_name, indices in groups.items():
        pts: list[tuple[float, float]] = []
        for i in indices:
            try:
                idx = int(i)
            except Exception:
                continue
            if idx < 0 or idx >= len(slots):
                continue
            slot = slots[idx]
            if slot.size < 2:
                continue
            x = float(slot[0])
            y = float(slot[1])
            if np.isfinite(x) and np.isfinite(y):
                pts.append((x, y))
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            centers[str(section_name).strip().lower()] = (float(np.mean(xs)), float(np.mean(ys)))
    return centers


def _place_space_y_band_bounds(margin_m: float | None = None) -> tuple[float, float]:
    _bind_core_globals()
    slots = get_place_slots()
    ys = [float(s[1]) for s in slots if isinstance(s, np.ndarray) and s.size >= 2 and np.isfinite(float(s[1]))]
    if not ys:
        return float("-inf"), float("inf")
    margin = float(max(0.0, SCENE_RECON_PLACE_Y_MARGIN_M if margin_m is None else margin_m))
    return float(min(ys) - margin), float(max(ys) + margin)


def _infer_section_for_place_xy(
    base_x: float,
    base_y: float,
    section_centers_xy: dict[str, tuple[float, float]],
    *,
    band_min: float | None = None,
    band_max: float | None = None,
    max_center_dist_m: float | None = None,
) -> tuple[str | None, dict]:
    _bind_core_globals()
    info = {
        "base_x": float(base_x) if np.isfinite(float(base_x)) else float("nan"),
        "base_y": float(base_y) if np.isfinite(float(base_y)) else float("nan"),
        "band_min": float(band_min) if band_min is not None else float("nan"),
        "band_max": float(band_max) if band_max is not None else float("nan"),
        "best_dist_m": float("inf"),
        "reason": "ok",
    }
    if (not np.isfinite(float(base_x))) or (not np.isfinite(float(base_y))) or (not section_centers_xy):
        info["reason"] = "invalid_or_missing_section_centers"
        return None, info
    if band_min is not None and float(base_y) < float(band_min):
        info["reason"] = "below_place_band"
        return None, info
    if band_max is not None and float(base_y) > float(band_max):
        info["reason"] = "above_place_band"
        return None, info

    best_name = None
    best_dist = float("inf")
    for name, center_xy in section_centers_xy.items():
        try:
            cx = float(center_xy[0])
            cy = float(center_xy[1])
        except Exception:
            continue
        d_xy = float(math.hypot(float(base_x) - cx, float(base_y) - cy))
        if d_xy < best_dist:
            best_dist = d_xy
            best_name = str(name)
    info["best_dist_m"] = float(best_dist)
    max_dist = float(
        max_center_dist_m if max_center_dist_m is not None else STARTUP_STACK_SECTION_XY_MARGIN_M
    )
    if best_name is None or best_dist > max_dist:
        info["reason"] = "too_far_from_section_center_xy"
        return None, info
    return best_name, info


def _infer_section_for_place_y(
    base_y: float,
    section_centers: dict[str, float],
    *,
    band_min: float | None = None,
    band_max: float | None = None,
    max_center_dist_m: float | None = None,
) -> tuple[str | None, dict]:
    _bind_core_globals()
    info = {
        "base_y": float(base_y) if np.isfinite(float(base_y)) else float("nan"),
        "band_min": float(band_min) if band_min is not None else float("nan"),
        "band_max": float(band_max) if band_max is not None else float("nan"),
        "best_dist_m": float("inf"),
        "reason": "ok",
    }
    if (not np.isfinite(float(base_y))) or (not section_centers):
        info["reason"] = "invalid_or_missing_section_centers"
        return None, info
    best_name = None
    best_dist = float("inf")
    for name, yc in section_centers.items():
        d = abs(float(base_y) - float(yc))
        if d < best_dist:
            best_dist = d
            best_name = str(name)
    info["best_dist_m"] = float(best_dist)
    if band_min is not None and float(base_y) < float(band_min):
        info["reason"] = "below_place_band"
        return None, info
    if band_max is not None and float(base_y) > float(band_max):
        info["reason"] = "above_place_band"
        return None, info
    max_dist = float(max_center_dist_m if max_center_dist_m is not None else SCENE_RECON_SECTION_MAX_DIST_M)
    if best_dist > max_dist:
        info["reason"] = "too_far_from_section_center"
        return None, info
    return best_name, info


def _infer_section_for_base_y(base_y: float, section_centers: dict[str, float]) -> str | None:
    _bind_core_globals()
    if not np.isfinite(float(base_y)) or not section_centers:
        return None
    best_name = None
    best_dist = float("inf")
    for name, yc in section_centers.items():
        d = abs(float(base_y) - float(yc))
        if d < best_dist:
            best_dist = d
            best_name = str(name)
    return best_name


def _resolve_verify_expected_color(
    placed_cube_color: str | None,
) -> tuple[str | None, str]:
    _bind_core_globals()
    color_name = str(placed_cube_color or "").strip().lower()
    if color_name in {"orange", "blue"}:
        return color_name, "placed_cube_color"
    # For generalized missions (e.g., alternating colors), avoid forcing a section-fixed color.
    return None, "none_unclassified"


def build_place_grid_slots(
    rows: int | None = None,
    cols: int | None = None,
    center_x: float | None = None,
    center_y: float | None = None,
    dx: float | None = None,
    dy: float | None = None,
):
    _bind_core_globals()
    if rows is None:
        rows = int(PLACE_GRID_ROWS)
    if cols is None:
        cols = int(PLACE_GRID_COLS)
    if center_x is None:
        center_x = float(PLACE_GRID_CENTER_X_M)
    if center_y is None:
        center_y = float(PLACE_GRID_CENTER_Y_M)
    if dx is None:
        dx = float(PLACE_GRID_DX_M)
    if dy is None:
        dy = float(PLACE_GRID_DY_M)
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    z = max(TABLE_Z_SAT_M, PLACE_RELEASE_Z_M)
    row_offsets = np.linspace(+dx, -dx, rows)
    col_offsets = np.linspace(-dy, +dy, cols)
    safe_slots: list[np.ndarray] = []
    for row_off in row_offsets:
        for col_off in col_offsets:
            x = float(center_x + row_off)
            y = float(center_y + col_off)
            target = np.array([x, y, z], dtype=float)
            reach = float(np.linalg.norm(target))
            if reach > MAX_REACH_M or reach < MIN_PLACE_REACH_M:
                continue
            clear_ok, _ = placement_clearance_ok(
                target_xyz=target,
                placed_targets=safe_slots,
                min_sep_m=MIN_PLACE_SLOT_SEPARATION_M,
            )
            if not clear_ok:
                continue
            safe_slots.append(target)
    return safe_slots
_PLACE_SLOT_CACHE: list[np.ndarray] | None = None
def get_place_slots() -> list[np.ndarray]:
    _bind_core_globals()
    global _PLACE_SLOT_CACHE
    if _PLACE_SLOT_CACHE is None:
        _PLACE_SLOT_CACHE = build_place_grid_slots()
    return _PLACE_SLOT_CACHE

def section_slot_groups(slots: list[np.ndarray]) -> dict[str, list[int]]:
    _bind_core_globals()
    if not slots:
        return {SECTION_LEFT_NAME: [], SECTION_RIGHT_NAME: []}
    ys = np.array([float(s[1]) for s in slots], dtype=float)
    y_mid = float(np.median(ys))
    left_idxs = [i for i, s in enumerate(slots) if float(s[1]) <= y_mid]
    right_idxs = [i for i, s in enumerate(slots) if float(s[1]) > y_mid]
    if not right_idxs:
        order = sorted(range(len(slots)), key=lambda i: float(slots[i][1]))
        cut = max(1, len(order) // 2)
        left_idxs, right_idxs = order[:cut], order[cut:]
    left_idxs = sorted(left_idxs, key=lambda i: float(slots[i][0]), reverse=True)
    right_idxs = sorted(right_idxs, key=lambda i: float(slots[i][0]), reverse=True)
    if bool(SECTION_LABEL_MIRROR):
        # Physical setup can be mirrored relative to logical left/right labels.
        left_idxs, right_idxs = right_idxs, left_idxs
    return {SECTION_LEFT_NAME: left_idxs, SECTION_RIGHT_NAME: right_idxs}

def next_slot_in_section(
    section_name: str,
    section_groups: dict[str, list[int]],
    placed_targets: list[np.ndarray],
    blocked_slots: set[int],
) -> int | None:
    _bind_core_globals()
    slots = get_place_slots()
    for idx in section_groups.get(section_name, []):
        if idx in blocked_slots:
            continue
        if idx < 0 or idx >= len(slots):
            continue
        ok, _, _ = slot_safety_status(slots[idx], placed_targets, MIN_PLACE_SLOT_SEPARATION_M)
        if ok:
            return int(idx)
    return None

# ============================= Projection / depth / base XYZ =============================
def _is_confirmed_active_placement(entry: dict | None) -> bool:
    _bind_core_globals()
    if not isinstance(entry, dict):
        return False
    if bool(entry.get("removed_by_return", False)):
        return False
    verify = entry.get("verify_result", None)
    if not isinstance(verify, dict):
        return False
    return bool(verify.get("confirmed", False))

def _is_stack_confirmed_active_placement(entry: dict | None) -> bool:
    _bind_core_globals()
    if not _is_confirmed_active_placement(entry):
        return False
    section_name = str(entry.get("section", "")).strip().lower()
    if section_name not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        return False
    command_name = str(entry.get("command", "")).strip().lower()
    if command_name not in {"place_left_stack", "place_right_stack"}:
        return False
    return _finite_xyz_or_none(entry.get("pick_origin_xyz")) is not None

def get_latest_confirmed_active_stack_placement(state: CycleState) -> dict | None:
    _bind_core_globals()
    for entry in reversed(list(state.placed_ledger)):
        if _is_stack_confirmed_active_placement(entry):
            return entry
    return None


def _get_locked_stack_anchor_xyz(
    state: CycleState,
    section_name: str,
) -> tuple[list[float] | None, str]:
    _bind_core_globals()
    section_norm = str(section_name).strip().lower()
    if section_norm not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        return None, "invalid_section"
    src = state.stack_anchor_xyz_by_section if isinstance(state.stack_anchor_xyz_by_section, dict) else {}
    xyz = _finite_xyz_or_none(src.get(section_norm))
    if xyz is None:
        return None, "none"
    source_src = (
        state.stack_anchor_source_by_section
        if isinstance(state.stack_anchor_source_by_section, dict)
        else {}
    )
    source = str(source_src.get(section_norm, "locked")).strip() or "locked"
    return [float(xyz[0]), float(xyz[1]), float(xyz[2])], f"locked:{source}"


def _set_locked_stack_anchor_xyz(
    state: CycleState,
    section_name: str,
    anchor_xyz: list[float] | tuple[float, ...] | np.ndarray,
    source: str,
) -> bool:
    _bind_core_globals()
    section_norm = str(section_name).strip().lower()
    if section_norm not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        return False
    try:
        arr = np.array(anchor_xyz, dtype=float).reshape(-1)
    except Exception:
        return False
    if arr.size < 2:
        return False
    if (not np.isfinite(float(arr[0]))) or (not np.isfinite(float(arr[1]))):
        return False
    z_val = float(arr[2]) if arr.size >= 3 and np.isfinite(float(arr[2])) else float(STACK_RELEASE_Z_GUARD_M)
    xyz_store = [float(arr[0]), float(arr[1]), float(z_val)]
    xyz_src = state.stack_anchor_xyz_by_section if isinstance(state.stack_anchor_xyz_by_section, dict) else {}
    xyz_out = dict(xyz_src)
    xyz_out[section_norm] = list(xyz_store)
    state.stack_anchor_xyz_by_section = dict(xyz_out)
    source_src = (
        state.stack_anchor_source_by_section
        if isinstance(state.stack_anchor_source_by_section, dict)
        else {}
    )
    source_out = dict(source_src)
    source_out[section_norm] = str(source or "unknown")
    state.stack_anchor_source_by_section = dict(source_out)
    return True


def _clear_locked_stack_anchor_xyz(state: CycleState, section_name: str) -> None:
    _bind_core_globals()
    section_norm = str(section_name).strip().lower()
    if section_norm not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        return
    xyz_src = state.stack_anchor_xyz_by_section if isinstance(state.stack_anchor_xyz_by_section, dict) else {}
    if section_norm in xyz_src:
        xyz_out = dict(xyz_src)
        xyz_out.pop(section_norm, None)
        state.stack_anchor_xyz_by_section = dict(xyz_out)
    source_src = (
        state.stack_anchor_source_by_section
        if isinstance(state.stack_anchor_source_by_section, dict)
        else {}
    )
    if section_norm in source_src:
        source_out = dict(source_src)
        source_out.pop(section_norm, None)
        state.stack_anchor_source_by_section = dict(source_out)


def _get_last_popped_xy(state: CycleState, section_name: str) -> list[float] | None:
    _bind_core_globals()
    section_norm = str(section_name).strip().lower()
    if section_norm == SECTION_LEFT_NAME:
        xy_row = getattr(state, "last_popped_left_xy", None)
    elif section_norm == SECTION_RIGHT_NAME:
        xy_row = getattr(state, "last_popped_right_xy", None)
    else:
        return None
    if not isinstance(xy_row, (list, tuple)) or len(xy_row) < 2:
        return None
    try:
        x_m = float(xy_row[0])
        y_m = float(xy_row[1])
    except Exception:
        return None
    if (not np.isfinite(x_m)) or (not np.isfinite(y_m)):
        return None
    return [float(x_m), float(y_m)]


def _set_last_popped_xy(state: CycleState, section_name: str, xy: list[float] | tuple[float, float]) -> bool:
    _bind_core_globals()
    section_norm = str(section_name).strip().lower()
    if section_norm not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        return False
    if not isinstance(xy, (list, tuple)) or len(xy) < 2:
        return False
    try:
        x_m = float(xy[0])
        y_m = float(xy[1])
    except Exception:
        return False
    if (not np.isfinite(x_m)) or (not np.isfinite(y_m)):
        return False
    xy_store = [float(x_m), float(y_m)]
    if section_norm == SECTION_LEFT_NAME:
        state.last_popped_left_xy = list(xy_store)
    else:
        state.last_popped_right_xy = list(xy_store)
    return True


def _clear_last_popped_xy(state: CycleState, section_name: str) -> None:
    _bind_core_globals()
    section_norm = str(section_name).strip().lower()
    if section_norm == SECTION_LEFT_NAME:
        state.last_popped_left_xy = None
    elif section_norm == SECTION_RIGHT_NAME:
        state.last_popped_right_xy = None


_COMMANDED_STACK_PLACE_COMMANDS = frozenset(
    {
        "place_right",
        "place_left",
        "place_right_stack",
        "place_left_stack",
    }
)
_COMMANDED_PLACE_BASE_SOURCE = "commanded_place_base_level0"


def _is_active_ledger_placement(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    return not bool(entry.get("removed_by_return", False))


def _locked_anchor_source_is_commanded_base(source: str) -> bool:
    src = str(source or "").strip().lower()
    return (
        _COMMANDED_PLACE_BASE_SOURCE in src
        or "verified_place_base" in src
    )


def _locked_anchor_source_is_startup_hydrate_top(source: str) -> bool:
    return "startup_hydrate_top" in str(source or "").strip().lower()


def get_commanded_stack_base_xyz(
    state: CycleState,
    section_name: str,
) -> tuple[list[float] | None, str]:
    """Earliest active level-0 place commanded expected_xyz on this side (not verify measured)."""
    _bind_core_globals()
    section_norm = str(section_name).strip().lower()
    if section_norm not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        return None, "invalid_section"
    best_commanded_xyz: list[float] | None = None
    best_cycle = float("inf")
    for entry in list(state.placed_ledger):
        if not _is_active_ledger_placement(entry):
            continue
        if str(entry.get("section", "")).strip().lower() != section_norm:
            continue
        try:
            stack_level = int(entry.get("stack_level", -1))
        except Exception:
            continue
        if int(stack_level) != 0:
            continue
        command_name = str(entry.get("command", "")).strip().lower()
        if command_name not in _COMMANDED_STACK_PLACE_COMMANDS:
            continue
        expected_xyz = _finite_xyz_or_none(entry.get("expected_xyz", None))
        if expected_xyz is None:
            continue
        try:
            cycle_val = float(entry.get("cycle", float("inf")))
        except Exception:
            cycle_val = float("inf")
        if (not np.isfinite(cycle_val)) or cycle_val > best_cycle:
            continue
        best_cycle = float(cycle_val)
        best_commanded_xyz = [
            float(expected_xyz[0]),
            float(expected_xyz[1]),
            float(expected_xyz[2]),
        ]
    if best_commanded_xyz is not None:
        return list(best_commanded_xyz), _COMMANDED_PLACE_BASE_SOURCE
    return None, "none"


def get_verified_stack_base_xyz(
    state: CycleState,
    section_name: str,
) -> tuple[list[float] | None, str]:
    """Deprecated: runtime stack anchor no longer uses verify measured_xyz."""
    return None, "none"


def commit_commanded_stack_base_anchor_from_place(
    state: CycleState,
    section_name: str,
    expected_xyz: list[float] | tuple[float, float, float] | np.ndarray | None,
    *,
    placed_stack_level: int,
) -> dict:
    _bind_core_globals()
    section_norm = str(section_name).strip().lower()
    if section_norm not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        return {"committed": False, "reason": "invalid_section", "section": section_norm}
    try:
        level_i = int(placed_stack_level)
    except Exception:
        level_i = -1
    if int(level_i) > 0:
        return {
            "committed": False,
            "reason": "not_base_level",
            "section": section_norm,
            "placed_stack_level": int(level_i),
        }
    xyz_norm = _finite_xyz_or_none(expected_xyz)
    if xyz_norm is None:
        print(
            f"[StackAnchorCommit] skip section={section_norm} "
            f"placed_stack_level={int(level_i)} reason=commanded_missing"
        )
        return {
            "committed": False,
            "reason": "commanded_missing",
            "section": section_norm,
            "placed_stack_level": int(level_i),
        }
    _set_locked_stack_anchor_xyz(state, section_norm, xyz_norm, _COMMANDED_PLACE_BASE_SOURCE)
    _clear_last_popped_xy(state, section_norm)
    print(
        f"[StackAnchorCommit] section={section_norm} source={_COMMANDED_PLACE_BASE_SOURCE} "
        f"xy=({float(xyz_norm[0]):.3f},{float(xyz_norm[1]):.3f}) "
        f"cleared_last_popped=True"
    )
    return {
        "committed": True,
        "reason": "ok",
        "section": section_norm,
        "anchor_xyz": list(xyz_norm),
        "cleared_last_popped": True,
    }


def commit_verified_stack_base_anchor_from_place(
    state: CycleState,
    section_name: str,
    measured_xyz: list[float] | tuple[float, float, float] | np.ndarray | None,
    *,
    placed_stack_level: int,
) -> dict:
    """Deprecated wrapper: use commanded expected_xyz, not verify measured_xyz."""
    return commit_commanded_stack_base_anchor_from_place(
        state=state,
        section_name=section_name,
        expected_xyz=measured_xyz,
        placed_stack_level=int(placed_stack_level),
    )


def _stack_anchor_z_for_last_popped(state: CycleState, section_name: str) -> float:
    _bind_core_globals()
    section_norm = str(section_name).strip().lower()
    lowest_z = float("inf")
    hydrated_row = get_startup_hydrated_section_row(state, section_norm)
    for row in list(hydrated_row.get("entries", [])):
        if not isinstance(row, dict):
            continue
        xyz = _finite_xyz_or_none(row.get("xyz", None))
        if xyz is None:
            continue
        try:
            z_val = float(xyz[2])
        except Exception:
            continue
        if np.isfinite(z_val) and z_val < lowest_z:
            lowest_z = float(z_val)
    if np.isfinite(lowest_z):
        return float(lowest_z)
    locked_anchor, _ = _get_locked_stack_anchor_xyz(state, section_norm)
    if isinstance(locked_anchor, (list, tuple)) and len(locked_anchor) >= 3:
        try:
            z_locked = float(locked_anchor[2])
        except Exception:
            z_locked = float("nan")
        if np.isfinite(z_locked):
            return float(z_locked)
    return float(STACK_RELEASE_Z_GUARD_M)


def get_latest_side_stack_anchor_xyz(
    state: CycleState,
    section_name: str,
) -> tuple[list[float] | None, str]:
    _bind_core_globals()
    section_norm = str(section_name).strip().lower()
    if section_norm not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        return None, "invalid_section"

    locked_anchor, locked_source = _get_locked_stack_anchor_xyz(state, section_norm)
    if locked_anchor is not None and _locked_anchor_source_is_commanded_base(str(locked_source)):
        return list(locked_anchor), _COMMANDED_PLACE_BASE_SOURCE

    commanded_xyz, commanded_source = get_commanded_stack_base_xyz(state, section_norm)
    if commanded_xyz is not None:
        _set_locked_stack_anchor_xyz(state, section_norm, commanded_xyz, _COMMANDED_PLACE_BASE_SOURCE)
        return list(commanded_xyz), str(commanded_source)

    if locked_anchor is not None and _locked_anchor_source_is_startup_hydrate_top(str(locked_source)):
        return list(locked_anchor), "startup_hydrate_top"

    last_popped_xy = _get_last_popped_xy(state, section_norm)
    if isinstance(last_popped_xy, (list, tuple)) and len(last_popped_xy) >= 2:
        z_anchor = _stack_anchor_z_for_last_popped(state, section_norm)
        return [
            float(last_popped_xy[0]),
            float(last_popped_xy[1]),
            float(z_anchor),
        ], "last_popped_xy"

    return None, "none"


def log_stack_anchor_missing(
    state: CycleState,
    section_name: str,
    *,
    stack_level: int,
) -> None:
    _bind_core_globals()
    section_norm = str(section_name).strip().lower()
    commanded_xyz, _ = get_commanded_stack_base_xyz(state, section_norm)
    last_popped_xy = _get_last_popped_xy(state, section_norm)
    locked_anchor, locked_source = _get_locked_stack_anchor_xyz(state, section_norm)
    auth_row = get_startup_hydrated_section_row(state, section_norm)
    try:
        auth_level = int(auth_row.get("stack_level", 0) or 0)
    except Exception:
        auth_level = 0
    entries_n = len(list(auth_row.get("entries", []))) if isinstance(auth_row.get("entries", []), list) else 0
    locked_tag = str(locked_source or "none").strip().lower() or "none"
    if locked_anchor is None:
        locked_tag = "none"
    print(
        f"[StackAnchor] missing section={section_norm} level={int(stack_level)} "
        f"commanded_level0={'yes' if commanded_xyz is not None else 'no'} "
        f"last_popped={'yes' if last_popped_xy is not None else 'no'} "
        f"locked={locked_tag} "
        f"auth_level={int(auth_level)} entries={int(entries_n)}"
    )

# ============================= Misplaced cube correction =============================
def _compact_section_truth_row(row: dict | None) -> dict:
    _bind_core_globals()
    src = row if isinstance(row, dict) else {}
    declared_level = src.get("stack_level", 0)
    seq_raw = src.get("color_sequence_bottom_to_top", [])
    level, slots, seq_use = _slots_from_level_and_sequence(declared_level, seq_raw)
    top_color = seq_use[-1] if seq_use else "unknown"
    out = {
        "stack_level": int(level),
        "top_color": str(top_color),
        "slots": dict(slots),
    }
    if bool(PLANNER_INCLUDE_COLOR_SEQUENCE):
        out["color_sequence_bottom_to_top"] = list(seq_use)
    return out

def _ledger_section_truth_row(state: CycleState, section_name: str) -> dict:
    _bind_core_globals()
    seq = get_section_confirmed_color_sequence_bottom_to_top(state, section_name)
    top_color = seq[-1] if seq else "unknown"
    return {
        "stack_level": int(len(seq)),
        "top_color": str(top_color),
        "color_sequence_bottom_to_top": list(seq),
    }

def run_place_space_truth_pass(
    *,
    state: CycleState,
    arm: Arm,
    per: Perception | None,
    det: YOLODetector | None,
    centered_pos: tuple[int, int] | None = None,
    active_track_id: int | None = None,
    mode: str = "pre_grasp_place_space",
    detector_draw: bool = False,
    show_window: bool = False,
    status_line: str = "",
) -> dict:
    _bind_core_globals()
    mode_name = str(mode).strip().lower()
    truth_source = "post_lift_refresh" if mode_name == "post_lift_place_space_refresh" else mode_name
    reconcile = reconcile_scene(
        state=state,
        arm=arm,
        per=per,
        det=det,
        side="all",
        mode=str(mode_name),
        include_pick_rows=False,
        detector_draw=bool(detector_draw),
        show_window=bool(show_window),
        status_line=str(status_line),
    )
    observed_left = _compact_section_truth_row(
        (reconcile.get("section_status", {}) if isinstance(reconcile, dict) else {}).get(SECTION_LEFT_NAME, {})
    )
    observed_right = _compact_section_truth_row(
        (reconcile.get("section_status", {}) if isinstance(reconcile, dict) else {}).get(SECTION_RIGHT_NAME, {})
    )
    ledger_left = _ledger_section_truth_row(state, SECTION_LEFT_NAME)
    ledger_right = _ledger_section_truth_row(state, SECTION_RIGHT_NAME)

    mismatch_sides: list[str] = []
    if (
        observed_left.get("color_sequence_bottom_to_top", []) != ledger_left.get("color_sequence_bottom_to_top", [])
        or int(observed_left.get("stack_level", 0)) != int(ledger_left.get("stack_level", 0))
    ):
        mismatch_sides.append(SECTION_LEFT_NAME)
    if (
        observed_right.get("color_sequence_bottom_to_top", []) != ledger_right.get("color_sequence_bottom_to_top", [])
        or int(observed_right.get("stack_level", 0)) != int(ledger_right.get("stack_level", 0))
    ):
        mismatch_sides.append(SECTION_RIGHT_NAME)

    target_uv = (
        [int(centered_pos[0]), int(centered_pos[1])]
        if isinstance(centered_pos, (list, tuple)) and len(centered_pos) >= 2
        else None
    )
    truth = {
        "status": str(reconcile.get("status", "unknown")) if isinstance(reconcile, dict) else "unknown",
        "source": str(truth_source),
        "collision_risk": bool(reconcile.get("collision_risk", False)) if isinstance(reconcile, dict) else False,
        "scene_revision": int(reconcile.get("scene_revision", state.scene_revision)) if isinstance(reconcile, dict) else int(state.scene_revision),
        "mismatch_sides": list(mismatch_sides),
        "section_status": {
            "observed": {
                SECTION_LEFT_NAME: observed_left,
                SECTION_RIGHT_NAME: observed_right,
            },
            "ledger": {
                SECTION_LEFT_NAME: ledger_left,
                SECTION_RIGHT_NAME: ledger_right,
            },
        },
        "target": {
            "track_id": (None if active_track_id is None else int(active_track_id)),
            "uv": target_uv,
        },
        "timestamp_ms": int(time.time() * 1000),
    }
    state.last_place_space_truth = dict(truth)
    state.place_space_check_scene_revision = int(state.scene_revision)
    state.place_space_check_target_track_id = (
        None if active_track_id is None else int(active_track_id)
    )
    state.place_space_check_target_uv = (
        None if target_uv is None else [int(target_uv[0]), int(target_uv[1])]
    )
    return truth

def _startup_default_hydrated_section_row() -> dict:
    _bind_core_globals()
    return {
        "stack_level": 0,
        "top_color": "unknown",
        "color_sequence_bottom_to_top": [],
        "tracks_bottom_to_top": [],
        "entries": [],
    }


def _startup_hydrate_sides_at_cap(entries_by_section: dict) -> bool:
    _bind_core_globals()
    cap = max(1, int(STARTUP_STACK_MAX_CUBES_PER_SIDE))
    for side in (SECTION_LEFT_NAME, SECTION_RIGHT_NAME):
        if int(len(entries_by_section.get(side, []))) < int(cap):
            return False
    return True


def _startup_hydrate_should_exit_when_sides_full(entries_by_section: dict) -> bool:
    _bind_core_globals()
    if not bool(STARTUP_STACK_EXIT_WHEN_SIDES_FULL):
        return False
    return bool(_startup_hydrate_sides_at_cap(entries_by_section))


def _normalize_hydrated_section_row(row: dict | None) -> dict:
    _bind_core_globals()
    src = row if isinstance(row, dict) else {}
    seq = [
        str(c).strip().lower()
        for c in list(src.get("color_sequence_bottom_to_top", []))
        if str(c).strip().lower() in {"orange", "blue", "unknown"}
    ]
    tracks_out: list[int | None] = []
    for tid in list(src.get("tracks_bottom_to_top", [])):
        if tid is None:
            tracks_out.append(None)
            continue
        try:
            tracks_out.append(int(tid))
        except (TypeError, ValueError):
            tracks_out.append(None)
    top_color = str(src.get("top_color", "unknown")).strip().lower()
    if top_color not in {"orange", "blue", "unknown"}:
        top_color = "unknown"
    try:
        level = int(src.get("stack_level", len(seq)) or 0)
    except Exception:
        level = int(len(seq))
    if len(seq) > int(level):
        level = int(len(seq))
    return {
        "stack_level": int(max(0, level)),
        "top_color": str(top_color),
        "color_sequence_bottom_to_top": list(seq),
        "tracks_bottom_to_top": list(tracks_out),
        "entries": list(src.get("entries", [])) if isinstance(src.get("entries", []), list) else [],
    }


def _merge_hydrated_section_row_keep_known(prev_row: dict | None, new_row: dict | None) -> dict:
    _bind_core_globals()
    prev = _normalize_hydrated_section_row(prev_row)
    new = _normalize_hydrated_section_row(new_row)
    try:
        new_level = int(new.get("stack_level", 0) or 0)
    except Exception:
        new_level = 0
    new_level = int(max(0, new_level))
    # If scan says empty, accept clear directly (do not preserve stale colors).
    if new_level <= 0:
        return dict(new)

    prev_seq = list(prev.get("color_sequence_bottom_to_top", []))
    new_seq = list(new.get("color_sequence_bottom_to_top", []))
    prev_tracks = list(prev.get("tracks_bottom_to_top", []))
    new_tracks = list(new.get("tracks_bottom_to_top", []))
    if len(new_seq) < new_level:
        new_seq.extend(["unknown"] * int(new_level - len(new_seq)))
    if len(new_tracks) < new_level:
        new_tracks.extend([None] * int(new_level - len(new_tracks)))
    if len(prev_seq) < int(prev.get("stack_level", 0) or 0):
        prev_seq.extend(["unknown"] * int(int(prev.get("stack_level", 0) or 0) - len(prev_seq)))
    if len(prev_tracks) < int(prev.get("stack_level", 0) or 0):
        prev_tracks.extend([None] * int(int(prev.get("stack_level", 0) or 0) - len(prev_tracks)))

    preserve_count = int(min(new_level, int(prev.get("stack_level", 0) or 0)))
    for i in range(preserve_count):
        prev_color = str(prev_seq[i]).strip().lower() if i < len(prev_seq) else "unknown"
        new_color = str(new_seq[i]).strip().lower() if i < len(new_seq) else "unknown"
        if new_color == "unknown" and prev_color in {"orange", "blue"}:
            new_seq[i] = str(prev_color)
        if (i < len(new_tracks)) and (new_tracks[i] is None) and (i < len(prev_tracks)):
            prev_tid = prev_tracks[i]
            if prev_tid is not None:
                try:
                    new_tracks[i] = int(prev_tid)
                except (TypeError, ValueError):
                    pass

    seq_use = list(new_seq[:new_level])
    top_color = "unknown" if not seq_use else str(seq_use[-1]).strip().lower()
    if top_color not in {"orange", "blue", "unknown"}:
        top_color = "unknown"
    merged = {
        "stack_level": int(new_level),
        "top_color": str(top_color),
        "color_sequence_bottom_to_top": list(seq_use),
        "tracks_bottom_to_top": list(new_tracks[:new_level]),
        "entries": list(new.get("entries", [])) if isinstance(new.get("entries", []), list) else [],
    }
    return _normalize_hydrated_section_row(merged)


def _startup_vote_hits_for_layer_slot(
  burst_rows: list[dict],
  *,
  predicted_xyz: np.ndarray,
  layer_xy_m: float,
  layer_z_m: float,
  excluded_track_ids: set[int] | None = None,
  blocked_track_ids: set[int] | None = None,
) -> list[dict]:
    """Filter burst observations to hits near a predicted layer slot."""
    excluded = set(excluded_track_ids or set())
    blocked = set(blocked_track_ids or set())
    vote_hits: list[dict] = []
    for row in list(burst_rows):
        prow = row.get("prow", {})
        if not isinstance(prow, dict):
            continue
        raw_tid = prow.get("track_id", None)
        try:
            tid_i = None if raw_tid is None else int(raw_tid)
        except Exception:
            tid_i = None
        if tid_i is not None:
            if int(tid_i) in excluded or int(tid_i) in blocked:
                continue
        xyz_i = np.array(row.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
        if xyz_i.size < 3 or not np.all(np.isfinite(xyz_i[:3])):
            continue
        d_xy = float(
            math.hypot(
                float(xyz_i[0]) - float(predicted_xyz[0]),
                float(xyz_i[1]) - float(predicted_xyz[1]),
            )
        )
        d_z = float(abs(float(xyz_i[2]) - float(predicted_xyz[2])))
        if d_xy > float(layer_xy_m) or d_z > float(layer_z_m):
            continue
        vote_hits.append(
            {
                "prow": dict(prow),
                "obs": row.get("obs"),
                "d_xy": float(d_xy),
                "d_z": float(d_z),
                "xyz": xyz_i[:3].copy(),
            }
        )
    return vote_hits


def _startup_side_full_rescan_top_to_bottom(
    *,
    section_name: str,
    expected_layers: int,
    missing_layers: list[int],
    anchor_xy: tuple[float, float],
    stack_base_z: float,
    entries_by_section: dict[str, list[dict]],
    section_centers_xy: dict,
    processed_track_ids: set[int],
    accepted_xyzs: list[np.ndarray],
    det,
    arm,
    per,
    state,
    conf_min: float,
    startup_target_min_conf: float,
    layer_xy_m: float,
    layer_z_m: float,
    rescan_frames: int,
    vote_min_hits: int,
    dz_step: float,
    observe_scene_frame_fn,
    classify_color_at_uv_fn,
    xyz_duplicate_fn,
    infer_section_fn,
) -> int:
    """One multi-frame burst + top-to-bottom slot vote for gap recovery (no handoff lock)."""
    _bind_core_globals()
    section_norm = str(section_name).strip().lower()
    missing_sorted = sorted({int(layer_i) for layer_i in list(missing_layers)}, reverse=True)
    if not missing_sorted:
        return 0
    rescan_frames_i = max(1, int(rescan_frames))
    vote_min_hits_i = max(1, min(int(rescan_frames_i), int(vote_min_hits)))
    dz_local = max(1e-6, float(dz_step))
    burst_rows: list[dict] = []
    for _scan_i in range(int(rescan_frames_i)):
        obs_scan = observe_scene_frame_fn(
            det=det,
            arm=arm,
            per=per,
            draw=False,
            projected_min_conf=float(conf_min),
            state=state,
            update_tracks=True,
        )
        if obs_scan is None:
            time.sleep(float(max(0.0, arm.sample_time)))
            continue
        for prow in list(obs_scan.projected_rows):
            conf_i = float(prow.get("conf", 0.0))
            if conf_i < float(startup_target_min_conf):
                continue
            raw_tid = prow.get("track_id", None)
            try:
                tid_i = None if raw_tid is None else int(raw_tid)
            except Exception:
                tid_i = None
            xyz_i = np.array(prow.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
            if xyz_i.size < 3 or not np.all(np.isfinite(xyz_i[:3])):
                continue
            section_name_xy, _assign_info = infer_section_fn(
                float(xyz_i[0]),
                float(xyz_i[1]),
                section_centers_xy,
                band_min=None,
                band_max=None,
                max_center_dist_m=float(STARTUP_STACK_LOCK_ASSIGN_XY_MARGIN_M),
            )
            section_norm_xy = "" if section_name_xy is None else str(section_name_xy).strip().lower()
            if section_norm_xy != section_norm:
                continue
            burst_rows.append(
                {
                    "prow": dict(prow),
                    "obs": obs_scan,
                    "xyz": xyz_i[:3].copy(),
                    "track_id": tid_i,
                }
            )
        time.sleep(float(max(0.0, arm.sample_time)))
    rescan_consumed_tids: set[int] = set()
    recovered = 0
    print(
        f"[StartupHydrateSideRescan] side={section_norm} expected={int(expected_layers)} "
        f"missing={missing_sorted} frames={int(rescan_frames_i)} "
        f"burst_rows={int(len(burst_rows))} vote_min={int(vote_min_hits_i)}"
    )
    for layer_idx in missing_sorted:
        predicted_xyz = np.array(
            [
                float(anchor_xy[0]),
                float(anchor_xy[1]),
                float(stack_base_z + int(layer_idx) * dz_local),
            ],
            dtype=float,
        )
        vote_hits = _startup_vote_hits_for_layer_slot(
            burst_rows,
            predicted_xyz=predicted_xyz,
            layer_xy_m=float(layer_xy_m),
            layer_z_m=float(layer_z_m),
            excluded_track_ids=rescan_consumed_tids,
            blocked_track_ids=processed_track_ids,
        )
        if len(vote_hits) < int(vote_min_hits_i):
            print(
                f"[StartupHydrateLayerVote] side={section_norm} layer={int(layer_idx)} "
                f"hits={len(vote_hits)}/{int(rescan_frames_i)} min={int(vote_min_hits_i)} "
                f"predicted_xyz={[float(predicted_xyz[0]), float(predicted_xyz[1]), float(predicted_xyz[2])]} "
                f"reason=below_min_hits rescan=yes"
            )
            print(
                f"[StartupHydrateLayerMissing] side={section_norm} layer={int(layer_idx)} "
                f"predicted_xyz={[float(predicted_xyz[0]), float(predicted_xyz[1]), float(predicted_xyz[2])]} "
                f"reason=no_visible_candidate"
            )
            continue
        xyz_stack = np.array([np.asarray(h["xyz"], dtype=float).reshape(3) for h in vote_hits])
        measured_xyz = np.median(xyz_stack, axis=0).reshape(-1)
        best_hit = min(
            vote_hits,
            key=lambda h: float(
                math.hypot(
                    float(h["xyz"][0]) - float(measured_xyz[0]),
                    float(h["xyz"][1]) - float(measured_xyz[1]),
                )
                + abs(float(h["xyz"][2]) - float(measured_xyz[2]))
            ),
        )
        best_match = dict(best_hit["prow"])
        best_obs = best_hit["obs"]
        print(
            f"[StartupHydrateLayerVote] side={section_norm} layer={int(layer_idx)} "
            f"hits={len(vote_hits)}/{int(rescan_frames_i)} min={int(vote_min_hits_i)} "
            f"predicted_xyz={[float(predicted_xyz[0]), float(predicted_xyz[1]), float(predicted_xyz[2])]} "
            f"median_xyz={[float(measured_xyz[0]), float(measured_xyz[1]), float(measured_xyz[2])]} "
            f"rescan=yes"
        )
        if bool(xyz_duplicate_fn(measured_xyz)):
            print(
                f"[StartupHydrateLayerMissing] side={section_norm} layer={int(layer_idx)} "
                f"predicted_xyz={[float(predicted_xyz[0]), float(predicted_xyz[1]), float(predicted_xyz[2])]} "
                f"reason=duplicate_of_committed"
            )
            continue
        uv_match = [int(best_match.get("u", 0)), int(best_match.get("v", 0))]
        color_name, color_conf = ("unknown", 0.0)
        if best_obs is not None:
            color_name, color_conf = classify_color_at_uv_fn(best_obs, uv_match)
        final_tid = None
        raw_tid = best_match.get("track_id", None)
        try:
            final_tid = None if raw_tid is None else int(raw_tid)
        except Exception:
            final_tid = None
        if final_tid is None:
            linked_tid = nearest_visible_track_by_uv(
                state, int(uv_match[0]), int(uv_match[1]), max_dist_px=90.0
            )
            if linked_tid is not None:
                final_tid = int(linked_tid)
        if final_tid is not None:
            processed_track_ids.add(int(final_tid))
            rescan_consumed_tids.add(int(final_tid))
        accepted_xyzs.append(measured_xyz[:3].copy())
        entries_by_section[str(section_norm)].append(
            {
                "track_id": final_tid,
                "uv": list(uv_match),
                "xyz": [
                    float(measured_xyz[0]),
                    float(measured_xyz[1]),
                    float(measured_xyz[2]),
                ],
                "color": str(color_name),
                "conf": float(color_conf),
                "order_index": int(len(entries_by_section[str(section_norm)])),
                "source": "side_full_rescan",
                "lock_source": "side_full_rescan",
                "target_track_id": final_tid,
                "selected_track_id": final_tid,
            }
        )
        recovered += 1
        print(
            f"[StartupHydrateLayerRecovered] side={section_norm} layer={int(layer_idx)} "
            f"predicted_xyz={[float(predicted_xyz[0]), float(predicted_xyz[1]), float(predicted_xyz[2])]} "
            f"measured_xyz={[float(measured_xyz[0]), float(measured_xyz[1]), float(measured_xyz[2])]} "
            f"track={final_tid} color={color_name} rescan=yes"
        )
    print(
        f"[StartupHydrateSideRescan] side={section_norm} recovered={int(recovered)} "
        f"missing_layers={missing_sorted}"
    )
    return int(recovered)


def run_startup_stack_identity_pass(
    *,
    state: CycleState,
    arm: Arm,
    per: Perception | None,
    det: YOLODetector | None,
    samples: int | None = None,
    show_window: bool = False,
    detector_draw: bool = False,
) -> dict:
    _bind_core_globals()
    summary = {
        "status": "ok",
        "sections": {
            SECTION_LEFT_NAME: _startup_default_hydrated_section_row(),
            SECTION_RIGHT_NAME: _startup_default_hydrated_section_row(),
        },
        "observed_stack_levels": {SECTION_LEFT_NAME: 0, SECTION_RIGHT_NAME: 0},
        "expected_stack_levels": {SECTION_LEFT_NAME: 0, SECTION_RIGHT_NAME: 0},
        "expected_level_shortfall_sides": [],
        "observed_sequences": {SECTION_LEFT_NAME: [], SECTION_RIGHT_NAME: []},
        "samples": max(1, int(samples if samples is not None else STARTUP_STACK_IDENTITY_SAMPLES)),
        "valid_frames": 0,
        "timestamp_ms": int(time.time() * 1000),
    }
    if det is None or per is None:
        summary["status"] = "dependencies_missing"
        return summary

    section_centers = _verify_section_y_centers()
    section_centers_xy = _verify_section_xy_centers()
    if bool(STARTUP_STACK_USE_PLACE_BAND):
        band_min, band_max = _place_space_y_band_bounds()
    else:
        band_min, band_max = float("-inf"), float("inf")
    if not section_centers:
        summary["status"] = "section_centers_missing"
        return summary
    if bool(STARTUP_STACK_ASSIGN_DEBUG):
        pass

    n = max(1, int(samples if samples is not None else STARTUP_STACK_IDENTITY_SAMPLES))
    scan_timeout_s = float(max(0.8, STARTUP_STACK_DISCOVERY_TIMEOUT_S))
    no_det_timeout_s = float(
        max(0.6, min(scan_timeout_s, STARTUP_STACK_DISCOVERY_NO_DET_TIMEOUT_S))
    )
    min_scan_s = float(max(0.3, min(3.0, STARTUP_STACK_DISCOVERY_MIN_SCAN_S)))
    stable_window_s = float(max(0.2, min(3.0, STARTUP_STACK_DISCOVERY_STABLE_S)))
    conf_min = float(max(SCENE_RECON_MIN_CONF, PLACE_VERIFY_MIN_CONF))
    groups: list[dict] = []
    peak_frame_candidates = 0

    def _section_distance_diag(x_m: float, y_m: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, center_xy in dict(section_centers_xy).items():
            try:
                cx = float(center_xy[0])
                cy = float(center_xy[1])
            except Exception:
                continue
            out[str(name)] = round(float(math.hypot(float(x_m) - cx, float(y_m) - cy)), 4)
        return out

    def _find_group_for_row(track_id: int | None, xyz: np.ndarray, u_px: int, v_px: int) -> dict | None:
        xy_merge_m = float(max(0.005, SCENE_RECON_DEDUP_XY_M))
        z_merge_m = float(
            min(
                max(0.005, SCENE_RECON_DEDUP_Z_M),
                max(0.010, float(STACK_LEVEL_DZ_M) * 0.45),
            )
        )
        if track_id is not None:
            for row in groups:
                if row.get("track_id", None) is None or int(row.get("track_id")) != int(track_id):
                    continue
                r_xyz = np.array(row.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
                if r_xyz.size >= 3 and np.all(np.isfinite(r_xyz[:3])):
                    d_xy_tid = float(math.hypot(float(xyz[0]) - float(r_xyz[0]), float(xyz[1]) - float(r_xyz[1])))
                    d_z_tid = float(abs(float(xyz[2]) - float(r_xyz[2])))
                    # If same track id jumps by about one cube height, treat as separate candidate.
                    if d_xy_tid > xy_merge_m or d_z_tid > z_merge_m:
                        continue
                uv_seed = row.get("uv_seed", None)
                if isinstance(uv_seed, (list, tuple)) and len(uv_seed) >= 2:
                    try:
                        du_px = abs(int(u_px) - int(uv_seed[0]))
                        dv_px = abs(int(v_px) - int(uv_seed[1]))
                    except (TypeError, ValueError):
                        du_px, dv_px = 0, 0
                    # Track IDs can alias across stacked cubes; split into separate groups if UV moved enough.
                    if du_px > 20 or dv_px > 20:
                        continue
                return row
        best = None
        best_score = float("inf")
        for row in groups:
            r_xyz = np.array(row.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
            if r_xyz.size < 3 or not np.all(np.isfinite(r_xyz[:3])):
                continue
            d_xy = float(math.hypot(float(xyz[0]) - float(r_xyz[0]), float(xyz[1]) - float(r_xyz[1])))
            d_z = float(abs(float(xyz[2]) - float(r_xyz[2])))
            if d_xy <= xy_merge_m and d_z <= z_merge_m:
                score = float(d_xy + 0.2 * d_z)
                if score < best_score:
                    best_score = score
                    best = row
        return best

    t_scan0 = time.time()
    hard_deadline = float(t_scan0 + scan_timeout_s)
    active_deadline = float(t_scan0 + no_det_timeout_s)
    last_new_group_t = float(t_scan0)
    try:
        while time.time() < min(hard_deadline, active_deadline):
            obs = observe_scene_frame(
                det=det,
                arm=arm,
                per=per,
                draw=bool(detector_draw),
                projected_min_conf=float(conf_min),
                state=state,
                update_tracks=True,
            )
            if obs is None:
                continue
            summary["valid_frames"] = int(summary["valid_frames"]) + 1
            frame_place_candidates = 0
            groups_before = int(len(groups))
            for prow in list(obs.projected_rows):
                conf = float(prow.get("conf", 0.0))
                if conf < float(conf_min):
                    continue
                xyz = np.array(prow.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
                if xyz.size < 3 or not np.all(np.isfinite(xyz[:3])):
                    continue
                if float(xyz[1]) <= float(PICK_MAX_BASE_Y_M):
                    continue
                if float(xyz[1]) < float(band_min) or float(xyz[1]) > float(band_max):
                    continue
                frame_place_candidates += 1
                raw_tid = prow.get("track_id", None)
                try:
                    track_id = None if raw_tid is None else int(raw_tid)
                except Exception:
                    track_id = None
                u_px = int(prow.get("u", 0))
                v_px = int(prow.get("v", 0))
                group = _find_group_for_row(track_id, xyz, u_px, v_px)
                if group is None:
                    group = {
                        "track_id": (None if track_id is None else int(track_id)),
                        "xyz_samples": [],
                        "uv_samples": [],
                        "conf_samples": [],
                        "uv_seed": [int(u_px), int(v_px)],
                        "bbox_xyxy": prow.get("bbox_xyxy", None),
                        "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                    }
                    groups.append(group)
                elif track_id is not None:
                    # Keep discovery group IDs fresh so startup checklist does not freeze stale early IDs.
                    group["track_id"] = int(track_id)
                group["xyz_samples"].append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
                group["uv_samples"].append([int(prow.get("u", 0)), int(prow.get("v", 0))])
                group["conf_samples"].append(float(conf))
                group["uv_seed"] = [int(u_px), int(v_px)]
                group["xyz"] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
                if prow.get("bbox_xyxy", None) is not None:
                    group["bbox_xyxy"] = prow.get("bbox_xyxy", None)

            peak_frame_candidates = max(int(peak_frame_candidates), int(frame_place_candidates))
            if int(frame_place_candidates) > 0:
                active_deadline = float(min(hard_deadline, time.time() + no_det_timeout_s))
            if int(len(groups)) > int(groups_before):
                last_new_group_t = float(time.time())

            if bool(show_window) and bool(SHOW_WINDOW):
                try:
                    elapsed_s = float(time.time() - t_scan0)
                    stable_s = float(max(0.0, time.time() - float(last_new_group_t)))
                    frame = render_operator_overlay(
                        frame=obs.image_display.copy(),
                        state=state,
                        ui_mode=UI_MODE,
                        tracks=state.track_memory,
                        active_track_id=state.active_target_track_id,
                        cx=int(obs.image_center_uv[0]),
                        cy=int(obs.image_center_uv[1]),
                        status_line=(
                            f"startup_identity_scan t={elapsed_s:.1f}/{scan_timeout_s:.1f}s "
                            f"nodet_left={max(0.0, active_deadline - time.time()):.1f}s "
                            f"groups={len(groups)} peak={int(peak_frame_candidates)} "
                            f"stable={stable_s:.1f}s"
                        ),
                    )
                    cv2.imshow(WINDOW_NAME, frame)
                    cv2.waitKey(1)
                except Exception:
                    pass

            elapsed_s = float(time.time() - t_scan0)
            stable_s = float(max(0.0, time.time() - float(last_new_group_t)))
            if (
                int(len(groups)) > 0
                and int(summary["valid_frames"]) >= int(n)
                and elapsed_s >= float(min_scan_s)
                and stable_s >= float(stable_window_s)
            ):
                break

        if int(summary["valid_frames"]) <= 0:
            summary["status"] = "no_observation"
            return summary

        rep_rows: list[dict] = []
        min_group_obs = max(1, int(STARTUP_STACK_MIN_GROUP_OBS))
        for group in list(groups):
            xyz_samples = np.array(group.get("xyz_samples", []), dtype=float)
            if xyz_samples.size == 0:
                continue
            if int(xyz_samples.shape[0]) < int(min_group_obs):
                continue
            xyz_med = np.median(xyz_samples, axis=0)
            xyz_norm = _finite_xyz_or_none(xyz_med)
            if xyz_norm is None:
                continue
            uv_samples = list(group.get("uv_samples", []))
            conf_samples = list(group.get("conf_samples", []))
            if uv_samples:
                best_i = int(np.argmax(np.array(conf_samples, dtype=float))) if conf_samples else 0
                best_i = max(0, min(best_i, len(uv_samples) - 1))
                uv_seed = [int(uv_samples[best_i][0]), int(uv_samples[best_i][1])]
            else:
                uv_seed = [0, 0]
            rep_rows.append(
                {
                    "track_id": group.get("track_id", None),
                    "xyz": list(xyz_norm),
                    "uv": list(uv_seed),
                    "bbox_xyxy": group.get("bbox_xyxy", None),
                    "conf": float(max(conf_samples) if conf_samples else 0.0),
                }
            )

        rep_rows.sort(
            key=lambda row: float(row.get("xyz", [0.0, 0.0, 0.0])[2]),
            reverse=bool(STARTUP_STACK_LOCK_TOP_FIRST),
        )
        if bool(STARTUP_STACK_ASSIGN_DEBUG):
            pass
        max_cubes_total = max(
            1,
            max(
                int(STARTUP_STACK_MAX_CUBES_PER_SIDE) * 2,
                int(STARTUP_STACK_MAX_TRACK_TARGETS),
            ),
        )
        rep_rows = rep_rows[:max_cubes_total]
        rep_rows_count = int(len(rep_rows))
        lock_failed_rows = 0
        processed_track_ids: set[int] = set()
        accepted_xyzs: list[np.ndarray] = []
        entries_by_section: dict[str, list[dict]] = {
            SECTION_LEFT_NAME: [],
            SECTION_RIGHT_NAME: [],
        }
        dedup_xy_m = float(max(0.020, SCENE_RECON_DEDUP_XY_M * 1.40))
        dedup_z_m = float(max(0.012, SCENE_RECON_DEDUP_Z_M * 0.5))

        # Current-frame visible-only startup targets:
        # lock exactly what is visible now and append newly appeared IDs during startup.
        checklist_rows: list[dict] = []
        checklist_tid_set: set[int] = set()
        startup_target_min_conf = float(STARTUP_TARGET_MIN_CONF)

        def _collect_visible_targets(obs_row: SceneObservation | None) -> tuple[list[dict], tuple[int, int]]:
            if obs_row is None:
                return [], (0, 0)
            cx_now, cy_now = int(obs_row.image_center_uv[0]), int(obs_row.image_center_uv[1])
            by_tid: dict[int, dict] = {}
            for prow in list(obs_row.projected_rows):
                conf_i = float(prow.get("conf", 0.0))
                if conf_i < float(max(conf_min, startup_target_min_conf)):
                    continue
                xyz_i = np.array(prow.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
                if xyz_i.size < 3 or not np.all(np.isfinite(xyz_i[:3])):
                    continue
                if float(xyz_i[1]) <= float(PICK_MAX_BASE_Y_M):
                    continue
                if float(xyz_i[1]) < float(band_min) or float(xyz_i[1]) > float(band_max):
                    continue
                raw_tid = prow.get("track_id", None)
                try:
                    tid_i = None if raw_tid is None else int(raw_tid)
                except Exception:
                    tid_i = None
                if tid_i is None:
                    continue
                row_i = {
                    "track_id": int(tid_i),
                    "xyz": [float(xyz_i[0]), float(xyz_i[1]), float(xyz_i[2])],
                    "uv": [int(prow.get("u", 0)), int(prow.get("v", 0))],
                    "bbox_xyxy": prow.get("bbox_xyxy", None),
                    "conf": float(conf_i),
                }
                prev_i = by_tid.get(int(tid_i))
                if prev_i is None or float(row_i["conf"]) > float(prev_i.get("conf", 0.0)):
                    by_tid[int(tid_i)] = dict(row_i)
            rows = list(by_tid.values())
            rows.sort(
                key=lambda row: (
                    int((int(row.get("uv", [0, 0])[0]) - int(cx_now)) ** 2 + (int(row.get("uv", [0, 0])[1]) - int(cy_now)) ** 2),
                    -float(row.get("conf", 0.0)),
                    int(row.get("track_id", -1)),
                )
            )
            return rows, (int(cx_now), int(cy_now))

        def _append_visible_targets_from_observation(obs_row: SceneObservation | None) -> int:
            if _startup_hydrate_should_exit_when_sides_full(entries_by_section):
                return 0
            rows, _ = _collect_visible_targets(obs_row)
            added = 0
            for row in rows:
                if int(len(checklist_rows)) >= int(max_cubes_total):
                    break
                tid_i = int(row.get("track_id"))
                if int(tid_i) in checklist_tid_set:
                    continue
                checklist_tid_set.add(int(tid_i))
                row_i = int(len(checklist_rows))
                checklist_rows.append(
                    {
                        "key": f"track:{tid_i}",
                        "track_id": int(tid_i),
                        "expected_track_id_initial": int(tid_i),
                        "final_track_id": None,
                        "rebound": False,
                        "rebind_reason": "",
                        "row_index": int(row_i),
                        "row": dict(row),
                        "status": "pending",
                        "attempts": 0,
                        "last_seen_ts": None,
                        "last_lock_reason": "pending",
                        "exit_reason": "",
                    }
                )
                added += 1
            return int(added)

        obs_lock_seed = observe_scene_frame(
            det=det,
            arm=arm,
            per=per,
            draw=bool(detector_draw),
            projected_min_conf=float(conf_min),
            state=state,
            update_tracks=True,
        )
        _append_visible_targets_from_observation(obs_lock_seed)

        checklist_committed = 0
        checklist_failed = 0
        lock_skipped_count = 0
        skip_reasons: dict[str, int] = {}
        if bool(STARTUP_STACK_ASSIGN_DEBUG):
            pass

        last_visibility_obs: SceneObservation | None = None

        def _pump_startup_preview(
            obs_preview: SceneObservation | None,
            status_line: str,
        ) -> None:
            if not (bool(show_window) and bool(SHOW_WINDOW)):
                return
            try:
                if obs_preview is None:
                    cv2.waitKey(1)
                    return
                frame = render_operator_overlay(
                    frame=obs_preview.image_display.copy(),
                    state=state,
                    ui_mode=UI_MODE,
                    tracks=state.track_memory,
                    active_track_id=state.active_target_track_id,
                    cx=int(obs_preview.image_center_uv[0]),
                    cy=int(obs_preview.image_center_uv[1]),
                    status_line=str(status_line),
                )
                cv2.imshow(WINDOW_NAME, frame)
                cv2.waitKey(1)
            except Exception:
                pass

        def _check_target_visible_now(target_tid: int) -> tuple[bool, float]:
            nonlocal last_visibility_obs
            obs_now = observe_scene_frame(
                det=det,
                arm=arm,
                per=per,
                draw=False,
                projected_min_conf=float(conf_min),
                state=state,
                update_tracks=True,
            )
            last_visibility_obs = obs_now
            if obs_now is None:
                return False, 0.0
            _append_visible_targets_from_observation(obs_now)
            best_conf = 0.0
            for cand in list(obs_now.candidates):
                raw_tid = cand.get("track_id", None)
                try:
                    cand_tid = None if raw_tid is None else int(raw_tid)
                except Exception:
                    cand_tid = None
                if cand_tid is None or int(cand_tid) != int(target_tid):
                    continue
                conf_c = float(cand.get("conf", 0.0))
                if conf_c > best_conf:
                    best_conf = float(conf_c)
            return bool(best_conf >= float(startup_target_min_conf)), float(best_conf)

        def _attempt_startup_checklist_item(item: dict) -> str:
            row = item.get("row", {}) if isinstance(item.get("row", {}), dict) else {}
            row_i = int(item.get("row_index", 0))
            target_tid = item.get("track_id", None)

            seed_xyz = np.array(row.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
            if seed_xyz.size < 3 or not np.all(np.isfinite(seed_xyz[:3])):
                return "retry"
            if bool(STARTUP_STACK_ASSIGN_DEBUG):
                pass

            if target_tid is None:
                item["lock_source"] = "track_missing"
                item["selected_track_id"] = None
                if bool(STARTUP_STACK_ASSIGN_DEBUG):
                    pass
                return "failed_no_track_id"

            if TRACK_ENABLE:
                track_row_now = state.track_memory.get(int(target_tid), None)
                if isinstance(track_row_now, dict) and int(track_row_now.get("miss_frames", 999)) == 0:
                    item["last_seen_ts"] = float(time.time())

            blocked_extra_ids: set[int] = set(int(tid) for tid in list(processed_track_ids))
            if target_tid is not None:
                try:
                    blocked_extra_ids.discard(int(target_tid))
                except Exception:
                    pass

            def _startup_track_locked_candidate(
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
                _ = (distance_to_blocked_xyz, blocked_xyzs, xy_margin_m, z_margin_m)
                if target_tid is not None and int(track_id) != int(target_tid):
                    reject_reason = (
                        f"wrong_track_for_item expected={int(target_tid)} got={int(track_id)}"
                    )
                    item["rebound"] = False
                    item["rebind_reason"] = str(reject_reason)
                    return {
                        "decision": "reject",
                        "reason": str(reject_reason),
                    }
                measure = collect_track_measurement(
                    track_id=int(track_id),
                    first_obs=obs,
                    first_candidate=selected_row,
                    sample_count_override=int(STARTUP_STACK_MEASURE_SAMPLES),
                )
                hits_i = int(measure.get("hits", 0))
                xyz_local = measure.get("median_xyz", None)
                xyz_arr = np.array(
                    xyz_local if isinstance(xyz_local, (list, tuple, np.ndarray)) else [np.nan, np.nan, np.nan],
                    dtype=float,
                ).reshape(-1)
                xyz_ok = bool(xyz_arr.size >= 3 and np.all(np.isfinite(xyz_arr[:3])))
                if (not xyz_ok) or hits_i < int(max(1, STARTUP_STACK_MEASURE_MIN_HITS)):
                    return {
                        "decision": "continue",
                        "reason": (
                            f"quality_gate_wait hits={hits_i}/{int(max(1, STARTUP_STACK_MEASURE_MIN_HITS))} "
                            f"xyz_ok={bool(xyz_ok)}"
                        ),
                        "selected_xyz": (
                            None
                            if not xyz_ok
                            else [float(xyz_arr[0]), float(xyz_arr[1]), float(xyz_arr[2])]
                        ),
                    }
                section_name_cb, _assign_info_cb = _infer_section_for_place_xy(
                    float(xyz_arr[0]),
                    float(xyz_arr[1]),
                    section_centers_xy,
                    band_min=None,
                    band_max=None,
                    max_center_dist_m=float(STARTUP_STACK_LOCK_ASSIGN_XY_MARGIN_M),
                )
                section_norm_cb = "" if section_name_cb is None else str(section_name_cb).strip().lower()
                if section_norm_cb not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
                    assign_reason = str(_assign_info_cb.get("reason", "unknown")) if isinstance(_assign_info_cb, dict) else "unknown"
                    return {
                        "decision": "continue",
                        "reason": f"unassigned_section:{assign_reason}",
                        "selected_xyz": [float(xyz_arr[0]), float(xyz_arr[1]), float(xyz_arr[2])],
                    }
                # Startup-only anti-diversion gate:
                # if this candidate is spatially the same cube as one already committed,
                # skip it without consuming reject budget.
                for prev_xyz in list(accepted_xyzs):
                    if prev_xyz.size < 3 or not np.all(np.isfinite(prev_xyz[:3])):
                        continue
                    d_xy_prev = float(
                        math.hypot(float(xyz_arr[0]) - float(prev_xyz[0]), float(xyz_arr[1]) - float(prev_xyz[1]))
                    )
                    d_z_prev = float(abs(float(xyz_arr[2]) - float(prev_xyz[2])))
                    if d_xy_prev <= float(dedup_xy_m) and d_z_prev <= float(dedup_z_m):
                        return {
                            "decision": "reject",
                            "reason": (
                                f"duplicate_measurement d_xy={d_xy_prev:.3f}<= {float(dedup_xy_m):.3f} "
                                f"d_z={d_z_prev:.3f}<= {float(dedup_z_m):.3f}"
                            ),
                            "blocked_xyz": [float(xyz_arr[0]), float(xyz_arr[1]), float(xyz_arr[2])],
                            "selected_xyz": [float(xyz_arr[0]), float(xyz_arr[1]), float(xyz_arr[2])],
                        }
                return {
                    "decision": "accept",
                    "reason": "startup_track_locked",
                    "selected_xyz": [float(xyz_arr[0]), float(xyz_arr[1]), float(xyz_arr[2])],
                    "accept_payload": {
                        "hits": int(hits_i),
                        "required_hits": int(max(1, STARTUP_STACK_MEASURE_MIN_HITS)),
                        "section": str(section_norm_cb),
                        "rebound": False,
                        "rebind_reason": "none",
                        "expected_track_id_initial": (
                            None
                            if item.get("expected_track_id_initial", None) is None
                            else int(item.get("expected_track_id_initial"))
                        ),
                        "final_track_id": int(track_id),
                    },
                }

            lock_timeout_s = float(max(0.8, STARTUP_STACK_LOCK_TIMEOUT_S))
            lock_session = run_track_handoff_session(
                state=state,
                arm=arm,
                per=per,
                det=det,
                reject_cap=max(1, int(PICK_OTHER_MAX_REJECTS)),
                hard_timeout_s=float(lock_timeout_s),
                xy_margin_m=float(PLACE_VERIFY_V2_XY_MARGIN_M),
                z_margin_m=float(PLACE_VERIFY_V2_Z_MARGIN_M),
                blocked_track_id=None,
                blocked_xyz=None,
                blocked_track_ids_extra=blocked_extra_ids,
                blocked_xyzs_extra=None,
                blocked_uv=None,
                status_prefix=f"startup_lock_t{int(target_tid)}",
                log_prefix="",
                disable_stable_gate=False,
                on_locked_candidate=_startup_track_locked_candidate,
                required_track_id=(None if target_tid is None else int(target_tid)),
                centered_frames_required=int(max(1, STARTUP_STACK_LOCK_FRAMES)),
                close_window_on_exit=False,
                center_ey_scale=float(STARTUP_STACK_CENTER_EY_SCALE),
                refresh_required_track_timeout_on_visible=True,
                refresh_required_track_max_s=float(max(lock_timeout_s, STARTUP_STACK_LOCK_STAGE_TIMEOUT_S)),
            )
            lock_status = str(lock_session.get("status", "observe_retry"))
            lock_exit_reason = str(lock_session.get("exit_reason", "observe_retry"))
            item["last_lock_reason"] = str(lock_exit_reason)
            item["lock_source"] = "track_handoff_session"
            item["selected_track_id"] = (
                None
                if lock_session.get("selected_track_id", None) is None
                else int(lock_session.get("selected_track_id"))
            )
            if lock_status != "ok":
                if str(lock_exit_reason).strip().lower().startswith("duplicate_measurement"):
                    item["last_lock_reason"] = "duplicate_measurement"
                    item["exit_reason"] = "duplicate_measurement"
                    return "failed_duplicate_measurement"
                track_row_now = state.track_memory.get(int(target_tid), None) if TRACK_ENABLE else None
                if isinstance(track_row_now, dict) and int(track_row_now.get("miss_frames", 999)) == 0:
                    item["last_seen_ts"] = float(time.time())
                return "retry"

            lock_uv = lock_session.get("centered_pos", None)
            lock_track_id = lock_session.get("selected_track_id", None)
            lock_xyz = lock_session.get("selected_xyz", None)
            if not (isinstance(lock_uv, (list, tuple)) and len(lock_uv) >= 2):
                item["last_lock_reason"] = "locked_without_uv"
                return "retry"
            measured_norm = _finite_xyz_or_none(lock_xyz)
            if measured_norm is None:
                item["last_lock_reason"] = "locked_without_xyz"
                return "retry"
            measure_hits = int((lock_session.get("accept_payload") or {}).get("hits", 0))
            section_name = str((lock_session.get("accept_payload") or {}).get("section", "")).strip().lower()
            if section_name not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
                section_name_xy, _ = _infer_section_for_place_xy(
                    float(measured_norm[0]),
                    float(measured_norm[1]),
                    section_centers_xy,
                    band_min=None,
                    band_max=None,
                    max_center_dist_m=float(STARTUP_STACK_LOCK_ASSIGN_XY_MARGIN_M),
                )
                section_name = (
                    ""
                    if section_name_xy is None
                    else str(section_name_xy).strip().lower()
                )
            if section_name not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
                item["last_lock_reason"] = "unassigned_section"
                return "retry"

            measured_arr = np.array(measured_norm, dtype=float).reshape(-1)
            is_duplicate = False
            for prev_xyz in list(accepted_xyzs):
                if prev_xyz.size < 3 or not np.all(np.isfinite(prev_xyz[:3])):
                    continue
                d_xy_prev = float(
                    math.hypot(float(measured_arr[0]) - float(prev_xyz[0]), float(measured_arr[1]) - float(prev_xyz[1]))
                )
                d_z_prev = float(abs(float(measured_arr[2]) - float(prev_xyz[2])))
                if d_xy_prev <= float(dedup_xy_m) and d_z_prev <= float(dedup_z_m):
                    is_duplicate = True
                    break
            if is_duplicate:
                item["last_lock_reason"] = "duplicate_measurement"
                return "retry"

            final_tid = (None if lock_track_id is None else int(lock_track_id))
            if final_tid is None:
                final_tid = (None if target_tid is None else int(target_tid))
            obs_color = observe_scene_frame(
                det=det,
                arm=arm,
                per=per,
                draw=False,
                projected_min_conf=float(TRACK_MIN_CONF),
                state=state,
                update_tracks=True,
            )
            color_name = "unknown"
            color_conf = 0.0
            if obs_color is not None:
                candidate = choose_candidate_near_uv(
                    obs_color.candidates,
                    int(lock_uv[0]),
                    int(lock_uv[1]),
                    min_conf=0.0,
                )
                bbox_candidate = (
                    candidate.get("bbox_xyxy", None)
                    if isinstance(candidate, dict)
                    else None
                )
                if bbox_candidate is not None:
                    color_name_raw, color_conf = classify_cube_color_patch(
                        obs_color.image_bgr,
                        bbox_xyxy=bbox_candidate,
                        center_uv=None,
                        bbox_core_ratio=0.55,
                    )
                    color_name = str(color_name_raw).strip().lower()
                    if color_name not in {"orange", "blue"} or float(color_conf) < float(STARTUP_STACK_COLOR_MIN_CONF):
                        color_name = "unknown"
                if final_tid is None and isinstance(candidate, dict):
                    raw_tid = candidate.get("track_id", None)
                    try:
                        final_tid = None if raw_tid is None else int(raw_tid)
                    except (TypeError, ValueError):
                        final_tid = None
            if final_tid is None and TRACK_ENABLE:
                linked_tid = nearest_visible_track_by_uv(state, int(lock_uv[0]), int(lock_uv[1]), max_dist_px=90.0)
                if linked_tid is not None:
                    final_tid = int(linked_tid)
            if final_tid is not None:
                processed_track_ids.add(int(final_tid))
                item["final_track_id"] = int(final_tid)
            accepted_xyzs.append(measured_arr[:3].copy())
            item["last_seen_ts"] = float(time.time())
            item["last_lock_reason"] = "committed"
            item["exit_reason"] = "track_committed"
            if bool(STARTUP_STACK_ASSIGN_DEBUG):
                pass

            entries_by_section[str(section_name)].append(
                {
                    "track_id": (None if final_tid is None else int(final_tid)),
                    "uv": [int(lock_uv[0]), int(lock_uv[1])],
                    "xyz": list(measured_norm),
                    "color": str(color_name),
                    "conf": float(color_conf),
                    "order_index": int(row_i),
                    "source": "lock_commit",
                    "lock_source": "track_handoff_session",
                    "target_track_id": (None if target_tid is None else int(target_tid)),
                    "selected_track_id": (None if final_tid is None else int(final_tid)),
                }
            )
            return "committed"

        max_track_attempts = max(1, int(STARTUP_STACK_MAX_TRACK_ATTEMPTS))
        max_visibility_defer_checks = max(1, int(STARTUP_STACK_VISIBILITY_DEFER_CHECKS))
        max_defer_passes = 2

        def _choose_next_pending_checklist_index(
            *,
            deferred_tid_set: set[int] | None = None,
        ) -> int | None:
            pending_indices = [
                i
                for i, item in enumerate(checklist_rows)
                if str(item.get("status", "pending")) in {"pending", "in_progress"}
            ]
            if deferred_tid_set:
                pending_filtered: list[int] = []
                for idx in pending_indices:
                    tid_i = checklist_rows[idx].get("track_id", None)
                    if tid_i is None:
                        pending_filtered.append(int(idx))
                        continue
                    try:
                        if int(tid_i) in deferred_tid_set:
                            continue
                    except Exception:
                        pass
                    pending_filtered.append(int(idx))
                pending_indices = list(pending_filtered)
            if not pending_indices:
                return None
            obs_pick = observe_scene_frame(
                det=det,
                arm=arm,
                per=per,
                draw=False,
                projected_min_conf=float(conf_min),
                state=state,
                update_tracks=True,
            )
            _append_visible_targets_from_observation(obs_pick)
            visible_rows, _ = _collect_visible_targets(obs_pick)
            closest_rank_by_tid: dict[int, int] = {}
            for rank_i, row_i in enumerate(visible_rows):
                try:
                    tid_i = int(row_i.get("track_id"))
                except Exception:
                    continue
                if tid_i not in closest_rank_by_tid:
                    closest_rank_by_tid[int(tid_i)] = int(rank_i)
            ranked_pending = []
            for idx in pending_indices:
                item = checklist_rows[idx]
                tid_item = item.get("track_id", None)
                if tid_item is None:
                    rank_tid = int(10**9)
                else:
                    try:
                        rank_tid = int(closest_rank_by_tid.get(int(tid_item), int(10**9)))
                    except Exception:
                        rank_tid = int(10**9)
                ranked_pending.append((int(rank_tid), int(item.get("row_index", idx)), int(idx)))
            ranked_pending.sort(key=lambda row: (int(row[0]), int(row[1]), int(row[2])))
            return int(ranked_pending[0][2]) if ranked_pending else None

        defer_pass_index = 0
        deferred_tids_pass0: set[int] = set()
        hydrate_early_exit_logged = False

        def _maybe_exit_hydrate_when_sides_full() -> bool:
            nonlocal hydrate_early_exit_logged
            if not _startup_hydrate_should_exit_when_sides_full(entries_by_section):
                return False
            if not bool(hydrate_early_exit_logged):
                left_n = int(len(entries_by_section.get(SECTION_LEFT_NAME, [])))
                right_n = int(len(entries_by_section.get(SECTION_RIGHT_NAME, [])))
                print(
                    f"[StartupHydrateEarlyExit] reason=per_side_cap_reached "
                    f"left={left_n} right={right_n}"
                )
                hydrate_early_exit_logged = True
            return True

        while True:
            if _maybe_exit_hydrate_when_sides_full():
                break
            deferred_filter = deferred_tids_pass0 if int(defer_pass_index) == 0 else set()
            active_index = _choose_next_pending_checklist_index(deferred_tid_set=deferred_filter)
            if active_index is None:
                if _maybe_exit_hydrate_when_sides_full():
                    break
                # If checklist seeding missed a transiently visible target, take one more
                # live snapshot before declaring there is nothing to attempt.
                if int(len(checklist_rows)) < int(max_cubes_total):
                    obs_refill = observe_scene_frame(
                        det=det,
                        arm=arm,
                        per=per,
                        draw=False,
                        projected_min_conf=float(conf_min),
                        state=state,
                        update_tracks=True,
                    )
                    added_refill = _append_visible_targets_from_observation(obs_refill)
                    if int(added_refill) > 0:
                        continue
                if int(defer_pass_index) + 1 < int(max_defer_passes) and deferred_tids_pass0:
                    if _maybe_exit_hydrate_when_sides_full():
                        break
                    defer_pass_index += 1
                    continue
                break
            active_item = checklist_rows[active_index]
            active_tid = active_item.get("track_id", None)
            state.active_target_track_id = (None if active_tid is None else int(active_tid))
            active_item["status"] = "in_progress"

            committed_now = False
            final_reason = ""
            active_item["visibility_conf"] = 0.0
            active_item["active_track_id"] = (None if active_tid is None else int(active_tid))
            # Let startup stage budget raise per-target timeout when requested so
            # transient visibility/track churn does not prematurely skip a cube.
            per_target_timeout_s = float(
                max(
                    max(0.8, STARTUP_STACK_LOCK_TIMEOUT_S) * float(max_track_attempts),
                    float(max(0.8, STARTUP_STACK_LOCK_STAGE_TIMEOUT_S)),
                )
            )
            target_t0 = float(time.time())
            attempt_i = 0
            invisible_checks = 0
            while True:
                if active_tid is None:
                    final_reason = "track_missing"
                    break

                visible_now, vis_conf = _check_target_visible_now(int(active_tid))
                active_item["visibility_conf"] = float(vis_conf)
                if not bool(visible_now):
                    invisible_checks += 1
                    _pump_startup_preview(
                        last_visibility_obs,
                        f"startup_wait_visible track={active_tid} "
                        f"conf={float(vis_conf):.2f} "
                        f"miss={int(invisible_checks)}/{int(max_visibility_defer_checks)}",
                    )
                    if int(invisible_checks) >= int(max_visibility_defer_checks):
                        if int(defer_pass_index) == 0:
                            final_reason = "temporarily_not_visible_deferred"
                        else:
                            final_reason = "not_currently_visible"
                        break
                    if (time.time() - float(target_t0)) >= float(per_target_timeout_s):
                        final_reason = "not_currently_visible"
                        break
                    active_item["last_lock_reason"] = "not_currently_visible_waiting"
                    time.sleep(float(max(0.0, arm.sample_time)))
                    continue
                invisible_checks = 0
                if (time.time() - float(target_t0)) >= float(per_target_timeout_s):
                    final_reason = "visible_unresolved_timeout"
                    break

                attempt_i += 1
                active_item["attempts"] = int(attempt_i)
                attempt_outcome = _attempt_startup_checklist_item(active_item)
                if str(attempt_outcome) == "committed":
                    active_item["status"] = "committed"
                    active_item["exit_reason"] = str(active_item.get("exit_reason") or "track_committed")
                    checklist_committed += 1
                    committed_now = True
                    break

                if str(attempt_outcome) == "failed_no_track_id":
                    final_reason = "track_missing"
                    break
                if str(attempt_outcome) == "failed_duplicate_measurement":
                    final_reason = "duplicate_measurement"
                    break

                last_lock_reason = str(active_item.get("last_lock_reason", "")).strip().lower()
                if last_lock_reason in {"no_track_candidate_timeout", "required_track_not_reacquired"}:
                    final_reason = "required_track_not_reacquired"
                elif last_lock_reason.startswith("duplicate_measurement"):
                    final_reason = "duplicate_measurement"
                elif last_lock_reason.startswith("wrong_track_for_item"):
                    final_reason = "wrong_track_for_item"
                elif last_lock_reason.startswith("unassigned_section"):
                    final_reason = "unassigned_section"
                elif last_lock_reason:
                    final_reason = str(last_lock_reason)
                else:
                    final_reason = "lock_failed"

                if int(attempt_i) >= int(max_track_attempts):
                    if not final_reason:
                        final_reason = "max_attempts_reached"
                    break
                time.sleep(float(max(0.0, arm.sample_time)))

            if committed_now:
                if _maybe_exit_hydrate_when_sides_full():
                    break
                continue

            if str(final_reason) == "temporarily_not_visible_deferred":
                active_item["status"] = "pending"
                active_item["exit_reason"] = str(final_reason)
                if bool(STARTUP_STACK_ASSIGN_DEBUG):
                    print(
                        f"[StartupHydrateDefer] pass={int(defer_pass_index)+1}/{int(max_defer_passes)} "
                        f"track={active_tid} reason={final_reason}"
                    )
                if active_tid is not None and int(defer_pass_index) == 0:
                    try:
                        deferred_tids_pass0.add(int(active_tid))
                    except Exception:
                        pass
                continue

            active_item["status"] = "failed"
            active_item["exit_reason"] = str(final_reason or "visible_unresolved_timeout")
            if (
                int(defer_pass_index) >= 1
                and str(final_reason) == "not_currently_visible"
                and active_tid is not None
            ):
                try:
                    was_deferred_pass0 = int(active_tid) in deferred_tids_pass0
                except Exception:
                    was_deferred_pass0 = False
                if bool(was_deferred_pass0):
                    print(
                        f"[StartupHydrateRetryMiss] pass={int(defer_pass_index)+1}/{int(max_defer_passes)} "
                        f"track={active_tid} reason={final_reason} "
                        f"best_visibility_conf={float(active_item.get('visibility_conf', 0.0) or 0.0):.2f}"
                    )
            checklist_failed += 1
            lock_skipped_count += 1
            reason_key = str(active_item["exit_reason"])
            skip_reasons[reason_key] = int(skip_reasons.get(reason_key, 0)) + 1

        state.active_target_track_id = None

        if bool(STARTUP_REFRESH_PASS_ENABLED) and not _startup_hydrate_should_exit_when_sides_full(
            entries_by_section
        ):
            refresh_added_tids: list[int] = []
            obs_refresh = observe_scene_frame(
                det=det,
                arm=arm,
                per=per,
                draw=False,
                projected_min_conf=float(conf_min),
                state=state,
                update_tracks=True,
            )
            if obs_refresh is not None:
                visible_rows, _ = _collect_visible_targets(obs_refresh)
                visible_rows.sort(key=lambda r: float(r.get("xyz", [0.0, 0.0, 0.0])[2]))
                for row in visible_rows:
                    try:
                        tid = int(row.get("track_id"))
                    except Exception:
                        continue
                    if tid in processed_track_ids:
                        continue
                    if tid in checklist_tid_set:
                        continue
                    tid_i = int(tid)
                    checklist_tid_set.add(int(tid_i))
                    row_i = int(len(checklist_rows))
                    checklist_rows.append(
                        {
                            "key": f"track:{tid_i}",
                            "track_id": int(tid_i),
                            "expected_track_id_initial": int(tid_i),
                            "final_track_id": None,
                            "rebound": False,
                            "rebind_reason": "",
                            "row_index": int(row_i),
                            "row": dict(row),
                            "status": "pending",
                            "attempts": 0,
                            "last_seen_ts": None,
                            "last_lock_reason": "pending",
                            "exit_reason": "",
                        }
                    )
                    refresh_added_tids.append(int(tid_i))

            if refresh_added_tids:
                print(f"[StartupHydrateRefresh] added_tids={refresh_added_tids}")
                for tid in list(refresh_added_tids):
                    idx_match = next(
                        (
                            i
                            for i, it in enumerate(checklist_rows)
                            if int(it.get("track_id", -1)) == int(tid)
                            and str(it.get("status", "")) == "pending"
                        ),
                        None,
                    )
                    if idx_match is None:
                        continue
                    active_item = checklist_rows[int(idx_match)]
                    active_tid = int(tid)
                    state.active_target_track_id = int(active_tid)
                    active_item["status"] = "in_progress"
                    try:
                        attempt_outcome = _attempt_startup_checklist_item(active_item)
                        if str(attempt_outcome) == "committed":
                            active_item["status"] = "committed"
                            checklist_committed += 1
                        else:
                            active_item["status"] = "failed"
                            active_item["fail_reason"] = str(
                                active_item.get("exit_reason", "") or attempt_outcome
                            )
                            checklist_failed += 1
                    except Exception as exc:
                        active_item["status"] = "failed"
                        active_item["fail_reason"] = f"refresh_attempt_exception:{exc}"
                        print(f"[StartupHydrateRefresh] tid={tid} exception={exc}")
                        checklist_failed += 1
            state.active_target_track_id = None

        # Track-first mode: only lock-committed measured entries populate startup hydrated stacks.

        lock_failed_rows = int(checklist_failed)

        z_predict_stats: dict[str, dict[str, int]] = {
            SECTION_LEFT_NAME: {"expected": 0, "committed_before": 0, "recovered": 0},
            SECTION_RIGHT_NAME: {"expected": 0, "committed_before": 0, "recovered": 0},
        }

        def _startup_section_stack_base_z(section_name: str) -> float:
            slots_local = get_place_slots()
            groups_local = section_slot_groups(slots_local)
            idxs_local = list(groups_local.get(str(section_name), []))
            if not idxs_local:
                return float(STACK_RELEASE_Z_GUARD_M)
            base_xyz_local = slot_target_xyz(int(idxs_local[0]))
            if base_xyz_local.size < 3 or not np.isfinite(float(base_xyz_local[2])):
                return float(STACK_RELEASE_Z_GUARD_M)
            return float(max(float(base_xyz_local[2]), float(STACK_RELEASE_Z_GUARD_M)))

        def _startup_layer_index_from_z(z_m: float, stack_base_z: float) -> int:
            dz_local = max(1e-6, float(STACK_LEVEL_DZ_M))
            return int(round((float(z_m) - float(stack_base_z)) / dz_local))

        def _startup_classify_color_at_uv(obs_row: SceneObservation | None, uv: list[int]) -> tuple[str, float]:
            if obs_row is None:
                return "unknown", 0.0
            candidate = choose_candidate_near_uv(
                obs_row.candidates,
                int(uv[0]),
                int(uv[1]),
                min_conf=0.0,
            )
            bbox_candidate = (
                candidate.get("bbox_xyxy", None)
                if isinstance(candidate, dict)
                else None
            )
            if bbox_candidate is None:
                return "unknown", 0.0
            color_name_raw, color_conf_local = classify_cube_color_patch(
                obs_row.image_bgr,
                bbox_xyxy=bbox_candidate,
                center_uv=None,
                bbox_core_ratio=0.55,
            )
            color_name_local = str(color_name_raw).strip().lower()
            if color_name_local not in {"orange", "blue"} or float(color_conf_local) < float(
                STARTUP_STACK_COLOR_MIN_CONF
            ):
                return "unknown", float(color_conf_local)
            return color_name_local, float(color_conf_local)

        def _startup_xyz_duplicate_of_accepted(xyz_arr: np.ndarray) -> bool:
            for prev_xyz in list(accepted_xyzs):
                if prev_xyz.size < 3 or not np.all(np.isfinite(prev_xyz[:3])):
                    continue
                d_xy_prev = float(
                    math.hypot(float(xyz_arr[0]) - float(prev_xyz[0]), float(xyz_arr[1]) - float(prev_xyz[1]))
                )
                d_z_prev = float(abs(float(xyz_arr[2]) - float(prev_xyz[2])))
                if d_xy_prev <= float(dedup_xy_m) and d_z_prev <= float(dedup_z_m):
                    return True
            return False

        if bool(STARTUP_STACK_Z_PREDICT_ENABLED):
            layer_xy_m = float(max(0.005, STARTUP_STACK_LAYER_MATCH_XY_M))
            layer_z_m = float(max(0.005, STARTUP_STACK_LAYER_MATCH_Z_M))
            scan_frames = max(1, int(STARTUP_STACK_LAYER_SCAN_FRAMES))
            vote_min_hits = max(1, min(int(scan_frames), int(STARTUP_STACK_LAYER_VOTE_MIN_HITS)))
            max_layers_side = max(1, int(STARTUP_STACK_MAX_CUBES_PER_SIDE))
            for section_name in [SECTION_LEFT_NAME, SECTION_RIGHT_NAME]:
                entries_now = [
                    row
                    for row in list(entries_by_section.get(section_name, []))
                    if isinstance(row.get("xyz", None), list) and len(row.get("xyz", [])) >= 3
                ]
                entries_now.sort(key=lambda row: float(row.get("xyz", [0.0, 0.0, 0.0])[2]))
                committed_before = int(len(entries_now))
                z_predict_stats[str(section_name)]["committed_before"] = int(committed_before)
                if committed_before <= 0:
                    continue
                top_entry = entries_now[-1]
                top_xyz = np.array(top_entry.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
                if top_xyz.size < 3 or not np.all(np.isfinite(top_xyz[:3])):
                    continue
                slot_base_z = _startup_section_stack_base_z(str(section_name))
                dz_step = max(1e-6, float(STACK_LEVEL_DZ_M))
                expected_layers = int(round((float(top_xyz[2]) - float(slot_base_z)) / dz_step)) + 1
                expected_layers = max(1, min(int(max_layers_side), int(expected_layers)))
                z_predict_stats[str(section_name)]["expected"] = int(expected_layers)
                if int(expected_layers) <= int(committed_before):
                    continue
                anchor_xy = (float(top_xyz[0]), float(top_xyz[1]))
                lowest_z = float(entries_now[0].get("xyz", [0.0, 0.0, 0.0])[2])
                lowest_committed_layer = max(0, int(expected_layers) - int(committed_before))
                stack_base_z = float(lowest_z) - float(lowest_committed_layer) * float(dz_step)
                occupied_layers: set[int] = set(
                    range(int(lowest_committed_layer), int(expected_layers))
                )
                missing_layers = [
                    int(layer_i)
                    for layer_i in range(int(expected_layers))
                    if int(layer_i) not in occupied_layers
                ]
                print(
                    f"[StartupHydrateLayerAnchor] side={section_name} "
                    f"expected={int(expected_layers)} committed={int(committed_before)} "
                    f"top_z={float(top_xyz[2]):.3f} lowest_z={float(lowest_z):.3f} "
                    f"slot_base_z={float(slot_base_z):.3f} measured_base_z={float(stack_base_z):.3f} "
                    f"missing={missing_layers}"
                )
                use_side_full_rescan = bool(
                    STARTUP_STACK_SIDE_FULL_RESCAN_ENABLED
                    and int(expected_layers) >= int(STARTUP_STACK_SIDE_FULL_RESCAN_MIN_EXPECTED)
                    and bool(missing_layers)
                )
                if bool(use_side_full_rescan):
                    rescan_frames = max(1, int(STARTUP_STACK_SIDE_FULL_RESCAN_FRAMES))
                    rescan_vote_min = max(1, min(int(rescan_frames), int(vote_min_hits)))
                    recovered_rescan = _startup_side_full_rescan_top_to_bottom(
                        section_name=str(section_name),
                        expected_layers=int(expected_layers),
                        missing_layers=list(missing_layers),
                        anchor_xy=(float(anchor_xy[0]), float(anchor_xy[1])),
                        stack_base_z=float(stack_base_z),
                        entries_by_section=entries_by_section,
                        section_centers_xy=section_centers_xy,
                        processed_track_ids=processed_track_ids,
                        accepted_xyzs=accepted_xyzs,
                        det=det,
                        arm=arm,
                        per=per,
                        state=state,
                        conf_min=float(conf_min),
                        startup_target_min_conf=float(startup_target_min_conf),
                        layer_xy_m=float(layer_xy_m),
                        layer_z_m=float(layer_z_m),
                        rescan_frames=int(rescan_frames),
                        vote_min_hits=int(rescan_vote_min),
                        dz_step=float(dz_step),
                        observe_scene_frame_fn=observe_scene_frame,
                        classify_color_at_uv_fn=_startup_classify_color_at_uv,
                        xyz_duplicate_fn=_startup_xyz_duplicate_of_accepted,
                        infer_section_fn=_infer_section_for_place_xy,
                    )
                    z_predict_stats[str(section_name)]["recovered"] = int(
                        z_predict_stats[str(section_name)].get("recovered", 0)
                    ) + int(recovered_rescan)
                    continue
                for layer_idx in missing_layers:
                    predicted_xyz = np.array(
                        [
                            float(anchor_xy[0]),
                            float(anchor_xy[1]),
                            float(stack_base_z + int(layer_idx) * dz_step),
                        ],
                        dtype=float,
                    )
                    vote_hits: list[dict] = []
                    near_miss_outside_gate = 0
                    near_miss_processed_tid = 0
                    near_miss_wrong_section = 0
                    section_candidate_count = 0
                    for _scan_i in range(int(scan_frames)):
                        obs_scan = observe_scene_frame(
                            det=det,
                            arm=arm,
                            per=per,
                            draw=False,
                            projected_min_conf=float(conf_min),
                            state=state,
                            update_tracks=True,
                        )
                        if obs_scan is None:
                            time.sleep(float(max(0.0, arm.sample_time)))
                            continue
                        for prow in list(obs_scan.projected_rows):
                            conf_i = float(prow.get("conf", 0.0))
                            if conf_i < float(startup_target_min_conf):
                                continue
                            raw_tid = prow.get("track_id", None)
                            try:
                                tid_i = None if raw_tid is None else int(raw_tid)
                            except Exception:
                                tid_i = None
                            xyz_i = np.array(
                                prow.get("xyz", [np.nan, np.nan, np.nan]), dtype=float
                            ).reshape(-1)
                            if xyz_i.size < 3 or not np.all(np.isfinite(xyz_i[:3])):
                                continue
                            section_name_xy, _assign_info = _infer_section_for_place_xy(
                                float(xyz_i[0]),
                                float(xyz_i[1]),
                                section_centers_xy,
                                band_min=None,
                                band_max=None,
                                max_center_dist_m=float(STARTUP_STACK_LOCK_ASSIGN_XY_MARGIN_M),
                            )
                            section_norm_xy = (
                                ""
                                if section_name_xy is None
                                else str(section_name_xy).strip().lower()
                            )
                            if section_norm_xy != str(section_name):
                                near_miss_wrong_section += 1
                                continue
                            section_candidate_count += 1
                            if tid_i is not None and int(tid_i) in processed_track_ids:
                                near_miss_processed_tid += 1
                                continue
                            d_xy = float(
                                math.hypot(
                                    float(xyz_i[0]) - float(predicted_xyz[0]),
                                    float(xyz_i[1]) - float(predicted_xyz[1]),
                                )
                            )
                            d_z = float(abs(float(xyz_i[2]) - float(predicted_xyz[2])))
                            if d_xy > float(layer_xy_m) or d_z > float(layer_z_m):
                                near_miss_outside_gate += 1
                                continue
                            vote_hits.append(
                                {
                                    "prow": dict(prow),
                                    "obs": obs_scan,
                                    "d_xy": float(d_xy),
                                    "d_z": float(d_z),
                                    "xyz": xyz_i[:3].copy(),
                                }
                            )
                        time.sleep(float(max(0.0, arm.sample_time)))
                    print(
                        f"[StartupHydrateLayerNearMiss] side={section_name} layer={int(layer_idx)} "
                        f"section_candidates={int(section_candidate_count)} "
                        f"outside_gate={int(near_miss_outside_gate)} "
                        f"already_processed_tid={int(near_miss_processed_tid)} "
                        f"wrong_section={int(near_miss_wrong_section)} "
                        f"gate_xy={float(layer_xy_m):.3f} gate_z={float(layer_z_m):.3f}"
                    )
                    if len(vote_hits) < int(vote_min_hits):
                        print(
                            f"[StartupHydrateLayerVote] side={section_name} layer={int(layer_idx)} "
                            f"hits={len(vote_hits)}/{int(scan_frames)} min={int(vote_min_hits)} "
                            f"predicted_xyz={[float(predicted_xyz[0]), float(predicted_xyz[1]), float(predicted_xyz[2])]} "
                            f"reason=below_min_hits"
                        )
                        print(
                            f"[StartupHydrateLayerMissing] side={section_name} layer={int(layer_idx)} "
                            f"predicted_xyz={[float(predicted_xyz[0]), float(predicted_xyz[1]), float(predicted_xyz[2])]} "
                            f"reason=no_visible_candidate"
                        )
                        continue
                    xyz_stack = np.array([np.asarray(h["xyz"], dtype=float).reshape(3) for h in vote_hits])
                    measured_xyz = np.median(xyz_stack, axis=0).reshape(-1)
                    best_hit = min(
                        vote_hits,
                        key=lambda h: float(
                            math.hypot(
                                float(h["xyz"][0]) - float(measured_xyz[0]),
                                float(h["xyz"][1]) - float(measured_xyz[1]),
                            )
                            + abs(float(h["xyz"][2]) - float(measured_xyz[2]))
                        ),
                    )
                    best_match = dict(best_hit["prow"])
                    best_obs = best_hit["obs"]
                    print(
                        f"[StartupHydrateLayerVote] side={section_name} layer={int(layer_idx)} "
                        f"hits={len(vote_hits)}/{int(scan_frames)} min={int(vote_min_hits)} "
                        f"predicted_xyz={[float(predicted_xyz[0]), float(predicted_xyz[1]), float(predicted_xyz[2])]} "
                        f"median_xyz={[float(measured_xyz[0]), float(measured_xyz[1]), float(measured_xyz[2])]}"
                    )
                    if _startup_xyz_duplicate_of_accepted(measured_xyz):
                        print(
                            f"[StartupHydrateLayerMissing] side={section_name} layer={int(layer_idx)} "
                            f"predicted_xyz={[float(predicted_xyz[0]), float(predicted_xyz[1]), float(predicted_xyz[2])]} "
                            f"reason=duplicate_of_committed"
                        )
                        continue
                    uv_match = [
                        int(best_match.get("u", 0)),
                        int(best_match.get("v", 0)),
                    ]
                    color_name, color_conf = _startup_classify_color_at_uv(best_obs, uv_match)
                    final_tid = None
                    raw_tid = best_match.get("track_id", None)
                    try:
                        final_tid = None if raw_tid is None else int(raw_tid)
                    except Exception:
                        final_tid = None
                    if final_tid is None:
                        linked_tid = nearest_visible_track_by_uv(
                            state, int(uv_match[0]), int(uv_match[1]), max_dist_px=90.0
                        )
                        if linked_tid is not None:
                            final_tid = int(linked_tid)
                    if final_tid is not None:
                        processed_track_ids.add(int(final_tid))
                    accepted_xyzs.append(measured_xyz[:3].copy())
                    entries_by_section[str(section_name)].append(
                        {
                            "track_id": final_tid,
                            "uv": list(uv_match),
                            "xyz": [
                                float(measured_xyz[0]),
                                float(measured_xyz[1]),
                                float(measured_xyz[2]),
                            ],
                            "color": str(color_name),
                            "conf": float(color_conf),
                            "order_index": int(len(entries_by_section[str(section_name)])),
                            "source": "layer_z_predict",
                            "lock_source": "layer_z_predict",
                            "target_track_id": final_tid,
                            "selected_track_id": final_tid,
                        }
                    )
                    z_predict_stats[str(section_name)]["recovered"] = int(
                        z_predict_stats[str(section_name)].get("recovered", 0)
                    ) + 1
                    print(
                        f"[StartupHydrateLayerRecovered] side={section_name} layer={int(layer_idx)} "
                        f"predicted_xyz={[float(predicted_xyz[0]), float(predicted_xyz[1]), float(predicted_xyz[2])]} "
                        f"measured_xyz={[float(measured_xyz[0]), float(measured_xyz[1]), float(measured_xyz[2])]} "
                        f"track={final_tid} color={color_name}"
                    )

        for section_name in [SECTION_LEFT_NAME, SECTION_RIGHT_NAME]:
            hydrated_entries = [
                row for row in list(entries_by_section.get(section_name, []))
                if isinstance(row.get("xyz", None), list) and len(row.get("xyz", [])) >= 3
            ]
            hydrated_entries.sort(key=lambda row: float(row.get("xyz", [0.0, 0.0, 0.0])[2]))
            max_cubes_side = max(1, int(STARTUP_STACK_MAX_CUBES_PER_SIDE))
            hydrated_entries = hydrated_entries[:max_cubes_side]
            seq = [str(row.get("color", "unknown")).strip().lower() for row in hydrated_entries]
            seq = [c if c in {"orange", "blue"} else "unknown" for c in seq]
            tracks = [
                (None if row.get("track_id", None) is None else int(row.get("track_id")))
                for row in hydrated_entries
            ]
            top_color = (seq[-1] if seq else "unknown")
            level_i = int(len(hydrated_entries))
            seq_pad = list(seq)
            if len(seq_pad) < int(level_i):
                seq_pad.extend(["unknown"] * int(level_i - len(seq_pad)))
            pos_bottom = (seq_pad[0] if level_i >= 1 and len(seq_pad) >= 1 else "empty")
            pos_middle = (seq_pad[1] if level_i >= 2 and len(seq_pad) >= 2 else "empty")
            pos_top = (seq_pad[2] if level_i >= 3 and len(seq_pad) >= 3 else "empty")
            section_row = {
                "stack_level": int(level_i),
                "top_color": str(top_color),
                "color_sequence_bottom_to_top": list(seq),
                "tracks_bottom_to_top": list(tracks),
                "entries": list(hydrated_entries),
            }
            if bool(STARTUP_STACK_ASSIGN_DEBUG):
                for entry_i, entry_row in enumerate(list(hydrated_entries)):
                    print(
                        f"[StartupHydrateEntry] side={section_name} idx={int(entry_i)+1}/{int(len(hydrated_entries))} "
                        f"track={entry_row.get('track_id', None)} xyz={_format_xyz_log_3(entry_row.get('xyz', None))} "
                        f"uv={entry_row.get('uv', None)} color={entry_row.get('color', 'unknown')} "
                        f"color_conf={float(entry_row.get('conf', 0.0)):.2f} "
                        f"source={entry_row.get('source', 'unknown')}"
                    )
            summary["sections"][section_name] = dict(section_row)
            summary["observed_stack_levels"][section_name] = int(section_row["stack_level"])
            summary["observed_sequences"][section_name] = list(section_row["color_sequence_bottom_to_top"])
            print(
                f"[StartupHydrate] {section_name} level={int(section_row['stack_level'])} "
                f"base={pos_bottom} middle={pos_middle} top={pos_top} "
                f"seq={section_row['color_sequence_bottom_to_top']} "
                f"tracks={section_row['tracks_bottom_to_top']}"
            )

        left_zs = z_predict_stats.get(SECTION_LEFT_NAME, {})
        right_zs = z_predict_stats.get(SECTION_RIGHT_NAME, {})
        print(
            f"[StartupHydrateZCheck] left expected={int(left_zs.get('expected', 0))} "
            f"committed_before={int(left_zs.get('committed_before', 0))} "
            f"recovered={int(left_zs.get('recovered', 0))} | "
            f"right expected={int(right_zs.get('expected', 0))} "
            f"committed_before={int(right_zs.get('committed_before', 0))} "
            f"recovered={int(right_zs.get('recovered', 0))}"
        )
        expected_stack_levels: dict[str, int] = {}
        expected_shortfall_sides: list[str] = []
        observed_lock_stack_levels = dict(summary["observed_stack_levels"])
        for section_name in [SECTION_LEFT_NAME, SECTION_RIGHT_NAME]:
            z_stats = z_predict_stats.get(str(section_name), {})
            observed_level = int(observed_lock_stack_levels.get(str(section_name), 0) or 0)
            expected_level = max(observed_level, int(z_stats.get("expected", 0) or 0))
            expected_stack_levels[str(section_name)] = int(expected_level)
            if int(expected_level) > int(observed_level):
                expected_shortfall_sides.append(str(section_name))
                section_row = dict(summary["sections"].get(str(section_name), _startup_default_hydrated_section_row()))
                seq_existing = [
                    c if str(c).strip().lower() in {"orange", "blue", "unknown"} else "unknown"
                    for c in list(section_row.get("color_sequence_bottom_to_top", []))
                ]
                missing_count = int(max(0, int(expected_level) - int(observed_level)))
                # Top Z implies a taller stack than we could separate. Preserve measured top layers
                # and infer unresolved lower-layer colors from the cube directly above them.
                seq_expected = (["unknown"] * int(missing_count) + list(seq_existing))[: int(expected_level)]
                inferred_rows: list[tuple[int, int, str]] = []
                for layer_i in range(len(seq_expected) - 2, -1, -1):
                    if str(seq_expected[layer_i]).strip().lower() != "unknown":
                        continue
                    above_color = str(seq_expected[layer_i + 1]).strip().lower()
                    if above_color not in {"orange", "blue"}:
                        continue
                    seq_expected[layer_i] = above_color
                    inferred_rows.append((int(layer_i), int(layer_i + 1), str(above_color)))
                for layer_i, above_i, inferred_color in reversed(inferred_rows):
                    print(
                        f"[StartupHydrateInferredColor] side={section_name} "
                        f"layer={int(layer_i)} inferred={inferred_color} "
                        f"from_layer={int(above_i)} reason=unknown_obfuscated_by_above"
                    )
                tracks_existing = list(section_row.get("tracks_bottom_to_top", []))
                tracks_expected = ([None] * int(missing_count) + list(tracks_existing))[: int(expected_level)]
                entries_existing = list(section_row.get("entries", []))
                section_row["stack_level"] = int(expected_level)
                section_row["top_color"] = str(seq_expected[-1] if seq_expected else "unknown")
                section_row["color_sequence_bottom_to_top"] = list(seq_expected)
                section_row["tracks_bottom_to_top"] = list(tracks_expected)
                section_row["entries"] = list(entries_existing)
                section_row["expected_shortfall_inferred_from_above"] = bool(inferred_rows)
                section_row["observed_lock_stack_level"] = int(observed_level)
                summary["sections"][str(section_name)] = dict(section_row)
                summary["observed_stack_levels"][str(section_name)] = int(expected_level)
                summary["observed_sequences"][str(section_name)] = list(seq_expected)
                print(
                    f"[StartupHydrateExpectedInferred] side={section_name} "
                    f"observed={int(observed_level)} expected={int(expected_level)} "
                    f"seq={list(seq_expected)}"
                )
        summary["expected_stack_levels"] = dict(expected_stack_levels)
        summary["expected_level_shortfall_sides"] = list(expected_shortfall_sides)
        summary["observed_lock_stack_levels"] = dict(observed_lock_stack_levels)
        if bool(expected_shortfall_sides):
            print(
                f"[StartupHydrateExpectedShortfall] expected={dict(expected_stack_levels)} "
                f"lock_committed={dict(observed_lock_stack_levels)} "
                f"effective={dict(summary['observed_stack_levels'])} "
                f"sides={list(expected_shortfall_sides)}"
            )
            # Effective stack state is now complete enough for planner safety: unresolved layers
            # are represented as occupied/unknown instead of empty.

        if all(int(summary["observed_stack_levels"].get(side, 0)) == 0 for side in [SECTION_LEFT_NAME, SECTION_RIGHT_NAME]):
            if int(rep_rows_count) > 0:
                summary["status"] = "lock_incomplete_no_commits"
            else:
                summary["status"] = "no_place_candidates"
        summary["rep_rows_count"] = int(rep_rows_count)
        summary["lock_failed_rows"] = int(lock_failed_rows)
        summary["visible_targets_count"] = int(len(checklist_rows))
        summary["lock_checklist_total"] = int(len(checklist_rows))
        summary["lock_checklist_committed"] = int(checklist_committed)
        summary["lock_checklist_failed"] = int(checklist_failed)
        summary["lock_committed_count"] = int(checklist_committed)
        summary["lock_skipped_count"] = int(lock_skipped_count)
        summary["skip_reasons"] = {str(k): int(v) for k, v in sorted(skip_reasons.items())}
        summary["track_resolution"] = [
            {
                "expected_track_id_initial": (
                    None
                    if item.get("expected_track_id_initial", None) is None
                    else int(item.get("expected_track_id_initial"))
                ),
                "track_id": (None if item.get("track_id", None) is None else int(item.get("track_id"))),
                "final_track_id": (
                    None
                    if item.get("final_track_id", None) is None
                    else int(item.get("final_track_id"))
                ),
                "rebound": bool(item.get("rebound", False)),
                "rebind_reason": str(item.get("rebind_reason", "")),
                "status": str(item.get("status", "unknown")),
                "attempt_count": int(item.get("attempts", 0)),
                "active_track_id": (
                    None
                    if item.get("active_track_id", None) is None
                    else int(item.get("active_track_id"))
                ),
                "visibility_conf": float(item.get("visibility_conf", 0.0) or 0.0),
                "last_seen_age_s": (
                    None
                    if item.get("last_seen_ts", None) is None
                    else float(max(0.0, time.time() - float(item.get("last_seen_ts"))))
                ),
                "exit_reason": str(item.get("exit_reason", "") or item.get("last_lock_reason", "unknown")),
            }
            for item in checklist_rows
        ]
        attempted_track_ids: set[int] = set()
        for item in checklist_rows:
            tid_item = item.get("track_id", None)
            if tid_item is not None:
                try:
                    attempted_track_ids.add(int(tid_item))
                except Exception:
                    pass
            tid_final = item.get("final_track_id", None)
            if tid_final is not None:
                try:
                    attempted_track_ids.add(int(tid_final))
                except Exception:
                    pass
        # Final fail-loud visibility gate:
        # if a track is visible at startup-hydrate end and we never attempted it, block readiness.
        obs_final_gate = observe_scene_frame(
            det=det,
            arm=arm,
            per=per,
            draw=False,
            projected_min_conf=float(conf_min),
            state=state,
            update_tracks=True,
        )
        final_visible_rows, _ = _collect_visible_targets(obs_final_gate)
        final_visible_track_ids: list[int] = []
        for row in final_visible_rows:
            try:
                tid_final_vis = int(row.get("track_id"))
            except Exception:
                continue
            final_visible_track_ids.append(int(tid_final_vis))
        unresolved_visible_track_ids = sorted(
            int(tid_i)
            for tid_i in set(final_visible_track_ids)
            if int(tid_i) not in attempted_track_ids
        )
        if unresolved_visible_track_ids:
            if _startup_hydrate_should_exit_when_sides_full(entries_by_section):
                print(
                    f"[StartupHydrateEarlyExit] ignoring_unresolved_visible_tracks="
                    f"{unresolved_visible_track_ids} reason=sides_at_cap"
                )
                summary["early_exit_unresolved_ignored"] = list(unresolved_visible_track_ids)
            else:
                summary["status"] = "visible_targets_unattempted"
                print(
                    f"[StartupHydrate] unresolved_visible_track_ids={unresolved_visible_track_ids} "
                    f"attempted={sorted(attempted_track_ids)}"
                )
        summary["attempted_track_ids"] = sorted(int(tid_i) for tid_i in attempted_track_ids)
        summary["final_visible_track_ids"] = sorted(int(tid_i) for tid_i in set(final_visible_track_ids))
        summary["unresolved_visible_track_ids"] = list(unresolved_visible_track_ids)
        summary["unresolved_visible_count"] = int(len(unresolved_visible_track_ids))
        summary["total_hydrated_count"] = int(
            summary["observed_stack_levels"].get(SECTION_LEFT_NAME, 0)
            + summary["observed_stack_levels"].get(SECTION_RIGHT_NAME, 0)
        )
        return summary
    finally:
        if bool(show_window) and bool(SHOW_WINDOW):
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


def apply_startup_stack_hydration(state: CycleState, startup_row: dict | None) -> dict[str, int]:
    _bind_core_globals()
    row = startup_row if isinstance(startup_row, dict) else {}
    hydrated_src = row.get("hydrated_stacks", None)
    if not isinstance(hydrated_src, dict):
        hydrated_src = {}
    sections_src = hydrated_src.get("sections", None)
    if not isinstance(sections_src, dict):
        sections_src = {}
    left_src = sections_src.get(SECTION_LEFT_NAME, _startup_default_hydrated_section_row())
    right_src = sections_src.get(SECTION_RIGHT_NAME, _startup_default_hydrated_section_row())
    left_row = _normalize_hydrated_section_row(left_src)
    right_row = _normalize_hydrated_section_row(right_src)
    state.startup_hydrated_sections = {
        SECTION_LEFT_NAME: dict(left_row),
        SECTION_RIGHT_NAME: dict(right_row),
    }
    levels = {
        SECTION_LEFT_NAME: int(left_row.get("stack_level", 0) or 0),
        SECTION_RIGHT_NAME: int(right_row.get("stack_level", 0) or 0),
    }
    for section_name, section_row in (
        (SECTION_LEFT_NAME, left_row),
        (SECTION_RIGHT_NAME, right_row),
    ):
        try:
            section_level = int(section_row.get("stack_level", 0) or 0)
        except Exception:
            section_level = 0
        if section_level <= 0:
            _clear_locked_stack_anchor_xyz(state, section_name)
            _clear_last_popped_xy(state, section_name)
            continue
        top_xyz: list[float] | None = None
        top_z = float("-inf")
        for entry in list(section_row.get("entries", [])):
            if not isinstance(entry, dict):
                continue
            xyz = _finite_xyz_or_none(entry.get("xyz", None))
            if xyz is None:
                continue
            try:
                z_val = float(xyz[2])
            except Exception:
                continue
            if (not np.isfinite(z_val)) or (z_val <= top_z):
                continue
            top_z = float(z_val)
            top_xyz = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        if top_xyz is not None:
            _set_locked_stack_anchor_xyz(state, section_name, top_xyz, "startup_hydrate_top")
        else:
            _clear_locked_stack_anchor_xyz(state, section_name)
    if not isinstance(state.last_begin_stack_verify, dict):
        state.last_begin_stack_verify = {}
    state.last_begin_stack_verify["hydrated_stacks"] = {
        "sections": {
            SECTION_LEFT_NAME: dict(left_row),
            SECTION_RIGHT_NAME: dict(right_row),
        },
        "observed_stack_levels": dict(levels),
        "observed_sequences": {
            SECTION_LEFT_NAME: list(left_row.get("color_sequence_bottom_to_top", [])),
            SECTION_RIGHT_NAME: list(right_row.get("color_sequence_bottom_to_top", [])),
        },
    }
    print(
        f"[StartupHydrate] stack_levels_synced "
        f"{SECTION_LEFT_NAME}={int(levels[SECTION_LEFT_NAME])} "
        f"{SECTION_RIGHT_NAME}={int(levels[SECTION_RIGHT_NAME])}"
    )
    left_seq_dbg = list(left_row.get("color_sequence_bottom_to_top", []))
    right_seq_dbg = list(right_row.get("color_sequence_bottom_to_top", []))
    print(
        f"[StartupHydrate] color_persist "
        f"{SECTION_LEFT_NAME}={left_seq_dbg} "
        f"{SECTION_RIGHT_NAME}={right_seq_dbg}"
    )
    return levels


def get_startup_hydrated_section_row(state: CycleState, section_name: str) -> dict:
    _bind_core_globals()
    src = state.startup_hydrated_sections if isinstance(state.startup_hydrated_sections, dict) else {}
    return _normalize_hydrated_section_row(src.get(str(section_name).strip().lower(), {}))


def _sync_last_begin_hydrated_stacks(state: CycleState) -> None:
    _bind_core_globals()
    left_row = get_startup_hydrated_section_row(state, SECTION_LEFT_NAME)
    right_row = get_startup_hydrated_section_row(state, SECTION_RIGHT_NAME)
    payload = {
        "sections": {
            SECTION_LEFT_NAME: dict(left_row),
            SECTION_RIGHT_NAME: dict(right_row),
        },
        "observed_stack_levels": {
            SECTION_LEFT_NAME: int(left_row.get("stack_level", 0) or 0),
            SECTION_RIGHT_NAME: int(right_row.get("stack_level", 0) or 0),
        },
        "observed_sequences": {
            SECTION_LEFT_NAME: list(left_row.get("color_sequence_bottom_to_top", [])),
            SECTION_RIGHT_NAME: list(right_row.get("color_sequence_bottom_to_top", [])),
        },
    }
    if not isinstance(state.last_begin_stack_verify, dict):
        state.last_begin_stack_verify = {}
    state.last_begin_stack_verify["hydrated_stacks"] = dict(payload)


def _set_authoritative_section_sequence(
    state: CycleState,
    section_name: str,
    seq_bottom_to_top: list[str],
) -> dict:
    _bind_core_globals()
    section_norm = str(section_name).strip().lower()
    if section_norm not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        return _startup_default_hydrated_section_row()
    existing = get_startup_hydrated_section_row(state, section_norm)
    seq_norm: list[str] = []
    for color in list(seq_bottom_to_top):
        color_norm = str(color).strip().lower()
        if color_norm in {"orange", "blue", "unknown"}:
            seq_norm.append(color_norm)
    seq_norm = list(seq_norm[: int(max(1, MAX_STACK_LEVELS_PER_SECTION))])
    level_i = int(max(0, min(len(seq_norm), int(max(1, MAX_STACK_LEVELS_PER_SECTION)))))
    _, _, seq_use = _slots_from_level_and_sequence(level_i, seq_norm)
    top_color = seq_use[-1] if seq_use else "unknown"
    prev_tracks = list(existing.get("tracks_bottom_to_top", []))
    tracks_use = list(prev_tracks[:level_i])
    if len(tracks_use) < level_i:
        tracks_use.extend([None] * int(level_i - len(tracks_use)))
    row = _normalize_hydrated_section_row(
        {
            "stack_level": int(level_i),
            "top_color": str(top_color),
            "color_sequence_bottom_to_top": list(seq_use),
            "tracks_bottom_to_top": list(tracks_use),
            "entries": [],
        }
    )
    src = state.startup_hydrated_sections if isinstance(state.startup_hydrated_sections, dict) else {}
    src_out = dict(src)
    src_out[section_norm] = dict(row)
    state.startup_hydrated_sections = dict(src_out)
    if int(level_i) <= 0:
        _clear_locked_stack_anchor_xyz(state, section_norm)
        _clear_last_popped_xy(state, section_norm)
    _sync_last_begin_hydrated_stacks(state)
    return dict(row)


def get_authoritative_stack_levels(state: CycleState) -> dict[str, int]:
    _bind_core_globals()
    left_row = get_startup_hydrated_section_row(state, SECTION_LEFT_NAME)
    right_row = get_startup_hydrated_section_row(state, SECTION_RIGHT_NAME)
    return {
        SECTION_LEFT_NAME: int(left_row.get("stack_level", 0) or 0),
        SECTION_RIGHT_NAME: int(right_row.get("stack_level", 0) or 0),
    }


def append_authoritative_stack_cube(
    state: CycleState,
    section_name: str,
    cube_color: str,
) -> dict:
    _bind_core_globals()
    section_norm = str(section_name).strip().lower()
    if section_norm not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        return {"changed": False, "reason": "invalid_section", "section": section_norm}
    color_norm = str(cube_color).strip().lower()
    if color_norm not in {"orange", "blue"}:
        color_norm = "unknown"
    row = get_startup_hydrated_section_row(state, section_norm)
    seq = list(row.get("color_sequence_bottom_to_top", []))
    max_layers = int(max(1, MAX_STACK_LEVELS_PER_SECTION))
    if len(seq) >= int(max_layers):
        return {
            "changed": False,
            "reason": "stack_full",
            "section": section_norm,
            "stack_level": int(row.get("stack_level", 0) or 0),
            "sequence": list(seq),
        }
    seq.append(str(color_norm))
    updated = _set_authoritative_section_sequence(state, section_norm, seq)
    return {
        "changed": True,
        "reason": "ok",
        "section": section_norm,
        "stack_level": int(updated.get("stack_level", 0) or 0),
        "sequence": list(updated.get("color_sequence_bottom_to_top", [])),
        "top_color": str(updated.get("top_color", "unknown")),
    }


def pop_authoritative_stack_top(
    state: CycleState,
    section_name: str,
    removed_xyz: list[float] | tuple[float, float, float] | np.ndarray | None = None,
) -> dict:
    _bind_core_globals()
    section_norm = str(section_name).strip().lower()
    if section_norm not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        return {"changed": False, "reason": "invalid_section", "section": section_norm}
    row = get_startup_hydrated_section_row(state, section_norm)
    seq = list(row.get("color_sequence_bottom_to_top", []))
    if not seq:
        return {
            "changed": False,
            "reason": "already_empty",
            "section": section_norm,
            "stack_level": int(row.get("stack_level", 0) or 0),
            "sequence": list(seq),
        }
    removed_color = str(seq[-1]).strip().lower()
    removed_xy: list[float] | None = None
    removed_xyz_norm = _finite_xyz_or_none(removed_xyz)
    if isinstance(removed_xyz_norm, list) and len(removed_xyz_norm) >= 2:
        removed_xy = [float(removed_xyz_norm[0]), float(removed_xyz_norm[1])]
    else:
        # Fallback to current locked anchor XY when caller did not provide the removed cube XYZ.
        anchor_xyz, _ = _get_locked_stack_anchor_xyz(state, section_norm)
        if isinstance(anchor_xyz, (list, tuple)) and len(anchor_xyz) >= 2:
            try:
                ax = float(anchor_xyz[0])
                ay = float(anchor_xyz[1])
            except Exception:
                ax = float("nan")
                ay = float("nan")
            if np.isfinite(ax) and np.isfinite(ay):
                removed_xy = [float(ax), float(ay)]
    seq = list(seq[:-1])
    updated = _set_authoritative_section_sequence(state, section_norm, seq)
    # Keep XY memory for stack place/rebuild after correction pops.
    if isinstance(removed_xy, list) and len(removed_xy) >= 2:
        _set_last_popped_xy(state, section_norm, removed_xy)
    return {
        "changed": True,
        "reason": "ok",
        "section": section_norm,
        "removed_color": removed_color,
        "stack_level": int(updated.get("stack_level", 0) or 0),
        "sequence": list(updated.get("color_sequence_bottom_to_top", [])),
        "top_color": str(updated.get("top_color", "unknown")),
        "saved_last_popped_xy": (
            None if not isinstance(removed_xy, list) or len(removed_xy) < 2 else [float(removed_xy[0]), float(removed_xy[1])]
        ),
    }

def _scene_sections_for_side(side: str) -> set[str]:
    _bind_core_globals()
    side_name = str(side or "all").strip().lower()
    if side_name in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        return {side_name}
    return {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}


def _row_color_for_reconcile(obs: SceneObservation, projected_row: dict) -> tuple[str, float]:
    _bind_core_globals()
    try:
        u = int(projected_row.get("u", 0))
        v = int(projected_row.get("v", 0))
    except Exception:
        u, v = 0, 0
    color_name, color_conf = classify_cube_color_patch(
        obs.image_bgr,
        bbox_xyxy=projected_row.get("bbox_xyxy", None),
        center_uv=(int(u), int(v)),
    )
    color_norm = str(color_name).strip().lower()
    if color_norm not in {"orange", "blue"}:
        color_norm = "unknown"
    return color_norm, float(color_conf)


def _append_unique_reconcile_row(
    rows: list[dict],
    row: dict,
    xy_merge_m: float,
    z_merge_m: float,
) -> None:
    _bind_core_globals()
    xyz = np.array(row.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
    if xyz.size < 3 or not np.all(np.isfinite(xyz[:3])):
        return
    for i, prev in enumerate(list(rows)):
        pxyz = np.array(prev.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
        if pxyz.size < 3 or not np.all(np.isfinite(pxyz[:3])):
            continue
        d_xy = float(math.hypot(float(xyz[0]) - float(pxyz[0]), float(xyz[1]) - float(pxyz[1])))
        d_z = float(abs(float(xyz[2]) - float(pxyz[2])))
        if d_xy <= float(xy_merge_m) and d_z <= float(z_merge_m):
            keep_new = float(row.get("conf", 0.0)) >= float(prev.get("conf", 0.0))
            if keep_new:
                rows[i] = dict(row)
            return
    rows.append(dict(row))


def _section_snapshot_signature(section_row: dict | None) -> dict:
    _bind_core_globals()
    row = section_row if isinstance(section_row, dict) else {}
    seq = row.get("color_sequence_bottom_to_top", [])
    seq_norm = [
        (str(c).strip().lower() if str(c).strip().lower() in {"orange", "blue"} else "unknown")
        for c in list(seq)
    ]
    return {
        "stack_level": int(row.get("stack_level", 0) or 0),
        "top_color": str(row.get("top_color", "unknown")).strip().lower(),
        "color_sequence_bottom_to_top": seq_norm,
    }


def reconcile_scene(
    *,
    state: CycleState,
    arm: Arm,
    per: Perception | None,
    det: YOLODetector | None,
    side: str = "all",
    mode: str = "scan",
    target_xyz: list[float] | tuple[float, float, float] | np.ndarray | None = None,
    include_pick_rows: bool = False,
    samples: int | None = None,
    min_conf: float | None = None,
    detector_draw: bool = False,
    show_window: bool = False,
    status_line: str = "",
) -> dict:
    _bind_core_globals()
    side_name = str(side or "all").strip().lower()
    mode_name = str(mode or "scan").strip().lower()
    selected_sections = _scene_sections_for_side(side_name)
    n = max(1, int(samples if samples is not None else SCENE_RECON_SAMPLES))
    conf_min = float(min_conf if min_conf is not None else SCENE_RECON_MIN_CONF)

    summary = {
        "status": "ok",
        "mode": str(mode_name),
        "side": str(side_name),
        "samples": int(n),
        "valid_frames": 0,
        "drift_detected": False,
        "drift_flags": [],
        "collision_risk": False,
        "collision_min_dxy_m": float("inf"),
        "collision_min_dz_m": float("inf"),
        "scene_revision": int(state.scene_revision),
        "target_xyz": _finite_xyz_or_none(target_xyz),
        "section_status": {
            SECTION_LEFT_NAME: {
                "stack_level": 0,
                "top_color": "unknown",
                "color_sequence_bottom_to_top": [],
                "observed_xyz": [],
            },
            SECTION_RIGHT_NAME: {
                "stack_level": 0,
                "top_color": "unknown",
                "color_sequence_bottom_to_top": [],
                "observed_xyz": [],
            },
        },
        "timestamp_ms": int(time.time() * 1000),
    }
    if det is None or per is None:
        summary["status"] = "dependencies_missing"
        state.last_scene_reconcile = dict(summary)
        return summary

    section_centers_xy = _verify_section_xy_centers()
    band_min, band_max = _place_space_y_band_bounds()
    rows_by_section: dict[str, list[dict]] = {
        SECTION_LEFT_NAME: [],
        SECTION_RIGHT_NAME: [],
    }
    selected_xyz_rows: list[np.ndarray] = []
    preview_failed = False

    for _ in range(n):
        obs = observe_scene_frame(
            det=det,
            arm=arm,
            per=per,
            draw=bool(detector_draw),
            projected_min_conf=float(conf_min),
            state=state,
            update_tracks=True,
        )
        if obs is None:
            continue
        summary["valid_frames"] = int(summary["valid_frames"]) + 1
        if bool(show_window) and bool(SHOW_WINDOW) and (not preview_failed):
            try:
                overlay_status = (
                    f"{mode_name} sample {int(summary['valid_frames'])}/{int(n)}"
                )
                extra_status = str(status_line or "").strip()
                if extra_status:
                    overlay_status = f"{overlay_status} | {extra_status}"
                frame = obs.image_display.copy()
                if TRACK_ENABLE:
                    frame = render_operator_overlay(
                        frame=frame,
                        state=state,
                        ui_mode=UI_MODE,
                        tracks=state.track_memory,
                        active_track_id=state.active_target_track_id,
                        cx=int(obs.image_center_uv[0]),
                        cy=int(obs.image_center_uv[1]),
                        status_line=overlay_status,
                    )
                else:
                    cv2.putText(
                        frame,
                        overlay_status,
                        (12, max(24, int(frame.shape[0]) - 16)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (255, 255, 255),
                        1,
                    )
                cv2.imshow(WINDOW_NAME, frame)
                cv2.waitKey(1)
            except Exception as exc:
                preview_failed = True
                print(f"[StartupVerify] preview window disabled after error: {exc}")
        for prow in list(obs.projected_rows):
            conf = float(prow.get("conf", 0.0))
            if conf < float(conf_min):
                continue
            xyz = np.array(prow.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
            if xyz.size < 3 or not np.all(np.isfinite(xyz[:3])):
                continue
            if (not include_pick_rows) and float(xyz[1]) <= float(PICK_MAX_BASE_Y_M):
                continue
            # Use XY-only section assignment for reconcile so side labels come from
            # physical slot proximity instead of Y-band heuristics.
            # Startup modes get a slightly larger XY margin.
            max_dist_m = (
                float(STARTUP_STACK_SECTION_XY_MARGIN_M)
                if str(mode_name).startswith("startup_")
                else float(SCENE_RECON_SECTION_MAX_DIST_M)
            )
            inferred_section, assign_info = _infer_section_for_place_xy(
                float(xyz[0]),
                float(xyz[1]),
                section_centers_xy,
                band_min=None,
                band_max=None,
                max_center_dist_m=float(max_dist_m),
            )
            if inferred_section is None:
                continue
            inferred_section = str(inferred_section).strip().lower()
            if inferred_section not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
                continue
            if inferred_section not in selected_sections and side_name != "all":
                continue
            color_name, color_conf = _row_color_for_reconcile(obs, prow)
            row = {
                "u": int(prow.get("u", 0)),
                "v": int(prow.get("v", 0)),
                "conf": float(conf),
                "track_id": prow.get("track_id", None),
                "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                "section": str(inferred_section),
                "color_name": str(color_name),
                "color_conf": float(color_conf),
            }
            _append_unique_reconcile_row(
                rows=rows_by_section[inferred_section],
                row=row,
                xy_merge_m=float(max(0.005, SCENE_RECON_DEDUP_XY_M)),
                z_merge_m=float(max(0.005, SCENE_RECON_DEDUP_Z_M)),
            )

    if int(summary["valid_frames"]) <= 0:
        summary["status"] = "no_observation"
        state.last_scene_reconcile = dict(summary)
        return summary

    for section_name in [SECTION_LEFT_NAME, SECTION_RIGHT_NAME]:
        sec_rows = list(rows_by_section.get(section_name, []))
        sec_rows.sort(key=lambda row: float(row.get("xyz", [0.0, 0.0, 0.0])[2]))
        seq = [str(row.get("color_name", "unknown")).strip().lower() for row in sec_rows]
        seq = [c if c in {"orange", "blue"} else "unknown" for c in seq]
        top_color = seq[-1] if seq else "unknown"
        observed_xyz = [
            _finite_xyz_or_none(row.get("xyz", None))
            for row in sec_rows
        ]
        observed_xyz = [xyz for xyz in observed_xyz if xyz is not None]
        summary["section_status"][section_name] = {
            "stack_level": int(len(sec_rows)),
            "top_color": str(top_color),
            "color_sequence_bottom_to_top": list(seq),
            "observed_xyz": observed_xyz,
        }
        if section_name in selected_sections or side_name == "all":
            for xyz_row in observed_xyz:
                arr = np.array(xyz_row, dtype=float).reshape(-1)
                if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                    selected_xyz_rows.append(arr[:3].copy())

    prev_snapshot = {
        SECTION_LEFT_NAME: _section_snapshot_signature(state.scene_snapshot_sections.get(SECTION_LEFT_NAME, {})),
        SECTION_RIGHT_NAME: _section_snapshot_signature(state.scene_snapshot_sections.get(SECTION_RIGHT_NAME, {})),
    }
    new_snapshot = {
        SECTION_LEFT_NAME: _section_snapshot_signature(summary["section_status"].get(SECTION_LEFT_NAME, {})),
        SECTION_RIGHT_NAME: _section_snapshot_signature(summary["section_status"].get(SECTION_RIGHT_NAME, {})),
    }
    changed_since_last = bool(prev_snapshot != new_snapshot)
    if changed_since_last:
        state.scene_revision = int(state.scene_revision) + 1
        summary["drift_flags"].append("scene_changed_since_last_scan")
    summary["scene_revision"] = int(state.scene_revision)

    for section_name in [SECTION_LEFT_NAME, SECTION_RIGHT_NAME]:
        if section_name not in selected_sections and side_name != "all":
            continue
        observed_known = [
            c for c in list(summary["section_status"].get(section_name, {}).get("color_sequence_bottom_to_top", []))
            if c in {"orange", "blue"}
        ]
        ledger_seq = get_section_confirmed_color_sequence_bottom_to_top(state, section_name)
        if observed_known != ledger_seq:
            summary["drift_flags"].append(f"ledger_mismatch_{section_name}")

    target_arr = None
    if isinstance(target_xyz, (list, tuple, np.ndarray)):
        try:
            tarr = np.array([float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])], dtype=float).reshape(-1)
        except Exception:
            tarr = np.array([np.nan, np.nan, np.nan], dtype=float)
        if tarr.size >= 3 and np.all(np.isfinite(tarr[:3])):
            target_arr = tarr[:3].copy()

    if target_arr is not None and selected_xyz_rows:
        best_xy = float("inf")
        best_z = float("inf")
        occ_xy = float(max(0.005, SCENE_RECON_OCCUPANCY_XY_M))
        occ_z = float(max(0.005, SCENE_RECON_OCCUPANCY_Z_M))
        for xyz_arr in selected_xyz_rows:
            d_xy = float(math.hypot(float(xyz_arr[0]) - float(target_arr[0]), float(xyz_arr[1]) - float(target_arr[1])))
            d_z = float(abs(float(xyz_arr[2]) - float(target_arr[2])))
            if d_xy < best_xy:
                best_xy = d_xy
                best_z = d_z
            if d_xy <= occ_xy and d_z <= occ_z and float(xyz_arr[2]) >= float(target_arr[2] - (0.50 * occ_z)):
                summary["collision_risk"] = True
        summary["collision_min_dxy_m"] = float(best_xy)
        summary["collision_min_dz_m"] = float(best_z)
        if bool(summary["collision_risk"]):
            summary["drift_flags"].append("collision_risk_target_occupied")

    summary["drift_detected"] = bool(summary["drift_flags"])
    state.scene_snapshot_sections = dict(new_snapshot)
    state.last_scene_reconcile = dict(summary)
    return summary


def _extract_valid_z(xyz: object) -> float:
    _bind_core_globals()
    if not isinstance(xyz, (list, tuple)) or len(xyz) < 3:
        return float("nan")
    try:
        z_val = float(xyz[2])
    except Exception:
        return float("nan")
    return z_val if np.isfinite(z_val) else float("nan")


def remeasure_stack_xyz_after_center(
    *,
    arm: Arm,
    per: Perception | None,
    det: YOLODetector | None,
    expected_xyz: object,
    pending_stack_level: int | None,
    section_name: str,
    expected_color: str | None,
    state: CycleState,
    attempt_i: int = 0,
) -> list[float] | None:
    _bind_core_globals()
    if per is None or det is None:
        return None
    if not isinstance(expected_xyz, (list, tuple)) or len(expected_xyz) < 3:
        return None
    expected_xyz_arr = np.array(expected_xyz, dtype=float)
    if expected_xyz_arr.size < 3 or not np.all(np.isfinite(expected_xyz_arr[:3])):
        return None

    stack_level_i = 0
    if pending_stack_level is not None:
        try:
            stack_level_i = max(0, int(pending_stack_level))
        except Exception:
            stack_level_i = 0
    expected_section_norm = str(section_name).strip().lower()
    if expected_section_norm not in {SECTION_LEFT_NAME, SECTION_RIGHT_NAME}:
        expected_section_norm = None
    else:
        expected_section_norm = str(expected_section_norm)
    expected_color_norm = str(expected_color or "").strip().lower()
    if expected_color_norm not in {"orange", "blue"}:
        expected_color_norm = None
    attempt_timeout_s = min(
        float(PLACE_VERIFY_V2_ACTIVE_CENTER_TIMEOUT_S),
        1.6 + (0.20 * max(0, int(attempt_i))),
    )
    center_min_conf = float(min(DETECT_CONF, PLACE_VERIFY_MIN_CONF))

    centered_uv = center_object_on_expected_slot(
        det=det,
        arm=arm,
        per=per,
        expected_xyz=expected_xyz_arr,
        timeout_s=float(attempt_timeout_s),
        required_centered_frames=3,
        min_conf=float(center_min_conf),
        radius_m=max(float(PLACE_VERIFY_V2_RADIUS_M), float(PLACE_VERIFY_RADIUS_M)),
        show_window=False,
        stack_level=int(stack_level_i),
        min_z_m=None,
        expected_section=expected_section_norm,
        expected_color=expected_color_norm,
        state=state,
    )
    if centered_uv is None:
        return None

    mx, my, mz = measure_base_point_from_uv(
        arm,
        per,
        int(centered_uv[0]),
        int(centered_uv[1]),
        n=max(2, min(4, int(N_MEAS))),
    )
    if not np.all(np.isfinite([mx, my, mz])):
        return None
    return [float(mx), float(my), float(mz)]


def remeasure_stack_xyz_until_stable(
    *,
    arm: Arm,
    per: Perception | None,
    det: YOLODetector | None,
    expected_xyz: object,
    pending_stack_level: int | None,
    section_name: str,
    expected_color: str | None,
    state: CycleState,
) -> tuple[list[float] | None, dict]:
    _bind_core_globals()
    attempts = max(1, int(STACK_REMEASURE_MAX_ATTEMPTS))
    required = max(1, min(attempts, int(STACK_REMEASURE_REQUIRED_VALID)))
    samples: list[np.ndarray] = []
    last_xyz: list[float] | None = None
    for k in range(attempts):
        xyz = remeasure_stack_xyz_after_center(
            arm=arm,
            per=per,
            det=det,
            expected_xyz=expected_xyz,
            pending_stack_level=pending_stack_level,
            section_name=section_name,
            expected_color=expected_color,
            state=state,
            attempt_i=k,
        )
        if xyz is not None and np.all(np.isfinite(np.array(xyz, dtype=float)[:3])):
            last_xyz = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
            samples.append(np.array(last_xyz, dtype=float))
            z_vals = [float(s[2]) for s in samples]
            z_spread = float(max(z_vals) - min(z_vals))
            print(
                f"[StackRemeasure] attempt {k + 1}/{attempts} valid z={float(last_xyz[2]):.3f} "
                f"(valid={len(samples)}/{required}, z_spread={z_spread:.3f} m)"
            )
            if len(samples) >= required and z_spread <= float(STACK_REMEASURE_MAX_Z_SPREAD_M):
                med = np.median(np.vstack(samples), axis=0)
                return (
                    [float(med[0]), float(med[1]), float(med[2])],
                    {
                        "status": "stable",
                        "attempts": int(attempts),
                        "valid": int(len(samples)),
                        "z_spread_m": float(z_spread),
                    },
                )
        else:
            print(f"[StackRemeasure] attempt {k + 1}/{attempts} invalid (no centered measurement).")
        if float(STACK_REMEASURE_PAUSE_S) > 0 and (k + 1) < attempts:
            time.sleep(max(0.0, float(STACK_REMEASURE_PAUSE_S)))

    if len(samples) >= required:
        z_vals = [float(s[2]) for s in samples]
        z_spread = float(max(z_vals) - min(z_vals))
        return None, {
            "status": "unstable",
            "attempts": int(attempts),
            "valid": int(len(samples)),
            "z_spread_m": float(z_spread),
            "last_xyz": last_xyz,
        }
    return None, {
        "status": "insufficient",
        "attempts": int(attempts),
        "valid": int(len(samples)),
        "last_xyz": last_xyz,
    }


def infer_stack_layers_from_measurement(
    *,
    measured_xyz: object,
    expected_xyz: object,
    slot_used: int | None,
    current_layers: int,
) -> int:
    _bind_core_globals()
    fallback_layers = max(0, int(current_layers) - 1)
    if slot_used is None:
        return fallback_layers

    measured_z = _extract_valid_z(measured_xyz)
    expected_z = _extract_valid_z(expected_xyz)
    if not np.isfinite(measured_z):
        return fallback_layers

    plausible_measured = (not np.isfinite(expected_z)) or (
        measured_z <= (expected_z + float(PLACE_VERIFY_V2_Z_MARGIN_M))
    )
    if not plausible_measured:
        return fallback_layers

    base_slot_xyz = slot_target_xyz(int(slot_used))
    base_slot_z = float(base_slot_xyz[2]) if np.all(np.isfinite(base_slot_xyz[:3])) else float("nan")
    if not np.isfinite(base_slot_z):
        return fallback_layers

    stack_base_z = max(base_slot_z, float(STACK_RELEASE_Z_GUARD_M))
    dz = max(1e-6, float(STACK_LEVEL_DZ_M))
    inferred_layers = int(round((measured_z - stack_base_z) / dz)) + 1
    inferred_layers = max(0, min(max(1, int(MAX_STACK_LEVELS_PER_SECTION)), inferred_layers))
    return int(inferred_layers)


# ============================= Planner / LLM policy calls =============================
