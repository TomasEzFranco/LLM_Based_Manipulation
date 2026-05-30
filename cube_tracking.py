#!/usr/bin/env python3
from __future__ import annotations

import math
import time

import numpy as np


def update_cube_tracks(
    state,
    detections: list[dict],
    max_miss_frames: int | None = None,
    now_ms: int | None = None,
    image_center_uv: tuple[int, int] | None = None,
) -> dict[int, dict]:
    import runtime_core as core

    if now_ms is None:
        now_ms = int(time.time() * 1000)
    if max_miss_frames is None:
        max_miss_frames = int(core.TRACK_MAX_MISS_FRAMES)
    max_miss_frames = max(1, int(max_miss_frames))

    if not core.TRACK_ENABLE:
        state.track_memory = {}
        state.active_target_track_id = None
        state.last_track_snapshot = {
            "enabled": False,
            "visible_track_count": 0,
            "visible_section_counts": {core.SECTION_LEFT_NAME: 0, core.SECTION_RIGHT_NAME: 0},
            "active_target_track_id": None,
            "image_center_uv": (None if image_center_uv is None else [int(image_center_uv[0]), int(image_center_uv[1])]),
            "visible_tracks": [],
        }
        return state.track_memory

    tracks = state.track_memory
    seen_track_ids: set[int] = set()
    missing_id_rows = 0

    for det_row in detections:
        raw_tid = det_row.get("track_id", None)
        try:
            track_id = None if raw_tid is None else int(raw_tid)
        except Exception:
            track_id = None
        if track_id is None:
            missing_id_rows += 1
            continue
        seen_track_ids.add(track_id)
        prev_seen = int(tracks.get(track_id, {}).get("seen_frames", 0))
        tracks[track_id] = {
            "track_id": int(track_id),
            "xyz": list(det_row.get("xyz", [np.nan, np.nan, np.nan])),
            "uv": [int(det_row.get("u", 0)), int(det_row.get("v", 0))],
            "bbox_xyxy": det_row.get("bbox_xyxy"),
            "conf": float(det_row.get("conf", 0.0)),
            "cls": det_row.get("cls"),
            "name": det_row.get("name"),
            "is_tracked": bool(det_row.get("is_tracked", True)),
            "miss_frames": 0,
            "seen_frames": prev_seen + 1,
            "last_seen_ms": int(now_ms),
        }

    for track_id in list(tracks.keys()):
        if int(track_id) in seen_track_ids:
            continue
        track = tracks.get(track_id)
        if not track:
            continue
        miss = int(track.get("miss_frames", 0)) + 1
        track["miss_frames"] = miss
        if miss > max_miss_frames:
            del tracks[track_id]

    if missing_id_rows > 0:
        state.track_untracked_frames += 1
        state.track_untracked_detections_total += int(missing_id_rows)
        if core.TRACK_WARN_MISSING_IDS:
            warn_interval_ms = max(0, int(float(core.TRACK_WARN_MISSING_IDS_INTERVAL_S) * 1000.0))
            if warn_interval_ms == 0 or (int(now_ms) - int(state.track_last_warn_ms)) >= warn_interval_ms:
                print(
                    f"[TrackWarn] missing_track_id rows={missing_id_rows}/{len(detections)} "
                    f"(cum_rows={state.track_untracked_detections_total}, frames={state.track_untracked_frames})"
                )
                state.track_last_warn_ms = int(now_ms)

    visible_tracks = [
        dict(track)
        for track in tracks.values()
        if int(track.get("miss_frames", 0)) == 0
    ]
    visible_tracks.sort(key=lambda row: int(row.get("track_id", 0)))
    visible_section_counts = estimate_visible_section_counts_from_tracks(tracks)

    if state.active_target_track_id is not None and state.active_target_track_id not in tracks:
        state.active_target_track_id = None

    state.last_track_snapshot = {
        "enabled": True,
        "visible_track_count": len(visible_tracks),
        "visible_section_counts": dict(visible_section_counts),
        "active_target_track_id": state.active_target_track_id,
        "image_center_uv": (None if image_center_uv is None else [int(image_center_uv[0]), int(image_center_uv[1])]),
        "visible_tracks": [
            {
                "track_id": int(row.get("track_id", -1)),
                "uv": list(row.get("uv", [0, 0])),
                "xyz": core._finite_xyz_or_none(row.get("xyz")),
                "conf": float(row.get("conf", 0.0)),
                "name": row.get("name"),
                "is_tracked": bool(row.get("is_tracked", False)),
            }
            for row in visible_tracks
        ],
        "missing_track_id_rows_last": int(missing_id_rows),
    }
    return tracks


def select_intended_track_for_pick(
    state,
    section_groups: dict[str, list[int]] | None = None,
    blocked_track_ids: set[int] | None = None,
    blocked_xyzs: list[list[float]] | None = None,
) -> int | None:
    import runtime_core as core

    _ = section_groups
    blocked_track_ids = set() if blocked_track_ids is None else {int(x) for x in blocked_track_ids}
    blocked_xyzs_norm: list[np.ndarray] = []
    if blocked_xyzs:
        for xyz in blocked_xyzs:
            if not isinstance(xyz, (list, tuple)) or len(xyz) < 3:
                continue
            try:
                arr = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2])], dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                blocked_xyzs_norm.append(arr[:3].copy())

    def _is_near_blocked_xyz(xyz_arr: np.ndarray) -> bool:
        if xyz_arr.size < 3 or not np.all(np.isfinite(xyz_arr[:3])):
            return False
        for blocked_xyz in blocked_xyzs_norm:
            d_xy = float(math.hypot(float(xyz_arr[0]) - float(blocked_xyz[0]), float(xyz_arr[1]) - float(blocked_xyz[1])))
            d_z = float(abs(float(xyz_arr[2]) - float(blocked_xyz[2])))
            if d_xy <= float(core.PICK_OTHER_BLOCK_XY_M) and d_z <= float(core.PICK_OTHER_BLOCK_Z_M):
                return True
        return False

    if not core.TRACK_ENABLE:
        state.active_target_track_id = None
        return None
    tracks = state.track_memory
    if not tracks:
        state.active_target_track_id = None
        return None
    visible_ids: list[int] = []
    for track_id, row in tracks.items():
        if int(row.get("miss_frames", 0)) != 0:
            continue
        tid = int(track_id)
        if tid in blocked_track_ids:
            continue
        if blocked_xyzs_norm:
            xyz = np.array(row.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
            if _is_near_blocked_xyz(xyz):
                continue
        visible_ids.append(tid)
    if not visible_ids:
        state.active_target_track_id = None
        return None
    # Legacy sticky-target behavior can be useful for stability, but strict top-pick
    # should re-evaluate all visible detections each cycle.
    if (not core.TRACK_PICK_TOP_STRICT) and state.active_target_track_id in visible_ids:
        return state.active_target_track_id

    center_uv = state.last_track_snapshot.get("image_center_uv", [320, 240])
    if center_uv is None or len(center_uv) != 2:
        center_uv = [320, 240]
    cx = int(center_uv[0])
    cy = int(center_uv[1])

    if core.TRACK_PICK_PREFER_TOP:
        top_candidates: list[tuple[float, float, float, int]] = []
        for track_id in visible_ids:
            row = tracks.get(track_id, {})
            xyz = np.array(row.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
            if xyz.size < 3 or not np.all(np.isfinite(xyz[:3])):
                continue
            if core.PICK_FILTER_BY_BASE_Y and np.isfinite(xyz[1]) and float(xyz[1]) > core.PICK_MAX_BASE_Y_M:
                continue
            uv = row.get("uv", [cx, cy])
            d2 = float((int(uv[0]) - cx) ** 2 + (int(uv[1]) - cy) ** 2)
            z = float(xyz[2])
            conf = float(row.get("conf", 0.0))
            top_candidates.append((z, conf, d2, int(track_id)))
        if top_candidates:
            # Highest-z first; then confidence; then closest-to-center as tie-breaker.
            top_candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
            if len(top_candidates) > 1:
                z0 = float(top_candidates[0][0])
                tie_band = max(0.0, float(core.TRACK_PICK_TOP_TIE_Z_M))
                near_top = [row for row in top_candidates if (z0 - float(row[0])) <= tie_band]
                if len(near_top) > 1:
                    near_top.sort(key=lambda item: (item[2], -item[1], -item[0]))
                    chosen = int(near_top[0][3])
                else:
                    chosen = int(top_candidates[0][3])
            else:
                chosen = int(top_candidates[0][3])
            state.active_target_track_id = chosen
            return chosen

    candidates: list[tuple[tuple[float, float], int]] = []
    for track_id in visible_ids:
        row = tracks.get(track_id, {})
        uv = row.get("uv", [cx, cy])
        xyz = np.array(row.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
        if core.PICK_FILTER_BY_BASE_Y and xyz.size >= 2 and np.isfinite(xyz[1]) and float(xyz[1]) > core.PICK_MAX_BASE_Y_M:
            continue
        dx = float(int(uv[0]) - cx)
        dy = float(int(uv[1]) - cy)
        d2 = dx * dx + dy * dy
        score = (d2, -float(row.get("conf", 0.0)))
        candidates.append((score, int(track_id)))

    if not candidates:
        # Fall back to any visible target if all failed pick-side filter.
        for track_id in visible_ids:
            row = tracks.get(track_id, {})
            score = (0.0, -float(row.get("conf", 0.0)))
            candidates.append((score, int(track_id)))

    candidates.sort(key=lambda item: item[0])
    chosen = int(candidates[0][1]) if candidates else None
    state.active_target_track_id = chosen
    return chosen


def nearest_visible_track_by_uv(
    state,
    u: int,
    v: int,
    max_dist_px: float = 90.0,
) -> int | None:
    best_track_id = None
    best_d2 = float("inf")
    max_d2 = float(max_dist_px) * float(max_dist_px)
    for track_id, row in state.track_memory.items():
        if int(row.get("miss_frames", 0)) != 0:
            continue
        uv = row.get("uv", None)
        if not isinstance(uv, (list, tuple)) or len(uv) < 2:
            continue
        du = float(int(uv[0]) - int(u))
        dv = float(int(uv[1]) - int(v))
        d2 = du * du + dv * dv
        if d2 < best_d2:
            best_d2 = d2
            best_track_id = int(track_id)
    if best_track_id is None or best_d2 > max_d2:
        return None
    return best_track_id


def estimate_visible_section_counts_from_tracks(tracks: dict[int, dict]) -> dict[str, int]:
    import runtime_core as core

    counts = {core.SECTION_LEFT_NAME: 0, core.SECTION_RIGHT_NAME: 0}
    slots = core.get_place_slots()
    groups = core.section_slot_groups(slots)
    left_idxs = groups.get(core.SECTION_LEFT_NAME, [])
    right_idxs = groups.get(core.SECTION_RIGHT_NAME, [])
    if not left_idxs or not right_idxs:
        return counts
    left_y = float(np.mean([float(slots[i][1]) for i in left_idxs]))
    right_y = float(np.mean([float(slots[i][1]) for i in right_idxs]))
    for row in tracks.values():
        if int(row.get("miss_frames", 0)) != 0:
            continue
        xyz = np.array(row.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
        if xyz.size < 3 or not np.all(np.isfinite(xyz[:3])):
            continue
        # Keep this as a "likely placed" cube signal.
        if float(xyz[1]) <= core.PICK_MAX_BASE_Y_M:
            continue
        y = float(xyz[1])
        if abs(y - left_y) <= abs(y - right_y):
            counts[core.SECTION_LEFT_NAME] += 1
        else:
            counts[core.SECTION_RIGHT_NAME] += 1
    return counts
