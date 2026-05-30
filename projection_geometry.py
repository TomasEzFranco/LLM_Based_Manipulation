#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

# ============================= Kinematics / camera transforms =============================
# --- Kinematic model calibration (meters) ---
SHOULDER_HEIGHT_M = 0.14      # base-to-shoulder vertical offset
LINK2_LENGTH_M = 0.35         # upper-arm effective length
LINK3_LENGTH_M = 0.05         # forearm/wrist offset used in IK transform chain
# --- Camera mount offset in end-effector frame (meters) ---
# Signs follow the DH/end-effector frame convention used in base_to_camera_T().
CAM_OFF_X_M = 0.045
# Updated from tuning sweeps: -0.016 gave best observed Y alignment.
CAM_OFF_Y_M = -0.016
CAM_OFF_Z_M = 0.22095


def get_cam_offsets() -> dict[str, float]:
    return {
        "cam_off_x_m": float(CAM_OFF_X_M),
        "cam_off_y_m": float(CAM_OFF_Y_M),
        "cam_off_z_m": float(CAM_OFF_Z_M),
    }


def set_cam_offsets(*, cam_off_x_m: float, cam_off_y_m: float, cam_off_z_m: float) -> None:
    global CAM_OFF_X_M, CAM_OFF_Y_M, CAM_OFF_Z_M
    CAM_OFF_X_M = float(cam_off_x_m)
    CAM_OFF_Y_M = float(cam_off_y_m)
    CAM_OFF_Z_M = float(cam_off_z_m)


def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,      ca,      d  ],
        [0.0,    0.0,     0.0,    1.0  ],
    ], dtype=float)


def base_to_camera_T(yaw: float, shoulder: float, elbow: float) -> np.ndarray:
    r = np.sqrt(LINK2_LENGTH_M**2 + LINK3_LENGTH_M**2)
    beta = np.arctan2(LINK3_LENGTH_M, LINK2_LENGTH_M)
    T01 = _dh(theta=yaw, d=SHOULDER_HEIGHT_M, a=0.0, alpha=-np.pi/2)
    T12 = _dh(theta=shoulder + beta - np.pi/2, d=0.0, a=r, alpha=0.0)
    T23 = _dh(theta=elbow - beta, d=0.0, a=0.0, alpha=-np.pi/2)
    T3C = np.array([
        [1, 0, 0, CAM_OFF_X_M],
        [0, 1, 0, CAM_OFF_Y_M],
        [0, 0, 1, CAM_OFF_Z_M],
        [0, 0, 0, 1]
    ], dtype=float)
    T_base_to_cam = T01 @ T12 @ T23 @ T3C
    return T_base_to_cam


# ============================= Projection / depth / base XYZ =============================
def uvz_to_xyz_cam(u: int, v: int, Z: float, intr) -> tuple[float,float,float]:
    if not np.isfinite(Z) or Z <= 0:
        return np.nan, np.nan, np.nan
    X = (u - intr.ppx) * Z / intr.fx
    Y = (v - intr.ppy) * Z / intr.fy
    return float(X), float(Y), float(Z)


def robust_depth_m(depth_frame, u, v, depth_scale, win=5, percentile=85):
    w, h = depth_frame.get_width(), depth_frame.get_height()
    half = max(1, win // 2)
    u0, v0 = max(0, u - half), max(0, v - half)
    u1, v1 = min(w - 1, u + half), min(h - 1, v + half)
    depth = np.asanyarray(depth_frame.get_data())
    patch = depth[v0:v1+1, u0:u1+1].astype(np.float32)
    vals = patch[patch > 0.0]
    if vals.size == 0:
        return np.nan
    return float(np.percentile(vals * depth_scale, percentile))


def estimate_base_xyz_from_uv_fast(
    arm,
    per,
    depth_frame,
    u: int,
    v: int,
    win: int = 3,
    percentile: float = 90.0,
):
    z_cam = robust_depth_m(depth_frame, int(u), int(v), per.depth_scale, win=win, percentile=percentile)
    if not np.isfinite(z_cam) or z_cam <= 0.0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    x_cam, y_cam, z_cam = uvz_to_xyz_cam(int(u), int(v), float(z_cam), per.intr)
    if not np.all(np.isfinite([x_cam, y_cam, z_cam])):
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    yaw, shoulder, elbow = arm.arm.measJointPosition[0:3].astype(float)
    t_cam_to_base = base_to_camera_T(yaw, shoulder, elbow)
    p_cam = np.array([x_cam, y_cam, z_cam, 1.0], dtype=float)
    p_base = t_cam_to_base @ p_cam
    return apply_scan_base_xy_offset(np.array(p_base[:3], dtype=float))


def apply_scan_base_xy_offset(xyz: np.ndarray) -> np.ndarray:
    """Apply global XY correction to every depth-projected cube (Z unchanged)."""
    out = np.array(xyz, dtype=float).reshape(-1).copy()
    try:
        import runtime_core as core
    except Exception:
        return out
    if not bool(getattr(core, "SCAN_BASE_XY_OFFSET_ENABLED", False)):
        return out
    if out.size >= 1 and np.isfinite(out[0]):
        out[0] = float(out[0]) + float(getattr(core, "SCAN_BASE_X_OFFSET_M", 0.0))
    if out.size >= 2 and np.isfinite(out[1]):
        out[1] = float(out[1]) + float(getattr(core, "SCAN_BASE_Y_OFFSET_M", 0.0))
    return out


def project_candidates_to_base(
    arm,
    per,
    depth_frame,
    candidates: list[dict],
    min_conf: float = 0.0,
) -> list[dict]:
    if per is None:
        return []
    rows: list[dict] = []
    for c in candidates:
        conf = float(c.get("conf", 0.0))
        if conf < float(min_conf):
            continue
        u = int(c.get("u", 0))
        v = int(c.get("v", 0))
        xyz = estimate_base_xyz_from_uv_fast(
            arm=arm,
            per=per,
            depth_frame=depth_frame,
            u=u,
            v=v,
        )
        if not np.all(np.isfinite(xyz)):
            continue
        rows.append(
            {
                "u": u,
                "v": v,
                "conf": conf,
                "cls": c.get("cls"),
                "name": c.get("name"),
                "bbox_xyxy": c.get("bbox_xyxy"),
                "track_id": c.get("track_id", None),
                "is_tracked": bool(c.get("is_tracked", False)),
                "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
            }
        )
    return rows
