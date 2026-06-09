#!/usr/bin/env python3
import os
import time
import numpy as np
def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}
# --- Visual centering controller ---
KYAW = 0.0015  # yaw gain (rad/pixel)
KSHOULDER = float(os.getenv("QARM_CENTER_KSHOULDER", "0.0006"))  # shoulder gain (rad/pixel)
KELBOW = float(os.getenv("QARM_CENTER_KELBOW", "0.0009"))  # elbow gain (rad/pixel)
# Small integral on ey to counter static bias/gravity/friction near center.
CENTER_EY_KI = float(os.getenv("QARM_CENTER_EY_KI", "0.0000"))  # shoulder rad / (pixel*frame)
CENTER_EY_KI_ELBOW = float(os.getenv("QARM_CENTER_EY_KI_ELBOW", "0.0003"))  # elbow rad / (pixel*frame)
CENTER_EY_I_CLAMP = float(os.getenv("QARM_CENTER_EY_I_CLAMP", "100"))  # pixel*frame accumulator clamp
CENTER_EY_I_DECAY = float(os.getenv("QARM_CENTER_EY_I_DECAY", "0.85"))  # leakage to avoid windup carryover
# When range gate is on, integral accumulates only while |ey| <= ENABLE_ABS_PX; farther errors leak instead.
CENTER_EY_I_ENABLE_ABS_PX = float(os.getenv("QARM_CENTER_EY_I_ENABLE_ABS_PX", "25.0"))
CENTER_EY_I_RANGE_GATE_ENABLED = _env_bool("QARM_CENTER_EY_I_RANGE_GATE_ENABLED", False)
CENTER_EY_I_DOWN_SCALE = float(os.getenv("QARM_CENTER_EY_I_DOWN_SCALE", "0.01"))
CENTER_EY_DOWN_DPHI_SCALE = float(os.getenv("QARM_CENTER_EY_DOWN_DPHI_SCALE", "0.7"))
CENTER_MOVE_SETTLE_S = float(os.getenv("QARM_CENTER_MOVE_SETTLE_S", "0.001"))
# Use 1 if positive ey corresponds to downward arm motion, -1 otherwise.
CENTER_EY_DOWN_SIGN = int(os.getenv("QARM_CENTER_EY_DOWN_SIGN", "1"))
MAX_JOINT_NUDGE = 0.09  # per-step joint nudge clamp (rad)
_center_ey_i_accum = 0.0
_center_ey_i_last_sign = 0
def _reset_centering_integrator() -> None:
    global _center_ey_i_accum, _center_ey_i_last_sign
    _center_ey_i_accum = 0.0
    _center_ey_i_last_sign = 0
def _leak_centering_integrator() -> None:
    global _center_ey_i_accum
    decay = float(np.clip(float(CENTER_EY_I_DECAY), 0.0, 1.0))
    _center_ey_i_accum *= decay
    if abs(_center_ey_i_accum) < 1e-6:
        _center_ey_i_accum = 0.0
def _compute_centering_nudge(ex: int, ey: int) -> np.ndarray:
    global _center_ey_i_accum, _center_ey_i_last_sign
    ey_i_term = 0.0
    ey_i_term_elbow = 0.0
    ey_sign = 0 if int(ey) == 0 else (1 if int(ey) > 0 else -1)
    down_sign = 1 if int(CENTER_EY_DOWN_SIGN) >= 0 else -1
    moving_down = bool(ey_sign != 0 and ey_sign == down_sign)
    if float(CENTER_EY_KI) > 0.0 or float(CENTER_EY_KI_ELBOW) > 0.0:
        if _center_ey_i_last_sign != 0 and ey_sign != 0 and ey_sign != _center_ey_i_last_sign:
            # Fast unwind when ey flips direction to reduce oscillation near center.
            _center_ey_i_accum *= 0.25
        if ey_sign != 0:
            _center_ey_i_last_sign = ey_sign
        range_gate = bool(CENTER_EY_I_RANGE_GATE_ENABLED)
        enable_px = float(max(0.0, CENTER_EY_I_ENABLE_ABS_PX))
        within_range = abs(int(ey)) <= enable_px
        if (not range_gate) or within_range:
            i_scale = float(CENTER_EY_I_DOWN_SCALE) if bool(moving_down) else 1.0
            _center_ey_i_accum += float(ey) * float(i_scale)
        elif range_gate:
            _leak_centering_integrator()
        i_clip = max(0.0, float(CENTER_EY_I_CLAMP))
        _center_ey_i_accum = float(np.clip(_center_ey_i_accum, -i_clip, i_clip))
        ey_i_term = float(CENTER_EY_KI) * float(_center_ey_i_accum)
        if float(CENTER_EY_KI_ELBOW) > 0.0:
            ey_i_term_elbow = float(CENTER_EY_KI_ELBOW) * float(_center_ey_i_accum)
    dphi = np.array(
        [
            -KYAW * ex,
            KSHOULDER * ey + ey_i_term,
            KELBOW * ey + ey_i_term_elbow,
            0.0,
        ],
        dtype=float,
    )
    if bool(moving_down):
        down_scale = float(max(0.0, min(1.0, CENTER_EY_DOWN_DPHI_SCALE)))
        dphi[1] *= float(down_scale)
        dphi[2] *= float(down_scale)
    return np.clip(dphi, -MAX_JOINT_NUDGE, MAX_JOINT_NUDGE)


def _maybe_apply_centering_nudge(
    arm,
    ex: int,
    ey: int,
    conf: float,
    frame_idx: int,
    *,
    detect_conf: float | None = None,
    center_verbose: bool | None = None,
) -> int:
    if detect_conf is None:
        detect_conf_use = float(os.getenv("QARM_DETECT_CONF", "0.5"))
    else:
        detect_conf_use = float(detect_conf)
    if center_verbose is None:
        center_verbose_use = os.getenv("QARM_CENTER_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"}
    else:
        center_verbose_use = bool(center_verbose)
    if conf < float(detect_conf_use):
        _leak_centering_integrator()
        time.sleep(0.1)
        return frame_idx
    frame_idx += 1
    yaw_now, sh_now, el_now = arm.arm.measJointPosition[0:3].astype(float)
    dphi = _compute_centering_nudge(ex, ey)
    if bool(center_verbose_use) and frame_idx % 5 == 0:
        print(
            f"[CENTER] ex={ex:+6.1f}px ey={ey:+6.1f}px conf={conf:.3f} | "
            f"dphi(rad)=[yaw={dphi[0]:+.4f}, sh={dphi[1]:+.4f}, el={dphi[2]:+.4f}]"
        )
        print(
            f"[CENTER] joints BEFORE (deg): "
            f"yaw={np.rad2deg(yaw_now):+6.1f}, "
            f"sh={np.rad2deg(sh_now):+6.1f}, "
            f"el={np.rad2deg(el_now):+6.1f}"
        )
    arm.nudge_joints(dphi)
    time.sleep(max(0.0, float(CENTER_MOVE_SETTLE_S)))
    if bool(center_verbose_use) and frame_idx % 5 == 0:
        yaw_new, sh_new, el_new = arm.arm.measJointPosition[0:3].astype(float)
        print(
            f"[CENTER] joints AFTER  (deg): "
            f"yaw={np.rad2deg(yaw_new):+6.1f}, "
            f"sh={np.rad2deg(sh_new):+6.1f}, "
            f"el={np.rad2deg(el_new):+6.1f}"
        )
    return frame_idx
