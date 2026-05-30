#!/usr/bin/env python3
"""Reusable current-based gripper limiting for QArm-style wrappers."""

from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np


@dataclass
class GripCurrentLimits:
    # Detection thresholds from measured runs
    grip_detect_a: float = 0.24
    grip_miss_max_a: float = 0.16
    grip_warn_a: float = 0.45
    grip_hard_a: float = 0.50
    emergency_trip_a: float = 0.60

    # Timing / debounce
    transient_ignore_s: float = 0.20
    debounce_samples: int = 8
    max_close_s: float = 1.40
    final_hold_s: float = 0.15

    # Grip command behavior
    min_grip: float = 0.10
    max_grip: float = 0.90
    grip_step: float = 0.005
    relax_step: float = 0.015
    warn_relax_enabled: bool = True
    warn_relax_step: float = 0.006
    warn_relax_debounce_samples: int = 3
    # Prevent early false-positive "gripped" before enough closure is commanded.
    min_detect_grip_cmd: float = 0.50
    # Optionally ignore overcurrent limiting until closure reaches this grip command.
    min_overcurrent_grip_cmd: float = 0.0


def _as_float_array(value):
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _clip_grip(grip_cmd: float, limits: GripCurrentLimits) -> float:
    return float(max(limits.min_grip, min(limits.max_grip, float(grip_cmd))))


def read_joint_currents(qarm_hw) -> np.ndarray:
    """Returns 5-length measured current vector if available; NaNs otherwise."""
    for name in ("measJointCurrent", "measMotorCurrent", "jointCurrent", "measCurrent"):
        if not hasattr(qarm_hw, name):
            continue
        arr = _as_float_array(getattr(qarm_hw, name))
        if arr is None:
            continue
        if arr.size >= 5:
            return arr[:5].astype(float)
        return np.pad(arr.astype(float), (0, 5 - arr.size), mode="constant", constant_values=np.nan)
    return np.full(5, np.nan)


def read_gripper_current(qarm_hw) -> float:
    joints = read_joint_currents(qarm_hw)
    if joints.size >= 5 and np.isfinite(joints[4]):
        return float(abs(joints[4]))
    return float("nan")


def read_total_arm_current(qarm_hw) -> float:
    joints = read_joint_currents(qarm_hw)
    if joints.size == 0:
        return float("nan")
    finite = joints[np.isfinite(joints)]
    if finite.size == 0:
        return float("nan")
    return float(np.sum(np.abs(finite)))


@dataclass
class MotionSupervisionLimits:
    # Gripper-current thresholds (A)
    gripper_warn_a: float = 0.45
    gripper_hard_a: float = 0.50
    gripper_emergency_a: float = 0.60

    # Total-arm current thresholds (A)
    total_warn_a: float = 2.6
    total_hard_a: float = 2.9
    total_emergency_a: float = 3.1

    # Recovery behavior
    debounce_samples: int = 8
    recover_debounce_samples: int = 4
    freeze_timeout_s: float = 1.5
    relax_step: float = 0.005
    min_grip: float = 0.10
    max_grip: float = 0.90
    warn_relax_enabled: bool = True
    warn_relax_step: float = 0.0015
    warn_relax_debounce_samples: int = 4


class MotionGripSupervisor:
    """
    Continuous move-time current supervisor.
    States:
      - ok
      - freeze_recovering
      - unrecoverable
    """

    def __init__(
        self,
        limits: MotionSupervisionLimits,
        initial_grip: float,
        label: str = "",
    ):
        self.limits = limits
        self.label = str(label)
        self.state = "ok"
        self.current_grip = _clip_grip(initial_grip, GripCurrentLimits(min_grip=limits.min_grip, max_grip=limits.max_grip))

        self._hard_count = 0
        self._warn_count = 0
        self._recover_count = 0
        self._warned = False
        self._freeze_started_s = None

    def _clip(self, grip_cmd: float) -> float:
        return float(max(self.limits.min_grip, min(self.limits.max_grip, float(grip_cmd))))

    def _start_freeze(self) -> None:
        if self.state != "freeze_recovering":
            self.state = "freeze_recovering"
            self._freeze_started_s = time.time()
            self._recover_count = 0

    def update(
        self,
        gripper_current_a: float,
        total_current_a: float,
        grip_cmd: float,
        relax_scale: float = 1.0,
        now_s: float | None = None,
    ) -> dict:
        if now_s is None:
            now_s = time.time()

        g = float(gripper_current_a) if np.isfinite(gripper_current_a) else float("nan")
        t = float(total_current_a) if np.isfinite(total_current_a) else float("nan")

        event = None
        reason = "ok"
        action = "advance"
        self.current_grip = self._clip(grip_cmd)

        if self.state == "unrecoverable":
            return {
                "state": self.state,
                "action": "abort",
                "event": "unrecoverable",
                "reason": "already_unrecoverable",
                "grip_cmd": self.current_grip,
                "gripper_current_a": g,
                "total_current_a": t,
            }

        if self.state == "ok":
            warn_hit_grip = bool(np.isfinite(g) and g >= float(self.limits.gripper_warn_a))
            warn_hit_total = bool(np.isfinite(t) and t >= float(self.limits.total_warn_a))
            warn_hit = bool(warn_hit_grip or warn_hit_total)
            if warn_hit and not self._warned:
                event = "warn"
                self._warned = True
            # Warn-relax is intentionally driven by gripper current only.
            # Total-arm warn can be high for reasons unrelated to gripper squeeze.
            if warn_hit_grip:
                self._warn_count += 1
            else:
                self._warn_count = 0

            emergency = (
                (np.isfinite(g) and g >= float(self.limits.gripper_emergency_a))
                or (np.isfinite(t) and t >= float(self.limits.total_emergency_a))
            )
            if emergency:
                self._start_freeze()
                reason = "emergency_threshold"

            hard_hit = (
                (np.isfinite(g) and g >= float(self.limits.gripper_hard_a))
                or (np.isfinite(t) and t >= float(self.limits.total_hard_a))
            )
            if hard_hit:
                self._hard_count += 1
            else:
                self._hard_count = 0

            if self._hard_count >= int(max(1, self.limits.debounce_samples)):
                self._start_freeze()
                reason = "hard_threshold_debounce"

            if (
                self.state == "ok"
                and bool(self.limits.warn_relax_enabled)
                and int(self._warn_count) >= int(max(1, self.limits.warn_relax_debounce_samples))
            ):
                step_scale = max(0.0, float(relax_scale))
                warn_relax_step = float(max(0.0, self.limits.warn_relax_step)) * step_scale
                if warn_relax_step > 0.0:
                    self.current_grip = self._clip(self.current_grip - warn_relax_step)
                    action = "advance"
                    reason = "warn_persistent_relax"
                    event = "warn_relax" if event is None else event
                self._warn_count = 0

            if self.state == "freeze_recovering":
                action = "freeze"
                if event is None:
                    event = "freeze"

        if self.state == "freeze_recovering":
            step_scale = max(0.0, float(relax_scale))
            relax_step = float(self.limits.relax_step) * step_scale
            gripper_overloaded = bool(np.isfinite(g) and g >= float(self.limits.gripper_warn_a))
            if gripper_overloaded:
                self.current_grip = self._clip(self.current_grip - relax_step)
            elif event is None:
                event = "freeze_hold_grip"
            action = "freeze"

            recovered = (
                (not np.isfinite(g) or g <= float(self.limits.gripper_warn_a))
                and (not np.isfinite(t) or t <= float(self.limits.total_warn_a))
            )
            if recovered:
                self._recover_count += 1
            else:
                self._recover_count = 0

            if self._recover_count >= int(max(1, self.limits.recover_debounce_samples)):
                self.state = "ok"
                action = "advance"
                event = "recover"
                self._hard_count = 0
                self._recover_count = 0
                self._warned = False
                reason = "recovered"

            freeze_elapsed = (float(now_s) - float(self._freeze_started_s)) if self._freeze_started_s is not None else 0.0
            if freeze_elapsed > float(self.limits.freeze_timeout_s):
                self.state = "unrecoverable"
                action = "abort"
                event = "unrecoverable"
                reason = "freeze_timeout"

        return {
            "state": self.state,
            "action": action,
            "event": event,
            "reason": reason,
            "grip_cmd": self.current_grip,
            "gripper_current_a": g,
            "total_current_a": t,
        }


def close_gripper_with_current_guard(
    arm_wrapper,
    target_xyz,
    grip_start: float,
    grip_target: float,
    limits: GripCurrentLimits,
):
    """
    Ramp gripper while holding XYZ pose and stop/relax by current thresholds.

    arm_wrapper requirements:
    - .ik(xyz, wrist_angle) -> 4 joint command
    - ._write(phi_cmd, grip_cmd)
    - .sample_time
    - .arm (QArm hw object with measured current signals)
    """
    xyz = np.asarray(target_xyz[:3], dtype=float)
    phi_target = np.asarray(arm_wrapper.ik(xyz, 0.0), dtype=float).reshape(-1)[:4]

    grip = _clip_grip(grip_start, limits)
    target = _clip_grip(grip_target, limits)
    if target < grip:
        grip = target

    t0 = time.time()
    i_peak = 0.0
    i_last = float("nan")
    hard_count = 0
    warn_count = 0
    detect_count = 0
    miss_count = 0
    warn_triggered = False
    detect_blocked_logged = False
    overcurrent_blocked_logged = False
    status = "timeout"

    while (time.time() - t0) < float(limits.max_close_s):
        step_start = time.time()

        arm_wrapper._write(phi_target, grip)
        i_last = read_gripper_current(arm_wrapper.arm)
        if np.isfinite(i_last):
            i_peak = max(i_peak, float(i_last))

        elapsed = time.time() - t0
        if elapsed >= float(limits.transient_ignore_s) and np.isfinite(i_last):
            overcurrent_allowed = bool(grip >= float(max(0.0, limits.min_overcurrent_grip_cmd)))
            if i_last >= float(limits.grip_warn_a) and not overcurrent_allowed and not overcurrent_blocked_logged:
                overcurrent_blocked_logged = True
                print(
                    "[GripCurrent] overcurrent sensed while gated by min grip "
                    f"i_last={i_last:.3f}A peak={i_peak:.3f}A warn={float(limits.grip_warn_a):.3f}A hard={float(limits.grip_hard_a):.3f}A "
                    f"emerg={float(limits.emergency_trip_a):.3f}A grip={grip:.3f} < min_overcurrent={float(limits.min_overcurrent_grip_cmd):.3f}"
                )

            if overcurrent_allowed:
                if i_last >= float(limits.grip_warn_a) and not warn_triggered:
                    warn_triggered = True
                    print(f"[GripCurrent] WARN at {i_last:.3f} A (>= {limits.grip_warn_a:.3f} A)")
                warn_count = warn_count + 1 if i_last >= float(limits.grip_warn_a) else 0

                if i_last >= float(limits.emergency_trip_a):
                    grip = _clip_grip(grip - float(limits.relax_step), limits)
                    arm_wrapper._write(phi_target, grip)
                    status = "overcurrent_emergency"
                    break

                hard_count = hard_count + 1 if i_last >= float(limits.grip_hard_a) else 0
                if hard_count >= int(limits.debounce_samples):
                    grip = _clip_grip(grip - float(limits.relax_step), limits)
                    arm_wrapper._write(phi_target, grip)
                    status = "overcurrent"
                    break

                if (
                    bool(limits.warn_relax_enabled)
                    and warn_count >= int(max(1, limits.warn_relax_debounce_samples))
                ):
                    grip = _clip_grip(grip - float(max(0.0, limits.warn_relax_step)), limits)
                    arm_wrapper._write(phi_target, grip)
                    warn_count = 0
            else:
                warn_count = 0
                hard_count = 0

            detect_allowed = bool(grip >= float(limits.min_detect_grip_cmd))
            if i_last >= float(limits.grip_detect_a) and not detect_allowed and not detect_blocked_logged:
                detect_blocked_logged = True
                print(
                    "[GripCurrent] detect gated by min grip "
                    f"(grip={grip:.3f} < min_detect={float(limits.min_detect_grip_cmd):.3f})"
                )
            detect_count = detect_count + 1 if (i_last >= float(limits.grip_detect_a) and detect_allowed) else 0
            if detect_count >= int(limits.debounce_samples):
                status = "gripped"
                break

        if grip < target:
            grip = _clip_grip(grip + float(limits.grip_step), limits)
        else:
            if np.isfinite(i_last) and i_last < float(limits.grip_miss_max_a):
                miss_count += 1
                if miss_count >= int(limits.debounce_samples):
                    status = "miss"
                    break
            else:
                miss_count = 0

        time.sleep(max(0.0, float(arm_wrapper.sample_time) - (time.time() - step_start)))

    hold_until = time.time() + float(limits.final_hold_s)
    while time.time() < hold_until:
        arm_wrapper._write(phi_target, grip)
        time.sleep(max(0.0, float(arm_wrapper.sample_time)))

    return {
        "status": status,
        "final_grip": float(grip),
        "gripper_current_peak_a": float(i_peak),
        "gripper_current_last_a": float(i_last) if np.isfinite(i_last) else float("nan"),
        "warn_triggered": bool(warn_triggered),
        "min_detect_grip_cmd": float(limits.min_detect_grip_cmd),
    }
