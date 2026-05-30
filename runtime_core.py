#!/usr/bin/env python3
# ============================= Imports / paths =============================
import time
import math
import os
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
import numpy as np
import cv2
from collections import deque

RUNTIME_BASE_DIR = Path(__file__).resolve().parent
if str(RUNTIME_BASE_DIR) not in sys.path:
    # Force local refactored runtime modules/packages to resolve first.
    sys.path.insert(0, str(RUNTIME_BASE_DIR))

from current_sensing_grip_limiting import (
    GripCurrentLimits,
    MotionGripSupervisor,
    MotionSupervisionLimits,
    close_gripper_with_current_guard,
    read_total_arm_current,
)
from centering_controller import (
    CENTER_EY_DOWN_DPHI_SCALE,
    CENTER_EY_DOWN_SIGN,
    CENTER_EY_I_CLAMP,
    CENTER_EY_I_DECAY,
    CENTER_EY_I_DOWN_SCALE,
    CENTER_EY_I_ENABLE_ABS_PX,
    CENTER_EY_I_RANGE_GATE_ENABLED,
    CENTER_EY_KI,
    CENTER_EY_KI_ELBOW,
    CENTER_MOVE_SETTLE_S,
    KELBOW,
    KSHOULDER,
    KYAW,
    MAX_JOINT_NUDGE,
    _compute_centering_nudge,
    _leak_centering_integrator,
    _maybe_apply_centering_nudge,
    _reset_centering_integrator,
)
from vision_runtime import (
    YOLODetector,
    YOLO_IOU_NMS,
    YOLO_STRETCH_SIZE,
    YOLO_STRETCH_SQUARE,
    split_merged_stack_candidates,
)
from projection_geometry import (
    apply_scan_base_xy_offset,
    base_to_camera_T,
    estimate_base_xyz_from_uv_fast,
    get_cam_offsets,
    project_candidates_to_base,
    robust_depth_m,
    set_cam_offsets,
    uvz_to_xyz_cam,
)
from cube_tracking import (
    estimate_visible_section_counts_from_tracks,
    nearest_visible_track_by_uv,
    select_intended_track_for_pick,
    update_cube_tracks,
)
from centering import (
    _draw_center_reference_overlay,
    _draw_center_stability_overlay,
    _draw_forbidden_uv_overlay,
    _show_center_frame,
    _handle_selected_center_candidate,
    center_object_slowly,
    center_object_on_expected_slot,
)
from verify_v2 import (
    _filter_verify_candidates,
    compute_verify_stack_min_z,
    compute_verify_z_margin,
    _filter_projected_slot_candidates,
    collect_slot_observations,
    associate_newest_placement,
    score_place_geometry,
    verify_last_place_reliability,
)
from planner_io import (
    build_live_policy_brain,
    choose_candidate_near_uv,
    choose_track_candidate_near_uv,
    collect_track_measurement_verify_style,
    build_prompted_allowed_commands,
    can_pick_misplaced_cube_now,
    can_pick_misplaced_on_side_now,
    _get_confirmed_entry_color_for_planner,
    get_section_confirmed_color_sequence_bottom_to_top,
    _planner_section_row_unified,
    _slots_from_level_and_sequence,
    _remove_nearest_placed_target,
    build_prompted_step_allowed_commands,
    maybe_append_policy_raw_row,
    build_prompted_planner_state,
)
from stack_scene import (
    _append_unique_reconcile_row,
    _clear_last_popped_xy,
    _clear_locked_stack_anchor_xyz,
    _compact_section_truth_row,
    _get_last_popped_xy,
    _get_locked_stack_anchor_xyz,
    _infer_section_for_base_y,
    _infer_section_for_place_xy,
    _infer_section_for_place_y,
    _is_confirmed_active_placement,
    _is_stack_confirmed_active_placement,
    _ledger_section_truth_row,
    _merge_hydrated_section_row_keep_known,
    _normalize_hydrated_section_row,
    _place_space_y_band_bounds,
    _resolve_verify_expected_color,
    _row_color_for_reconcile,
    _scene_sections_for_side,
    _section_snapshot_signature,
    _set_authoritative_section_sequence,
    _set_last_popped_xy,
    _set_locked_stack_anchor_xyz,
    _startup_default_hydrated_section_row,
    _sync_last_begin_hydrated_stacks,
    _verify_section_xy_centers,
    _verify_section_y_centers,
    append_authoritative_stack_cube,
    apply_startup_stack_hydration,
    build_place_grid_slots,
    get_authoritative_stack_levels,
    get_latest_confirmed_active_stack_placement,
    get_latest_side_stack_anchor_xyz,
    log_stack_anchor_missing,
    get_commanded_stack_base_xyz,
    get_verified_stack_base_xyz,
    commit_commanded_stack_base_anchor_from_place,
    commit_verified_stack_base_anchor_from_place,
    get_place_slots,
    get_startup_hydrated_section_row,
    infer_stack_layers_from_measurement,
    next_slot_in_section,
    pop_authoritative_stack_top,
    reconcile_scene,
    remeasure_stack_xyz_after_center,
    remeasure_stack_xyz_until_stable,
    run_place_space_truth_pass,
    run_startup_stack_identity_pass,
    section_slot_groups,
    _extract_valid_z,
)
from misplaced_actions import (
    execute_pick_misplaced_cube_action,
    execute_return_placed_cube_correction,
    run_return_handoff_stage,
    run_return_verify_and_handoff_session,
    run_return_verify_stage,
    run_track_handoff_session,
)
from pick_actions import (
    acquire_and_center_intended_cube,
    goto_correction_drop_transit,
    retreat_after_correction_drop,
    run_grasp_and_carry_common,
    run_pick_center_cycle,
    run_pick_other_session,
)
from place_actions import (
    PlacePlan,
    _build_place_plan,
    _compute_place_transit_extra_m,
    _execute_place_plan,
    _goto_place_pose,
    _goto_place_vertical_segment,
    _resolve_place_target_xyz,
    _validate_place_target,
    execute_prompted_place_action,
    execute_return_cube_action,
    placement_clearance_ok,
    safe_place,
    slot_safety_status,
    slot_target_xyz,
)
# --- Quanser (from your example) ---
from pal.products.qarm import QArm
from hal.products.qarm import QArmUtilities
# --- RealSense ---
import pyrealsense2 as rs
try:
    from llm_commander.planner.live_policy_brain import LivePolicyBrain, LivePolicyConfig
    LIVE_POLICY_IMPORT_ERROR = ""
except Exception as _direct_import_exc:
    try:
        from llm_commander.planner import LivePolicyBrain, LivePolicyConfig
        LIVE_POLICY_IMPORT_ERROR = ""
    except Exception as _planner_import_exc:
        LivePolicyBrain = None
        LivePolicyConfig = None
        LIVE_POLICY_IMPORT_ERROR = (
            f"direct_import={type(_direct_import_exc).__name__}: {_direct_import_exc} | "
            f"planner_import={type(_planner_import_exc).__name__}: {_planner_import_exc}"
        )

DEFAULT_RESULTS_ROOT = RUNTIME_BASE_DIR / "Test Results"
DEFAULT_YOLO_MODEL_PATH = RUNTIME_BASE_DIR / "best.pt"
DEFAULT_POLICY_PROMPT_PATH = RUNTIME_BASE_DIR / "llm_commander" / "prompts" / "live_sort_operator_v21.txt"
# Baseline de-bloat checklist (implementation guardrails):
# - Keep CLI and env variable compatibility stable.
# - Refactor large procedural blocks into smaller staged helpers.
# - Keep prompted runtime path minimal and explicit.
# ============================= Tunables =============================
SAMPLE_RATE_HZ = 200
# --- Key task-space poses: [x, y, z, grip] ---
HOME = np.array([0.45, 0.00, 0.49, 0.0])          # default safe pose
LOOKING = np.array([0.25, 0.00, 0.18, 0.0])       # neutral vision pose
PICK_LOOKING = LOOKING + np.array([0.0, -0.12, 0.0, 0.0])   # pick-side scan
PLACE_LOOKING = LOOKING + np.array([0.0, 0.12, 0.0, 0.0])   # place-side scan
CARRY_MID = np.array([0.40, 0.00, 0.30, 0.0])     # one carry waypoint before place
# --- Global safety and measurement ---
MAX_REACH_M = 0.641            # max radial reach in meters
MIN_PLACE_REACH_M = 0.30       # do not place too close to base/arm region
TABLE_Z_SAT_M = 0.03           # minimum commanded z to avoid table contact
N_MEAS = 8                     # depth samples per measurement
MEAS_MEDIAN_WIN = 5            # depth patch window size for robust depth
# --- Gripper / sensing hard limits ---
MAX_GRIP_CMD = float(os.getenv("QARM_MAX_GRIP_CMD", "0.58"))  # hardware grip command limit
PREGRASP_LIFT = 0.2            # vertical clearance before descend
CENTERED_FRAMES_REQUIRED = int(os.getenv("QARM_CENTERED_FRAMES_REQUIRED", "3"))
EMPTY_SCENE_CONFIRM_PASSES = 2
MAX_CYCLES_WITHOUT_PLACE_PROGRESS = 3
YOLO_MODEL_PATH = str(DEFAULT_YOLO_MODEL_PATH)
TARGET_CLASSES = ("Cube",)
YOLO_CONF = float(os.getenv("QARM_YOLO_CONF", "0.45"))  # base detector threshold (Bot-SORT low-conf cascade)
SHOW_WINDOW = True
WINDOW_NAME = "YOLO Detection - QArm"
UI_MODE = os.getenv("QARM_UI_MODE", "minimal").strip().lower()
UI_DRAW_ALL_BOXES = os.getenv("QARM_UI_DRAW_ALL_BOXES", "1").strip().lower() in {"1", "true", "yes", "on"}
UI_SHOW_TRACK_IDS = os.getenv("QARM_UI_SHOW_TRACK_IDS", "1").strip().lower() in {"1", "true", "yes", "on"}
UI_SHOW_ALL_TRACK_IDS_MINIMAL = os.getenv("QARM_UI_SHOW_ALL_TRACK_IDS_MINIMAL", "1").strip().lower() in {"1", "true", "yes", "on"}
STARTUP_VERIFY_SHOW_WINDOW = os.getenv("QARM_STARTUP_VERIFY_SHOW_WINDOW", "1").strip().lower() in {"1", "true", "yes", "on"}
# Two color columns (left/right sections) with one base slot each = logical 1x2 layout.
# In this generator, columns are along Y (`PLACE_GRID_COLS`) and rows are along X (`PLACE_GRID_ROWS`).
PLACE_GRID_ROWS = 1
PLACE_GRID_COLS = 2
PLACE_GRID_CENTER_X_M = 0.41

def _env_bool(name: str, default: bool) -> bool:
    default_text = "1" if default else "0"
    return os.getenv(name, default_text).strip().lower() in {"1", "true", "yes", "on"}

def _env_csv_set(name: str, default: str) -> set[str]:
    raw = os.getenv(name, default)
    out: set[str] = set()
    for part in str(raw).split(","):
        token = str(part).strip().lower()
        if token:
            out.add(token)
    return out

def _parse_float_tuple_env(name: str, default: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in os.getenv(name, default).split(",") if part.strip())

# ============================= Tuning: Gripper Close Loop =============================
# Close-time current guard while squeezing at grasp pose.
GRIP_DEFAULT = float(os.getenv("QARM_GRIP_DEFAULT", "0.58"))  # Gripper command (unitless), not current.
GRIP_HARD_A = float(os.getenv("QARM_GRIP_HARD_A", "0.61"))  # A
GRIP_WARN_A = float(os.getenv("QARM_GRIP_WARN_A", "0.58"))  # A
GRIP_EMERGENCY_A = float(os.getenv("QARM_GRIP_EMERGENCY_A", "0.64"))  # A (current), different quantity from GRIP_DEFAULT.
GRIP_DETECT_A = float(os.getenv("QARM_GRIP_DETECT_A", "0.45"))
GRIP_MIN_DETECT_CMD = float(os.getenv("QARM_GRIP_MIN_DETECT_CMD", "0.55"))
GRIP_MIN_OVERCURRENT_CMD = float(os.getenv("QARM_GRIP_MIN_OVERCURRENT_CMD", "0.30"))
GRIP_MISS_MAX_A = float(os.getenv("QARM_GRIP_MISS_MAX_A", "0.16"))
GRIP_DEBOUNCE_SAMPLES = int(os.getenv("QARM_GRIP_DEBOUNCE", "6"))
GRIP_TRANSIENT_IGNORE_S = float(os.getenv("QARM_GRIP_TRANSIENT_IGNORE_S", "0.0"))
GRIP_MAX_CLOSE_S = float(os.getenv("QARM_GRIP_MAX_CLOSE_S", "2.00"))
GRIP_FINAL_HOLD_S = float(os.getenv("QARM_GRIP_FINAL_HOLD_S", "0.12"))
GRIP_STEP = float(os.getenv("QARM_GRIP_STEP", "0.005"))
GRIP_RELAX_STEP = float(os.getenv("QARM_GRIP_RELAX_STEP", "0.008"))
GRIP_WARN_RELAX_ENABLED = _env_bool("QARM_GRIP_WARN_RELAX_ENABLED", True)
GRIP_WARN_RELAX_STEP = float(os.getenv("QARM_GRIP_WARN_RELAX_STEP", "0.006"))
GRIP_WARN_RELAX_DEBOUNCE = int(os.getenv("QARM_GRIP_WARN_RELAX_DEBOUNCE", "3"))
GRIP_HARD_SWEEP_HINT_A = _parse_float_tuple_env("QARM_GRIP_HARD_SWEEP_A", "0.55,0.57,0.59")
GRIP_SOFT_RETRY_ENABLED = _env_bool("QARM_GRIP_SOFT_RETRY_ENABLED", False)
GRIP_SOFT_RETRY_NEAR_CONTACT_A = float(os.getenv("QARM_GRIP_SOFT_RETRY_NEAR_CONTACT_A", "0.10"))
GRIP_SOFT_RETRY_REOPEN_DELTA = float(os.getenv("QARM_GRIP_SOFT_RETRY_REOPEN_DELTA", "0.015"))
GRIP_SOFT_RETRY_MAX_CLOSE_EXTRA_S = float(os.getenv("QARM_GRIP_SOFT_RETRY_MAX_CLOSE_EXTRA_S", "0.45"))
GRIP_SOFT_RETRY_STEP_SCALE = float(os.getenv("QARM_GRIP_SOFT_RETRY_STEP_SCALE", "0.90"))
GRIP_SOFT_RETRY_DETECT_SCALE = float(os.getenv("QARM_GRIP_SOFT_RETRY_DETECT_SCALE", "0.98"))
GRIP_SOFT_RETRY_TRANSIENT_IGNORE_S = float(os.getenv("QARM_GRIP_SOFT_RETRY_TRANSIENT_IGNORE_S", "0.0"))
GRIP_MIN_SUCCESS_HOLD_CMD = float(os.getenv("QARM_GRIP_MIN_SUCCESS_HOLD_CMD", "0.54"))

# ============================= Tuning: Move-Time Current Guard =============================
# Current supervision while arm is moving/carrying.
MOTION_GUARD_ENABLED = _env_bool("QARM_MOTION_GUARD_ENABLED", True)
MOTION_GRIP_WARN_A = float(os.getenv("QARM_MOVE_GRIP_WARN_A", f"{GRIP_WARN_A:.3f}"))
MOTION_GRIP_HARD_A = float(os.getenv("QARM_MOVE_GRIP_HARD_A", f"{GRIP_HARD_A:.3f}"))
MOTION_GRIP_EMERGENCY_A = float(os.getenv("QARM_MOVE_GRIP_EMERGENCY_A", f"{GRIP_EMERGENCY_A:.3f}"))
MOTION_TOTAL_WARN_A = float(os.getenv("QARM_MOVE_TOTAL_WARN_A", "2.9"))  # A (sum of |joint currents|)
MOTION_TOTAL_HARD_A = float(os.getenv("QARM_MOVE_TOTAL_HARD_A", "3.7"))  # A
MOTION_TOTAL_EMERGENCY_A = float(os.getenv("QARM_MOVE_TOTAL_EMERGENCY_A", "3.85"))  # A; default comes from QARM_MOVE_TOTAL_EMERGENCY_A.
MOTION_DEBOUNCE_SAMPLES = int(os.getenv("QARM_MOVE_DEBOUNCE", "8"))
MOTION_FREEZE_TIMEOUT_S = float(os.getenv("QARM_MOVE_FREEZE_TIMEOUT_S", "2.0"))
MOTION_RELAX_STEP = float(os.getenv("QARM_MOVE_RELAX_STEP", "0.0030"))
MOTION_RELAX_TOP_MULT = float(os.getenv("QARM_MOVE_RELAX_TOP_MULT", "1.6"))
MOTION_WARN_RELAX_ENABLED = _env_bool("QARM_MOVE_WARN_RELAX_ENABLED", True)
MOTION_WARN_RELAX_STEP = float(os.getenv("QARM_MOVE_WARN_RELAX_STEP", "0.0035"))
MOTION_WARN_RELAX_DEBOUNCE = int(os.getenv("QARM_MOVE_WARN_RELAX_DEBOUNCE", "3"))

# ============================= Tuning: Pick Verification =============================
POST_LIFT_VERIFY_SAMPLES = 18
POST_LIFT_VERIFY_MIN_HITS = 8
POST_LIFT_VERIFY_MIN_CURRENT_A = float(os.getenv("QARM_POST_LIFT_MIN_A", "0.105"))
POST_LIFT_VERIFY_HOLD_S = 0.25
POST_LIFT_TUNE_ENABLED = _env_bool("QARM_POST_LIFT_TUNE_ENABLED", True)
POST_LIFT_TUNE_TARGET_A = float(
    os.getenv("QARM_POST_LIFT_TUNE_TARGET_A", "0.150")
)
POST_LIFT_TUNE_MAX_STEPS = int(os.getenv("QARM_POST_LIFT_TUNE_MAX_STEPS", "8"))
POST_LIFT_TUNE_STEP = float(os.getenv("QARM_POST_LIFT_TUNE_STEP", "0.0025"))
POST_LIFT_TUNE_SAMPLES = int(os.getenv("QARM_POST_LIFT_TUNE_SAMPLES", "4"))
POST_LIFT_CARRY_BLEED_ENABLED = _env_bool("QARM_POST_LIFT_CARRY_BLEED_ENABLED", True)
POST_LIFT_CARRY_TARGET_A = float(os.getenv("QARM_POST_LIFT_CARRY_TARGET_A", "0.20"))
POST_LIFT_CARRY_MAX_A = float(os.getenv("QARM_POST_LIFT_CARRY_MAX_A", "0.28"))
POST_LIFT_CARRY_GRIP_TRIGGER_CMD = float(
    os.getenv("QARM_POST_LIFT_CARRY_GRIP_TRIGGER_CMD", str(MAX_GRIP_CMD))
)
POST_LIFT_CARRY_MAX_STEPS = int(os.getenv("QARM_POST_LIFT_CARRY_MAX_STEPS", "6"))
POST_LIFT_CARRY_TUNE_STEP = float(os.getenv("QARM_POST_LIFT_CARRY_TUNE_STEP", "0.002"))
POST_LIFT_CARRY_MIN_GRIP_CMD = float(
    os.getenv("QARM_POST_LIFT_CARRY_MIN_GRIP_CMD", str(GRIP_MIN_SUCCESS_HOLD_CMD))
)
POST_LIFT_CARRY_BLEED_MOVE_TIME = _env_bool("QARM_POST_LIFT_CARRY_BLEED_MOVE_TIME", True)
CORRECTION_RETREAT_HOME_ENABLED = _env_bool("QARM_CORRECTION_RETREAT_HOME_ENABLED", True)
CORRECTION_DROP_TRANSIT_ENABLED = _env_bool("QARM_CORRECTION_DROP_TRANSIT_ENABLED", False)
CORRECTION_DROP_TRANSIT = np.array(
    [
        float(os.getenv("QARM_CORRECTION_DROP_TRANSIT_X_M", "0.45")),
        float(os.getenv("QARM_CORRECTION_DROP_TRANSIT_Y_M", "-0.05")),
        float(os.getenv("QARM_CORRECTION_DROP_TRANSIT_Z_M", "0.32")),
        0.0,
    ],
    dtype=float,
)
PICK_PLACED_SOLE_TRACK_RETRIES = max(
    0,
    int(os.getenv("QARM_PICK_PLACED_SOLE_TRACK_RETRIES", "2")),
)
# pick_placed_left/right lock stage (handoff center + verify); longer than generic pick_misplaced.
PICK_PLACED_LOCK_TIMEOUT_S = float(os.getenv("QARM_PICK_PLACED_LOCK_TIMEOUT_S", "10.0"))
PICK_PLACED_MAX_TOTAL_LOCK_TIME_S = float(
    os.getenv("QARM_PICK_PLACED_MAX_TOTAL_LOCK_TIME_S", "30.0")
)

# ============================= Tuning: Mission/Cycle Limits =============================
MIN_PLACE_SLOT_SEPARATION_M = float(os.getenv("QARM_PLACE_MIN_SEP_M", "0.075"))
CENTER_TIMEOUT_S = float(os.getenv("QARM_CENTER_TIMEOUT_S", "7.0"))
CENTER_TIMEOUT_ACTIVE_S = float(
    os.getenv("QARM_CENTER_ACTIVE_TIMEOUT_S", os.getenv("QARM_CENTER_TIMEOUT_S", "5.0"))
)
CENTER_TIMEOUT_NO_DETECTION_S = float(os.getenv("QARM_CENTER_NO_DETECTION_TIMEOUT_S", "5.0"))
PICK_ORIENTATION_CHECK_AFTER_GRASP_FAILS = max(
    1,
    int(os.getenv("QARM_PICK_ORIENTATION_CHECK_AFTER_GRASP_FAILS", "1")),
)
PICK_ORIENTATION_CHECK_AFTER_ACTIVE_MISSES = max(
    1,
    int(os.getenv("QARM_PICK_ORIENTATION_CHECK_AFTER_ACTIVE_MISSES", "1")),
)
PICK_PLACED_EMPTY_COOLDOWN_STEPS = int(
    os.getenv("QARM_PICK_PLACED_EMPTY_COOLDOWN_STEPS", "2")
)
PICK_CORRECTION_FAIL_HYDRATE_REFRESH_ENABLED = _env_bool(
    "QARM_PICK_CORRECTION_FAIL_HYDRATE_REFRESH_ENABLED",
    False,
)

# ============================= Tuning: Offset Auto-Tune Mode =============================
RUNTIME_MODE_DEFAULT = os.getenv("QARM_RUNTIME_MODE", "prompted").strip().lower()
CALIB_PROFILE_PATH = os.getenv("QARM_CALIB_PROFILE_PATH", "").strip()
CALIB_PROFILE_AUTO = _env_bool("QARM_CALIB_PROFILE_AUTO", True)
CALIB_PROFILE_AUTO_NAME = os.getenv("QARM_CALIB_PROFILE_AUTO_NAME", "latest.json").strip() or "latest.json"
TUNE_PROFILES_DIR = RUNTIME_BASE_DIR / "tune_profiles"
TUNE_RUNS_ROOT = Path(os.getenv("QARM_TUNE_RUNS_DIR", str(DEFAULT_RESULTS_ROOT / "tune_runs")))
TUNE_MAX_ITERS = int(os.getenv("QARM_TUNE_MAX_ITERS", "20"))
TUNE_TARGET_XY_ERR_M = float(os.getenv("QARM_TUNE_TARGET_XY_ERR_M", "0.010"))
TUNE_TARGET_Z_ERR_M = float(os.getenv("QARM_TUNE_TARGET_Z_ERR_M", "0.012"))
TUNE_CONSECUTIVE_TARGET_HITS = int(os.getenv("QARM_TUNE_CONSECUTIVE_TARGET_HITS", "3"))
TUNE_GAIN_X = float(os.getenv("QARM_TUNE_GAIN_X", "0.35"))
TUNE_GAIN_Y = float(os.getenv("QARM_TUNE_GAIN_Y", "0.35"))
TUNE_GAIN_Z = float(os.getenv("QARM_TUNE_GAIN_Z", "0.35"))
TUNE_MAX_STEP_X_M = float(os.getenv("QARM_TUNE_MAX_STEP_X_M", "0.003"))
TUNE_MAX_STEP_Y_M = float(os.getenv("QARM_TUNE_MAX_STEP_Y_M", "0.003"))
TUNE_MAX_STEP_Z_M = float(os.getenv("QARM_TUNE_MAX_STEP_Z_M", "0.003"))
TUNE_MAX_DELTA_FROM_BASELINE_M = float(os.getenv("QARM_TUNE_MAX_DELTA_FROM_BASELINE_M", "0.030"))
TUNE_WORSEN_TOL = float(os.getenv("QARM_TUNE_WORSEN_TOL", "0.0015"))
TUNE_BACKOFF = float(os.getenv("QARM_TUNE_BACKOFF", "0.60"))
TUNE_MAX_FAILS = int(os.getenv("QARM_TUNE_MAX_FAILS", "4"))
TUNE_MAX_FAIL_RECOVERIES = int(os.getenv("QARM_TUNE_MAX_FAIL_RECOVERIES", "8"))
TUNE_ENABLE_GRASP_Z_TUNE = _env_bool("QARM_TUNE_ENABLE_GRASP_Z_TUNE", True)
TUNE_GRASP_Z_GAIN = float(os.getenv("QARM_TUNE_GRASP_Z_GAIN", "0.12"))
TUNE_GRASP_Z_MAX_STEP = float(os.getenv("QARM_TUNE_GRASP_Z_MAX_STEP", "0.010"))
TUNE_GRASP_Z_FAIL_STEP = float(os.getenv("QARM_TUNE_GRASP_Z_FAIL_STEP", "0.010"))
TUNE_GRASP_Z_MIN = float(os.getenv("QARM_TUNE_GRASP_Z_MIN", "0.58"))
TUNE_GRASP_Z_MAX = float(os.getenv("QARM_TUNE_GRASP_Z_MAX", "0.90"))

# ============================= Tuning: Vision + Centering =============================
PICK_FILTER_BY_BASE_Y = _env_bool("QARM_PICK_FILTER_BY_BASE_Y", True)
PICK_MAX_BASE_Y_M = float(os.getenv("QARM_PICK_MAX_BASE_Y_M", "0.02"))
CENTER_VERBOSE = _env_bool("QARM_CENTER_VERBOSE", False)
DETECT_CONF = float(os.getenv("QARM_DETECT_CONF", "0.6"))
COMMIT_CONF = float(os.getenv("QARM_COMMIT_CONF", "0.6"))
YOLO_BBOX_SPLIT_ENABLED = _env_bool("QARM_YOLO_BBOX_SPLIT_ENABLED", False)
YOLO_BBOX_SPLIT_MIN_HEIGHT_M = float(os.getenv("QARM_YOLO_BBOX_SPLIT_MIN_HEIGHT_M", "0.05"))
YOLO_BBOX_SPLIT_MIN_ASPECT = float(os.getenv("QARM_YOLO_BBOX_SPLIT_MIN_ASPECT", "1.25"))
YOLO_BBOX_SPLIT_MAX_CUBES = int(os.getenv("QARM_YOLO_BBOX_SPLIT_MAX_CUBES", "3"))

# ============================= Tuning: Data Capture =============================
AUTO_CAPTURE_LOCALIZATION_IMAGES = _env_bool("QARM_CAPTURE_LOCALIZATION_IMAGES", False)
LOCALIZATION_CAPTURE_ROOT = os.getenv("QARM_CAPTURE_DIR", str(DEFAULT_RESULTS_ROOT / "auto_collected_localizations"))
LOCALIZATION_CAPTURE_SAVE_ANNOTATED = _env_bool("QARM_CAPTURE_ANNOTATED", True)

# ============================= Tuning: Place Motion Geometry =============================
PLACE_ROW_Y_M = float(PLACE_LOOKING[1])
PLACE_APPROACH_LIFT_M = float(os.getenv("QARM_PLACE_APPROACH_LIFT_M", "0.13"))
PLACE_NEAR_DESCENT_OFFSET_M = float(os.getenv("QARM_PLACE_NEAR_OFFSET_M", "0.025"))
PLACE_TRANSIT_STACK_START_LEVEL = int(os.getenv("QARM_PLACE_TRANSIT_STACK_START_LEVEL", "3"))
PLACE_TRANSIT_STACK_DZ_M = float(os.getenv("QARM_PLACE_TRANSIT_STACK_DZ_M", "0.020"))
PLACE_TRANSIT_STACK_MAX_EXTRA_M = float(os.getenv("QARM_PLACE_TRANSIT_STACK_MAX_EXTRA_M", "0.08"))
PLACE_RELEASE_Z_M = float(os.getenv("QARM_PLACE_RELEASE_Z_M", "0.037"))
RETURN_CUBE_Z_LIFT_M = float(os.getenv("QARM_RETURN_CUBE_Z_LIFT_M", "0.005"))
PLACE_RELEASE_TOUCH_OPEN_GRIP = float(os.getenv("QARM_PLACE_RELEASE_TOUCH_OPEN_GRIP", "0.24"))
PLACE_RELEASE_MAX_DELTA = float(os.getenv("QARM_PLACE_RELEASE_MAX_DELTA", "0.24"))
PLACE_RELEASE_OPEN_GRIP = float(os.getenv("QARM_PLACE_RELEASE_OPEN_GRIP", "0.24"))
PLACE_RELEASE_CLEARANCE_M = float(os.getenv("QARM_PLACE_RELEASE_CLEARANCE_M", "0.025"))
PLACE_OPEN_HOLD_S = float(os.getenv("QARM_PLACE_OPEN_HOLD_S", "0.22"))
PLACE_RELEASE_CLEARANCE_DURATION_S = float(os.getenv("QARM_PLACE_RELEASE_CLEARANCE_S", "0.70"))
PLACE_ALIGN_DURATION_S = float(os.getenv("QARM_PLACE_ALIGN_S", "1.6"))
PLACE_DESCEND_NEAR_DURATION_S = float(os.getenv("QARM_PLACE_DESCEND_NEAR_S", "1.1"))
PLACE_DESCEND_FINAL_DURATION_S = float(os.getenv("QARM_PLACE_DESCEND_FINAL_S", "1.5"))
PLACE_RELEASE_DURATION_S = float(os.getenv("QARM_PLACE_RELEASE_S", "1.5"))
PLACE_RETREAT_DURATION_S = float(os.getenv("QARM_PLACE_RETREAT_S", "0.75"))
PLACE_RELEASE_JOINT_TOL_DEG = float(os.getenv("QARM_PLACE_RELEASE_JOINT_TOL_DEG", "2.6"))
PLACE_RELEASE_SETTLE_S = float(os.getenv("QARM_PLACE_RELEASE_SETTLE_S", "2.6"))
PLACE_FAIL_CONTINUE_REASONS = _env_csv_set(
    "QARM_PLACE_FAIL_CONTINUE_REASONS",
    "place_release_clearance_failed,place_retreat_vertical_failed",
)
PLACE_VERTICAL_STEP_M = float(os.getenv("QARM_PLACE_VERTICAL_STEP_M", "0.006"))
PLACE_VERTICAL_STEPS_PER_SEGMENT = int(os.getenv("QARM_PLACE_VERTICAL_STEPS_PER_SEGMENT", "70"))

# ============================= Tuning: Place Grid =============================
PLACE_GRID_DX_M = float(os.getenv("QARM_PLACE_GRID_DX_M", "0.07"))
PLACE_GRID_DY_M = float(os.getenv("QARM_PLACE_GRID_DY_M", "0.0625"))
PLACE_GRID_CENTER_Y_M = float(os.getenv("QARM_PLACE_GRID_CENTER_Y_M", "0.1625"))

# ============================= Tuning: LLM Backend =============================
# Mission prompts (also mirrored in experiments/prompts.json for run_prompt_trials).
# Set QARM_MISSION_PROMPT or swap the default below. End-condition (stop_run) lines are fixed per mission.
_MISSION_PROMPT_P1_COLOR_SPLIT = (
    "1. Stack Blue cubes left.\n"
    "2. Stack Orange cubes right.\n"
    "3. Use pick_placed_left/right to make corrections."
)
_MISSION_PROMPT_P2_ORANGE_RIGHT = (
    "1. Stack all orange cubes on right stack only.\n"
    "2. Left stack should be empty.\n"
    "3. Use pick_placed_left/right to make the required correction.\n"
    "4. Use stop_run when 3 orange cubes stacked on right."
)
_MISSION_PROMPT_P3_ALTERNATING = (
    "1. Build TWO stacks, each exactly 3 cubes high.\n"
    "2. Fill LEFT stack first, then RIGHT stack.\n"
    "3. Required pattern for each stack (bottom to top): BLUE, ORANGE, BLUE.\n"
    "4. If centered cube color is not legal for the current target stack level, use pick_other before grasp.\n"
    "5. Use pick_placed_left/right only to correct a placed stack that violates the required pattern.\n"
    "6. Use stop_run only when LEFT and RIGHT are both [BLUE, ORANGE, BLUE]."
)
# Prompt 4 (push test):
# "1. Use push_cube once on the centered pick-space cube.\n"
# "2. Then use stop_run immediately."
MISSION_PROMPT = os.getenv("QARM_MISSION_PROMPT", _MISSION_PROMPT_P3_ALTERNATING)
# Keep stack height scalar defined before any env defaults that reference it.
STACK_LEVEL_DZ_M = float(os.getenv("QARM_STACK_LEVEL_DZ_M", "0.060"))
PLACE_STACK_LEVEL_DZ_M = float(os.getenv("QARM_PLACE_STACK_LEVEL_DZ_M", "0.060"))
PLACE_STACK_UPPER_EXTRA_Z_M = float(os.getenv("QARM_PLACE_STACK_UPPER_EXTRA_Z_M", "0.003"))
PLACE_STACK_LEVEL3_EXTRA_Z_M = float(os.getenv("QARM_PLACE_STACK_LEVEL3_EXTRA_Z_M", "0.005"))
MISPLACED_RETURN_DROP_X_M = float(
    os.getenv(
        "QARM_MISPLACED_RETURN_DROP_X_M",
        f"{(float(PLACE_GRID_CENTER_X_M) + float(PLACE_GRID_DX_M)):.4f}",
    )
)
MISPLACED_RETURN_DROP_Y_M = float(
    os.getenv("QARM_MISPLACED_RETURN_DROP_Y_M", f"{(float(PICK_LOOKING[1]) - 0.02):.4f}")
)
MISPLACED_RETURN_DROP_Z_LIFT_M = float(os.getenv("QARM_MISPLACED_RETURN_DROP_Z_LIFT_M", "0.010"))
MISPLACED_RETURN_DROP_Z_M = float(
    os.getenv(
        "QARM_MISPLACED_RETURN_DROP_Z_M",
        f"{max(float(TABLE_Z_SAT_M), float(PLACE_RELEASE_Z_M)) + float(MISPLACED_RETURN_DROP_Z_LIFT_M):.4f}",
    )
)
MISPLACED_RETURN_GRID_COLS = max(
    1,
    int(os.getenv("QARM_MISPLACED_RETURN_GRID_COLS", "2")),
)
MISPLACED_RETURN_GRID_MAX_SLOTS = max(
    1,
    int(os.getenv("QARM_MISPLACED_RETURN_GRID_MAX_SLOTS", "4")),
)
MISPLACED_RETURN_GRID_DX_M = float(
    os.getenv("QARM_MISPLACED_RETURN_GRID_DX_M", "-0.115")
)
MISPLACED_RETURN_GRID_DY_M = float(
    os.getenv("QARM_MISPLACED_RETURN_GRID_DY_M", "-0.125")
)
MISPLACED_PICK_REQUIRED_HITS = int(os.getenv("QARM_MISPLACED_PICK_REQUIRED_HITS", "8"))
MISPLACED_PICK_MEASURE_SAMPLES = int(os.getenv("QARM_MISPLACED_PICK_MEASURE_SAMPLES", "8"))
MISPLACED_PICK_HARD_TIMEOUT_S = float(os.getenv("QARM_MISPLACED_PICK_HARD_TIMEOUT_S", "6.0"))
MISPLACED_PICK_LOCK_COMMIT_CONF = float(os.getenv("QARM_MISPLACED_PICK_LOCK_COMMIT_CONF", "0.82"))
MISPLACED_PICK_MISMATCH_STRIKES_REQUIRED = max(
    1,
    int(os.getenv("QARM_MISPLACED_PICK_MISMATCH_STRIKES_REQUIRED", "2")),
)
PICK_PLACED_HANDOFF_SECTION_HARD_FILTER = _env_bool(
    "QARM_PICK_PLACED_HANDOFF_SECTION_HARD_FILTER", False
)
PICK_PLACED_VERIFY_STRIKES = max(
    1,
    int(os.getenv("QARM_PICK_PLACED_VERIFY_STRIKES", "1")),
)
MISPLACED_PICK_NO_VALID_DETECTIONS_FRAMES = max(
    1,
    int(os.getenv("QARM_MISPLACED_PICK_NO_VALID_DETECTIONS_FRAMES", "5")),
)
MISPLACED_PICK_POST_CENTER_REFRESH_S = float(
    os.getenv("QARM_MISPLACED_PICK_POST_CENTER_REFRESH_S", "0.8")
)
MISPLACED_PICK_POST_CENTER_REFRESH_MIN_FRAMES = int(
    os.getenv("QARM_MISPLACED_PICK_POST_CENTER_REFRESH_MIN_FRAMES", "2")
)
MISPLACED_PICK_MAX_LOCK_ATTEMPTS = max(
    1,
    int(os.getenv("QARM_MISPLACED_PICK_MAX_LOCK_ATTEMPTS", "6")),
)
MISPLACED_PICK_MAX_TOTAL_LOCK_TIME_S = float(
    os.getenv("QARM_MISPLACED_PICK_MAX_TOTAL_LOCK_TIME_S", "18.0")
)
MISPLACED_PLACE_LOOK_Y_OFFSET = float(
    os.getenv("QARM_MISPLACED_PLACE_LOOK_Y_OFFSET", "0.09")
)
MISPLACED_PICK_QUALITY_WAIT_PER_TRACK_CAP = max(
    1,
    int(os.getenv("QARM_MISPLACED_PICK_QUALITY_WAIT_PER_TRACK_CAP", "6")),
)
MISPLACED_PICK_ATTEMPT_NO_PROGRESS_CAP = max(
    1,
    int(os.getenv("QARM_MISPLACED_PICK_ATTEMPT_NO_PROGRESS_CAP", "10")),
)
MISPLACED_PICK_TRACK_ID_ONLY_LOCK = _env_bool(
    "QARM_MISPLACED_PICK_TRACK_ID_ONLY_LOCK",
    True,
)
MISPLACED_PICK_ENFORCE_PICK_SPACE_GATE = _env_bool(
    "QARM_MISPLACED_PICK_ENFORCE_PICK_SPACE_GATE",
    True,
)
MISPLACED_PICK_HEIGHT_STEP_M = float(os.getenv("QARM_MISPLACED_PICK_HEIGHT_STEP_M", "0.060"))
MISPLACED_PICK_HEIGHT_TOL_M = float(os.getenv("QARM_MISPLACED_PICK_HEIGHT_TOL_M", "0.008"))
MISPLACED_PICK_REQUIRE_TOP_LEVEL_MATCH = _env_bool(
    "QARM_MISPLACED_PICK_REQUIRE_TOP_LEVEL_MATCH",
    True,
)
MISPLACED_PICK_ALLOW_HIGHER_THAN_EXPECTED = _env_bool(
    "QARM_MISPLACED_PICK_ALLOW_HIGHER_THAN_EXPECTED",
    True,
)
MISPLACED_PICK_TOP_HEIGHT_TOL_M = float(
    os.getenv("QARM_MISPLACED_PICK_TOP_HEIGHT_TOL_M", "0.012")
)
PICK_PLACED_HANDOFF_PREFER_MAX_Z = _env_bool(
    "QARM_PICK_PLACED_HANDOFF_PREFER_MAX_Z",
    True,
)
STACK_REBUILD_LEVEL0_Y_BIAS_M = float(
    os.getenv("QARM_STACK_REBUILD_LEVEL0_Y_BIAS_M", "0.008")
)
LLM_POLICY_BACKEND = os.getenv("QARM_LLM_POLICY_BACKEND", "ollama").strip().lower()
LLM_POLICY_MODEL = os.getenv("QARM_LLM_POLICY_MODEL", "gemma4:26B").strip()
LLM_POLICY_ENDPOINT = os.getenv("QARM_LLM_POLICY_ENDPOINT", "http://127.0.0.1:11434").strip()
LLM_POLICY_TIMEOUT_S = float(os.getenv("QARM_LLM_POLICY_TIMEOUT_S", "20.0"))
LLM_POLICY_THINK = _env_bool("QARM_LLM_POLICY_THINK", False)
LLM_POLICY_MAX_REPROMPT = int(os.getenv("QARM_LLM_POLICY_MAX_REPROMPT", "1"))
LLM_POLICY_NUM_PREDICT = int(os.getenv("QARM_LLM_POLICY_NUM_PREDICT", "0"))
LLM_POLICY_PROMPT_PATH = os.getenv(
    "QARM_LLM_POLICY_PROMPT_PATH",
    str(DEFAULT_POLICY_PROMPT_PATH),
).strip()

# ============================= Tuning: Prompted Mission + Stack Controls =============================
SECTION_LEFT_NAME = os.getenv("QARM_SECTION_LEFT_NAME", "left").strip().lower()
SECTION_RIGHT_NAME = os.getenv("QARM_SECTION_RIGHT_NAME", "right").strip().lower()
SECTION_LABEL_MIRROR = _env_bool("QARM_SECTION_LABEL_MIRROR", True)
ENABLE_STACK_ACTIONS = _env_bool("QARM_ENABLE_STACK_ACTIONS", True)
# Keep physical stack capacity fixed at 3-high per side.
# If env is set lower (e.g., 2), the policy can stop early at 4 total cubes.
MAX_STACK_LEVELS_PER_SECTION = max(3, int(os.getenv("QARM_MAX_STACK_LEVELS_PER_SECTION", "3")))
STACK_RELEASE_Z_GUARD_M = float(os.getenv("QARM_STACK_RELEASE_Z_GUARD_M", "0.028"))
PROMPTED_SAFE_PICK_REACH_M = float(os.getenv("QARM_PROMPTED_SAFE_PICK_REACH_M", "0.64"))
POLICY_PRINT_RAW = _env_bool("QARM_POLICY_PRINT_RAW", True)
POLICY_PRINT_RAW_BLOCK = _env_bool("QARM_POLICY_PRINT_RAW_BLOCK", True)
POLICY_LOG_ALLOWED_COMMANDS = _env_bool("QARM_POLICY_LOG_ALLOWED_COMMANDS", True)
POLICY_TRACE_SAVE_RAW = _env_bool("QARM_POLICY_TRACE_SAVE_RAW", True)
PLANNER_INCLUDE_COLOR_SEQUENCE = _env_bool("QARM_PLANNER_INCLUDE_COLOR_SEQUENCE", False)

# ============================= Tuning: Grasp Trajectory =============================
GRASP_APPROACH_SEGMENT_S = float(os.getenv("QARM_GRASP_APPROACH_SEGMENT_S", "1.20"))
GRASP_APPROACH_STEPS = int(os.getenv("QARM_GRASP_APPROACH_STEPS", "90"))# reduced from 95
GRASP_SETTLE_BEFORE_CLOSE_S = float(os.getenv("QARM_GRASP_SETTLE_BEFORE_CLOSE_S", "0.18"))
GRASP_APPROACH_GRIP = float(os.getenv("QARM_GRASP_APPROACH_GRIP", "0.15"))
GRASP_LIFT_JOINT_TOL_DEG = float(os.getenv("QARM_GRASP_LIFT_JOINT_TOL_DEG", "3.2"))
GRASP_LIFT_SEGMENT_S = float(os.getenv("QARM_GRASP_LIFT_SEGMENT_S", "1.80"))
GRASP_LIFT_STEPS = int(os.getenv("QARM_GRASP_LIFT_STEPS", "90"))
GRASP_SETTLE_BEFORE_LIFT_S = float(os.getenv("QARM_GRASP_SETTLE_BEFORE_LIFT_S", "0.20"))
GRASP_PRE_LIFT_BLEED_ENABLED = _env_bool("QARM_GRASP_PRE_LIFT_BLEED_ENABLED", True)
GRASP_PRE_LIFT_BLEED_GRIP_TRIGGER = float(
    os.getenv("QARM_GRASP_PRE_LIFT_BLEED_GRIP_TRIGGER", str(MAX_GRIP_CMD))
)
GRASP_PRE_LIFT_BLEED_PEAK_A = float(os.getenv("QARM_GRASP_PRE_LIFT_BLEED_PEAK_A", "0.52"))
GRASP_PRE_LIFT_BLEED_MAX_STEPS = max(1, int(os.getenv("QARM_GRASP_PRE_LIFT_BLEED_MAX_STEPS", "4")))
GRASP_PRE_LIFT_BLEED_TUNE_STEP = float(os.getenv("QARM_GRASP_PRE_LIFT_BLEED_TUNE_STEP", "0.002"))
# Grasp Z: depth below measured top = (1-frac)*edge (+ optional flat add), capped at cube center.
# frac=0.50 -> midpoint of the top cube (half cube height from measured top face).
GRASP_Z_PICK_FRACTION = float(os.getenv("QARM_GRASP_Z_PICK_FRACTION", "0.50"))
GRASP_Z_PICK_FRAC_MIN = float(os.getenv("QARM_GRASP_Z_PICK_FRAC_MIN", "0.50"))
GRASP_Z_PICK_FRAC_MAX = float(os.getenv("QARM_GRASP_Z_PICK_FRAC_MAX", "0.90"))
# Optional extra depth (m); default 0 - use fraction-only unless explicitly set.
GRASP_Z_DEPTH_FROM_TOP_M = float(os.getenv("QARM_GRASP_Z_DEPTH_FROM_TOP_M", "0.000"))
GRASP_CUBE_EDGE_M = float(os.getenv("QARM_GRASP_CUBE_EDGE_M", f"{STACK_LEVEL_DZ_M:.3f}"))
GRASP_FAR_XY_Z_LIFT_ENABLED = _env_bool("QARM_GRASP_FAR_XY_Z_LIFT_ENABLED", False)
GRASP_FAR_XY_Z_LIFT_REACH_M = float(os.getenv("QARM_GRASP_FAR_XY_Z_LIFT_REACH_M", "0.550"))
GRASP_FAR_XY_Z_LIFT_M = float(os.getenv("QARM_GRASP_FAR_XY_Z_LIFT_M", "0.0000"))
GRASP_PICK_X_BIAS_M = float(os.getenv("QARM_GRASP_PICK_X_BIAS_M", "0.005"))#.1 works but sketchy
GRASP_PICK_Y_BIAS_M = float(os.getenv("QARM_GRASP_PICK_Y_BIAS_M", "0.000"))#.06
PICK_MISPLACED_GRASP_X_OFFSET_M = float(os.getenv("QARM_PICK_MISPLACED_GRASP_X_OFFSET_M", "0.000"))
PICK_MISPLACED_GRASP_Y_OFFSET_M = float(os.getenv("QARM_PICK_MISPLACED_GRASP_Y_OFFSET_M", "0.000"))
PICK_MISPLACED_GRASP_Y_PER_LEVEL_M = float(os.getenv("QARM_PICK_MISPLACED_GRASP_Y_PER_LEVEL_M", "0.005"))
PICK_MISPLACED_GRASP_Y_MAX_M = float(os.getenv("QARM_PICK_MISPLACED_GRASP_Y_MAX_M", "0.010"))
PICK_MISPLACED_GRASP_Z_OFFSET_M = float(os.getenv("QARM_PICK_MISPLACED_GRASP_Z_OFFSET_M", "0.006"))
PICK_MISPLACED_GRASP_HIGH_Z_EXTRA_M = float(
    os.getenv("QARM_PICK_MISPLACED_GRASP_HIGH_Z_EXTRA_M", "0.010")
)
# Disabled by default: stack anchor + scan XY correction carry place column (was +6 mm Y on slot place).
PLACE_Y_BIAS_M = float(os.getenv("QARM_PLACE_Y_BIAS_M", "0.0"))
# Command-time XY correction on place targets (arm). Default off - prefer scan + verify eval.
PLACE_CMD_XY_OFFSET_ENABLED = _env_bool("QARM_PLACE_CMD_XY_OFFSET_ENABLED", False)
PLACE_CMD_OFFSET_SKIP_STACK_ANCHOR = _env_bool("QARM_PLACE_CMD_OFFSET_SKIP_STACK_ANCHOR", False) #Was true, left stack fell
PLACE_CMD_X_OFFSET_M = float(os.getenv("QARM_PLACE_CMD_X_OFFSET_M", "0.007"))
PLACE_CMD_Y_OFFSET_M = float(os.getenv("QARM_PLACE_CMD_Y_OFFSET_M", "-0.021"))
STACK_X_LEVEL_OFFSET_ENABLED = _env_bool("QARM_STACK_X_LEVEL_OFFSET_ENABLED", False)
STACK_X_LEVEL1_OFFSET = float(os.getenv("QARM_STACK_X_LEVEL1_OFFSET", "0.005"))
STACK_X_LEVEL2_OFFSET = float(os.getenv("QARM_STACK_X_LEVEL2_OFFSET", "0.010"))
STACK_PICK_X_OFFSET_ENABLED = _env_bool("QARM_STACK_PICK_X_OFFSET_ENABLED", True)
STACK_PICK_X_OFFSET_REQUIRE_PICK_X = _env_bool("QARM_STACK_PICK_X_OFFSET_REQUIRE_PICK_X", True)
STACK_PICK_X_NEAR_M = float(os.getenv("QARM_STACK_PICK_X_NEAR_M", "0.320"))
STACK_PICK_X_FAR_M = float(os.getenv("QARM_STACK_PICK_X_FAR_M", "0.526"))
STACK_PICK_X_OFFSET_NEAR_M = float(os.getenv("QARM_STACK_PICK_X_OFFSET_NEAR_M", "-0.006"))
STACK_PICK_X_OFFSET_FAR_M = float(os.getenv("QARM_STACK_PICK_X_OFFSET_FAR_M", "0.01"))
STACK_PICK_X_OFFSET_CLAMP_MIN_M = float(os.getenv("QARM_STACK_PICK_X_OFFSET_CLAMP_MIN_M", "-0.005"))
STACK_PICK_X_OFFSET_CLAMP_MAX_M = float(os.getenv("QARM_STACK_PICK_X_OFFSET_CLAMP_MAX_M", "0.01"))
# Release XY opposite of grasp pick bias so cube center aligns with stack anchor (Z unchanged).
PLACE_PICK_BIAS_COMPENSATE_ENABLED = _env_bool("QARM_PLACE_PICK_BIAS_COMPENSATE_ENABLED", True)
PLACE_PICK_BIAS_COMPENSATE_SCALE = float(os.getenv("QARM_PLACE_PICK_BIAS_COMPENSATE_SCALE", "1"))
PLACE_PICK_BIAS_COMPENSATE_STACK_ANCHOR_ENABLED = _env_bool(
    "QARM_PLACE_PICK_BIAS_COMPENSATE_STACK_ANCHOR_ENABLED",
    False,
)
# Disabled by default: +X per stack level from measured Z (re-enable via env after place/scan tune).
GRASP_STACK_FORWARD_ENABLE = _env_bool("QARM_GRASP_STACK_FORWARD_ENABLE", True)
# Measured Z above this (table/loose picks stay ~0.05-0.07 m) starts stack-level X bias.
GRASP_STACK_FORWARD_Z_START_M = float(os.getenv("QARM_GRASP_STACK_FORWARD_Z_START_M", "0.055"))
# +X per stack level (STACK_LEVEL_DZ_M step); compensates perspective on elevated cube faces.
GRASP_STACK_FORWARD_PER_LEVEL_M = float(os.getenv("QARM_GRASP_STACK_FORWARD_PER_LEVEL_M", "0.006"))
GRASP_STACK_FORWARD_MAX_LEVELS = int(os.getenv("QARM_GRASP_STACK_FORWARD_MAX_LEVELS", "3"))
GRASP_STACK_FORWARD_MAX_M = float(os.getenv("QARM_GRASP_STACK_FORWARD_MAX_M", "0.016"))
# Legacy continuous gain (used only when per_level_m <= 0).
GRASP_STACK_FORWARD_GAIN_X_PER_M = float(os.getenv("QARM_GRASP_STACK_FORWARD_GAIN_X_PER_M", "0.140"))
PUSH_TARGET_X_M = float(os.getenv("QARM_PUSH_TARGET_X_M", "0.400"))
PUSH_TARGET_Y_M = float(os.getenv("QARM_PUSH_TARGET_Y_M", "0.000"))
PUSH_TARGET_Z_M = float(os.getenv("QARM_PUSH_TARGET_Z_M", "0.050"))
PUSH_GRIP_RATIO = float(os.getenv("QARM_PUSH_GRIP_RATIO", "0.80"))
PUSH_APPROACH_OFFSET_M = float(os.getenv("QARM_PUSH_APPROACH_OFFSET_M", "0.030"))
PUSH_STEP_M = float(os.getenv("QARM_PUSH_STEP_M", "0.025"))
PUSH_MAX_STEPS = int(os.getenv("QARM_PUSH_MAX_STEPS", "6"))
PUSH_FINAL_XY_TOL_M = float(os.getenv("QARM_PUSH_FINAL_XY_TOL_M", "0.050"))
PUSH_Z_CLEARANCE_M = float(os.getenv("QARM_PUSH_Z_CLEARANCE_M", "0.012"))
PUSH_CORRIDOR_HALF_WIDTH_M = float(os.getenv("QARM_PUSH_CORRIDOR_HALF_WIDTH_M", "0.045"))
PUSH_MIN_PROGRESS_M = float(os.getenv("QARM_PUSH_MIN_PROGRESS_M", "0.010"))

# ============================= Tuning: Planner Context + Color Classifier =============================
COLOR_MIN_SAT = float(os.getenv("QARM_COLOR_MIN_SAT", "45.0"))
COLOR_MIN_VAL = float(os.getenv("QARM_COLOR_MIN_VAL", "38.0"))
COLOR_MIN_VALID_PIXELS = int(os.getenv("QARM_COLOR_MIN_VALID_PIXELS", "70"))
COLOR_ORANGE_H_MIN = float(os.getenv("QARM_COLOR_ORANGE_H_MIN", "4.0"))
COLOR_ORANGE_H_MAX = float(os.getenv("QARM_COLOR_ORANGE_H_MAX", "30.0"))
COLOR_BLUE_H_MIN = float(os.getenv("QARM_COLOR_BLUE_H_MIN", "78.0"))
COLOR_BLUE_H_MAX = float(os.getenv("QARM_COLOR_BLUE_H_MAX", "122.0"))

# ============================= Tuning: Prompted Verify Hooks =============================
PROMPTED_COLUMN_ONLY = _env_bool("QARM_PROMPTED_COLUMN_ONLY", False)
PICK_STABILITY_RECHECK_ENABLED = _env_bool("QARM_PICK_STABILITY_RECHECK", True)
PICK_STABILITY_RECHECK_DELAY_S = float(os.getenv("QARM_PICK_STABILITY_RECHECK_DELAY_S", "0.18"))
PICK_STABILITY_RECHECK_SAMPLES = int(os.getenv("QARM_PICK_STABILITY_RECHECK_SAMPLES", "8"))
PICK_STABILITY_RECHECK_MIN_HITS = int(os.getenv("QARM_PICK_STABILITY_RECHECK_MIN_HITS", "4"))
PICK_STABILITY_RECHECK_MIN_CURRENT_A = float(
    os.getenv("QARM_PICK_STABILITY_RECHECK_MIN_A", f"{POST_LIFT_VERIFY_MIN_CURRENT_A:.4f}")
)
PLACE_VERIFY_MIN_CONF = float(os.getenv("QARM_PLACE_VERIFY_MIN_CONF", "0.50"))
PLACE_VERIFY_RADIUS_M = float(os.getenv("QARM_PLACE_VERIFY_RADIUS_M", "0.050"))
PX_TOL = int(os.getenv("QARM_CENTER_PX_TOL", "6"))  # centered when |ex|,|ey| <= PX_TOL pixels
SCENE_RECON_SAMPLES = int(os.getenv("QARM_SCENE_RECON_SAMPLES", "2"))
SCENE_RECON_MIN_CONF = float(os.getenv("QARM_SCENE_RECON_MIN_CONF", f"{PLACE_VERIFY_MIN_CONF:.3f}"))
SCENE_RECON_DEDUP_XY_M = float(os.getenv("QARM_SCENE_RECON_DEDUP_XY_M", "0.035"))
SCENE_RECON_DEDUP_Z_M = float(os.getenv("QARM_SCENE_RECON_DEDUP_Z_M", "0.040"))
SCENE_RECON_OCCUPANCY_XY_M = float(os.getenv("QARM_SCENE_RECON_OCCUPANCY_XY_M", "0.045"))
SCENE_RECON_OCCUPANCY_Z_M = float(os.getenv("QARM_SCENE_RECON_OCCUPANCY_Z_M", "0.045"))
SCENE_RECON_PAUSE_S = float(os.getenv("QARM_SCENE_RECON_PAUSE_S", "0.05"))
SCENE_RECON_SECTION_MAX_DIST_M = float(os.getenv("QARM_SCENE_RECON_SECTION_MAX_DIST_M", "0.075"))
SCENE_RECON_PLACE_Y_MARGIN_M = float(os.getenv("QARM_SCENE_RECON_PLACE_Y_MARGIN_M", "0.030"))
MISPLACED_PICK_SECTION_BAND_MARGIN_M = float(
    os.getenv("QARM_MISPLACED_PICK_SECTION_BAND_MARGIN_M", "0.080")
)
_MISPLACED_PICK_SECTION_MAX_DIST_ENV = os.getenv("QARM_MISPLACED_PICK_SECTION_MAX_DIST_M")
MISPLACED_PICK_SECTION_MAX_DIST_M = float(
    _MISPLACED_PICK_SECTION_MAX_DIST_ENV
    if _MISPLACED_PICK_SECTION_MAX_DIST_ENV is not None
    else f"{max(0.110, float(SCENE_RECON_SECTION_MAX_DIST_M)):.3f}"
)
STARTUP_STACK_IDENTITY_SAMPLES = int(os.getenv("QARM_STARTUP_STACK_IDENTITY_SAMPLES", "2"))
STARTUP_STACK_LOCK_TIMEOUT_S = float(os.getenv("QARM_STARTUP_STACK_LOCK_TIMEOUT_S", "3.0"))
STARTUP_STACK_LOCK_FRAMES = int(os.getenv("QARM_STARTUP_STACK_LOCK_FRAMES", "3"))
STARTUP_STACK_MEASURE_SAMPLES = int(os.getenv("QARM_STARTUP_STACK_MEASURE_SAMPLES", "5"))
STARTUP_STACK_MEASURE_MIN_HITS = int(os.getenv("QARM_STARTUP_STACK_MEASURE_MIN_HITS", "3"))
STARTUP_STACK_MEASURE_TIMEOUT_S = float(os.getenv("QARM_STARTUP_STACK_MEASURE_TIMEOUT_S", "1.4"))
STARTUP_STACK_DISCOVERY_TIMEOUT_S = float(os.getenv("QARM_STARTUP_STACK_DISCOVERY_TIMEOUT_S", "9.0"))
STARTUP_STACK_DISCOVERY_MIN_SCAN_S = float(os.getenv("QARM_STARTUP_STACK_DISCOVERY_MIN_SCAN_S", "1.2"))
STARTUP_STACK_DISCOVERY_STABLE_S = float(os.getenv("QARM_STARTUP_STACK_DISCOVERY_STABLE_S", "0.7"))
STARTUP_STACK_DISCOVERY_NO_DET_TIMEOUT_S = float(
    os.getenv("QARM_STARTUP_STACK_DISCOVERY_NO_DET_TIMEOUT_S", "3.0")
)
STARTUP_STACK_USE_PLACE_BAND = _env_bool("QARM_STARTUP_STACK_USE_PLACE_BAND", False)
STARTUP_STACK_LOCK_STAGE_TIMEOUT_S = float(
    os.getenv("QARM_STARTUP_STACK_LOCK_STAGE_TIMEOUT_S", "14.0")
)
STARTUP_STACK_CENTER_EY_SCALE = float(
    np.clip(
        float(os.getenv("QARM_STARTUP_STACK_CENTER_EY_SCALE", "0.72")),
        0.20,
        1.20,
    )
)
STARTUP_STACK_MAX_TRACK_ATTEMPTS = int(
    os.getenv("QARM_STARTUP_STACK_MAX_TRACK_ATTEMPTS", "2")
)
STARTUP_STACK_VISIBILITY_DEFER_CHECKS = int(
    os.getenv("QARM_STARTUP_STACK_VISIBILITY_DEFER_CHECKS", "8")
)
STARTUP_TARGET_MIN_CONF = float(os.getenv("QARM_STARTUP_TARGET_MIN_CONF", "0.70"))
STARTUP_STACK_Z_PREDICT_ENABLED = _env_bool("QARM_STARTUP_STACK_Z_PREDICT_ENABLED", True)
STARTUP_REFRESH_PASS_ENABLED = _env_bool("QARM_STARTUP_REFRESH_PASS_ENABLED", True)
STARTUP_STACK_LAYER_MATCH_XY_M = float(os.getenv("QARM_STARTUP_STACK_LAYER_MATCH_XY_M", "0.030"))
STARTUP_STACK_LAYER_MATCH_Z_M = float(os.getenv("QARM_STARTUP_STACK_LAYER_MATCH_Z_M", "0.025"))
STARTUP_STACK_LAYER_SCAN_FRAMES = int(os.getenv("QARM_STARTUP_STACK_LAYER_SCAN_FRAMES", "6"))
STARTUP_STACK_LAYER_VOTE_MIN_HITS = int(os.getenv("QARM_STARTUP_STACK_LAYER_VOTE_MIN_HITS", "1"))
STARTUP_STACK_REQUIRE_EXPECTED_LAYERS = _env_bool("QARM_STARTUP_STACK_REQUIRE_EXPECTED_LAYERS", True)
STARTUP_STACK_BOOTSTRAP_MAX_PASSES = max(1, int(os.getenv("QARM_STARTUP_STACK_BOOTSTRAP_MAX_PASSES", "3")))
STARTUP_STACK_SIDE_FULL_RESCAN_ENABLED = _env_bool("QARM_STARTUP_STACK_SIDE_FULL_RESCAN_ENABLED", False)
STARTUP_STACK_SIDE_FULL_RESCAN_MIN_EXPECTED = int(
    os.getenv("QARM_STARTUP_STACK_SIDE_FULL_RESCAN_MIN_EXPECTED", "3")
)
STARTUP_STACK_SIDE_FULL_RESCAN_FRAMES = int(os.getenv("QARM_STARTUP_STACK_SIDE_FULL_RESCAN_FRAMES", "12"))
STARTUP_STACK_LOCK_TOP_FIRST = _env_bool("QARM_STARTUP_STACK_LOCK_TOP_FIRST", False) # turned to off,less likely to overlap if bottom cube centered.
STARTUP_STACK_MIN_GROUP_OBS = int(os.getenv("QARM_STARTUP_STACK_MIN_GROUP_OBS", "5"))
STARTUP_STACK_ASSIGN_DEBUG = _env_bool("QARM_STARTUP_STACK_ASSIGN_DEBUG", True)
STARTUP_STACK_COLOR_MIN_CONF = float(
    os.getenv(
        "QARM_STARTUP_STACK_COLOR_MIN_CONF",
        "0.70",
    )
)
STARTUP_STACK_MAX_CUBES_PER_SIDE = int(
    os.getenv(
        "QARM_STARTUP_STACK_MAX_CUBES_PER_SIDE",
        os.getenv("QARM_MAX_STACK_LEVELS_PER_SECTION", "3"),
    )
)
# Allow startup hydrate to track/attempt more IDs than the final per-side stack cap.
STARTUP_STACK_MAX_TRACK_TARGETS = int(
    os.getenv(
        "QARM_STARTUP_STACK_MAX_TRACK_TARGETS",
        "12",
    )
)
STARTUP_STACK_EXIT_WHEN_SIDES_FULL = _env_bool("QARM_STARTUP_STACK_EXIT_WHEN_SIDES_FULL", True)
SCENE_RECON_PREGRASP_REJECT_CAP = int(os.getenv("QARM_SCENE_RECON_PREGRASP_REJECT_CAP", "4"))
SCENE_RECON_PREGRASP_TIMEOUT_S = float(os.getenv("QARM_SCENE_RECON_PREGRASP_TIMEOUT_S", "6.0"))
PLACE_SPACE_TRUTH_FRESH_UV_PX = float(os.getenv("QARM_PLACE_SPACE_TRUTH_FRESH_UV_PX", "8.0"))
SCENE_RECON_PREGRASP_REQUIRED_HITS = int(
    os.getenv(
        "QARM_SCENE_RECON_PREGRASP_REQUIRED_HITS",
        os.getenv("QARM_PLACE_VERIFY_V2_MIN_HITS", "5"),
    )
)
_PLACE_VERIFY_V2_XY_MARGIN_DEFAULT_M = float(os.getenv("QARM_PLACE_VERIFY_V2_XY_MARGIN_M", "0.052"))
_PLACE_VERIFY_V2_Z_MARGIN_DEFAULT_M = float(os.getenv("QARM_PLACE_VERIFY_V2_Z_MARGIN_M", "0.015"))
_STARTUP_STACK_SECTION_XY_MARGIN_DEFAULT_M = max(
    float(SCENE_RECON_SECTION_MAX_DIST_M),
    float(_PLACE_VERIFY_V2_XY_MARGIN_DEFAULT_M) * 1.75,
    0.140,
)
STARTUP_STACK_SECTION_XY_MARGIN_M = float(
    os.getenv("QARM_STARTUP_STACK_SECTION_XY_MARGIN_M", f"{_STARTUP_STACK_SECTION_XY_MARGIN_DEFAULT_M:.3f}")
)
STARTUP_STACK_LOCK_ASSIGN_XY_MARGIN_M = float(
    os.getenv(
        "QARM_STARTUP_STACK_LOCK_ASSIGN_XY_MARGIN_M",
        f"{max(float(STARTUP_STACK_SECTION_XY_MARGIN_M), 0.180):.3f}",
    )
)
if _MISPLACED_PICK_SECTION_MAX_DIST_ENV is None:
    MISPLACED_PICK_SECTION_MAX_DIST_M = float(STARTUP_STACK_LOCK_ASSIGN_XY_MARGIN_M)
_SCENE_RECON_PREGRASP_SEED_UV_DEFAULT_PX = max(6.0, float(PX_TOL) + 2.0)
_SCENE_RECON_PREGRASP_SEED_XY_DEFAULT_M = max(0.015, _PLACE_VERIFY_V2_XY_MARGIN_DEFAULT_M * 0.75)
_SCENE_RECON_PREGRASP_SEED_Z_DEFAULT_M = max(
    0.015,
    min(_PLACE_VERIFY_V2_Z_MARGIN_DEFAULT_M * 0.60, float(STACK_LEVEL_DZ_M) * 0.55),
)
SCENE_RECON_PREGRASP_SEED_UV_MARGIN_PX = float(
    os.getenv("QARM_SCENE_RECON_PREGRASP_SEED_UV_MARGIN_PX", f"{_SCENE_RECON_PREGRASP_SEED_UV_DEFAULT_PX:.1f}")
)
SCENE_RECON_PREGRASP_SEED_XY_MARGIN_M = float(
    os.getenv("QARM_SCENE_RECON_PREGRASP_SEED_XY_MARGIN_M", f"{_SCENE_RECON_PREGRASP_SEED_XY_DEFAULT_M:.3f}")
)
SCENE_RECON_PREGRASP_SEED_Z_MARGIN_M = float(
    os.getenv("QARM_SCENE_RECON_PREGRASP_SEED_Z_MARGIN_M", f"{_SCENE_RECON_PREGRASP_SEED_Z_DEFAULT_M:.3f}")
)
PREGRASP_PLACE_SPACE_VALIDATE_ENABLED = _env_bool("QARM_PREGRASP_PLACE_SPACE_VALIDATE_ENABLED", False)
PREGRASP_RESTORE_JOINT_TOL_DEG = float(os.getenv("QARM_PREGRASP_RESTORE_JOINT_TOL_DEG", "1.2"))
PREGRASP_RESTORE_DURATION_S = float(os.getenv("QARM_PREGRASP_RESTORE_DURATION_S", "1.0"))
PREGRASP_RESTORE_STEPS = int(os.getenv("QARM_PREGRASP_RESTORE_STEPS", "70"))

# ============================= Tuning: Place Verification V2 (Identity-Aware, Geometry) =============================
PLACE_VERIFY_V2_ENABLED = _env_bool("QARM_PLACE_VERIFY_V2_ENABLED", True)
PLACE_VERIFY_V2_SAMPLES_PRE = int(os.getenv("QARM_PLACE_VERIFY_V2_SAMPLES_PRE", "2"))
PLACE_VERIFY_V2_SAMPLES_POST = int(os.getenv("QARM_PLACE_VERIFY_V2_SAMPLES_POST", "8"))
PLACE_VERIFY_V2_RADIUS_M = float(os.getenv("QARM_PLACE_VERIFY_V2_RADIUS_M", f"{PLACE_VERIFY_RADIUS_M:.3f}"))
PLACE_VERIFY_V2_XY_MARGIN_M = _PLACE_VERIFY_V2_XY_MARGIN_DEFAULT_M
PLACE_VERIFY_V2_Z_MARGIN_M = _PLACE_VERIFY_V2_Z_MARGIN_DEFAULT_M
PLACE_VERIFY_V2_EXPECTED_X_OFFSET_M = float(os.getenv("QARM_PLACE_VERIFY_V2_EXPECTED_X_OFFSET_M", "0.007"))
PLACE_VERIFY_V2_EXPECTED_Y_OFFSET_M = float(os.getenv("QARM_PLACE_VERIFY_V2_EXPECTED_Y_OFFSET_M", "-0.021"))
PLACE_VERIFY_V2_EXPECTED_Z_OFFSET_M = float(os.getenv("QARM_PLACE_VERIFY_V2_EXPECTED_Z_OFFSET_M", "0.010"))
PLACE_VERIFY_V2_EXPECTED_EVAL_USE_OFFSETS = _env_bool(
    "QARM_PLACE_VERIFY_V2_EXPECTED_EVAL_USE_OFFSETS",
    False,
)
PLACE_VERIFY_V2_SURFACE_Z_OFFSET_M = float(os.getenv("QARM_PLACE_VERIFY_V2_SURFACE_Z_OFFSET_M", "0.030"))
# Projected cube XYZ bias correction (all scans). Lifts/lowers raw depth projection toward command frame.
# Default Y is +0.021 m (inverse of verify eval Y on expected). Z is never modified here.
SCAN_BASE_XY_OFFSET_ENABLED = _env_bool("QARM_SCAN_BASE_XY_OFFSET_ENABLED", True)
SCAN_BASE_X_OFFSET_M = float(
    os.getenv("QARM_SCAN_BASE_X_OFFSET_M", str(PLACE_VERIFY_V2_EXPECTED_X_OFFSET_M))
)
SCAN_BASE_Y_OFFSET_M = float(
    os.getenv(
        "QARM_SCAN_BASE_Y_OFFSET_M",
        str(-float(PLACE_VERIFY_V2_EXPECTED_Y_OFFSET_M)),
    )
)
# Tighten Z validation for higher stacks (e.g., 3rd cube) to avoid false confirms
# when a top cube drops and aligns near the 2nd-cube height.
PLACE_VERIFY_V2_TIGHT_Z_STACK_LEVEL_MIN = int(os.getenv("QARM_PLACE_VERIFY_V2_TIGHT_Z_STACK_LEVEL_MIN", "3"))
PLACE_VERIFY_V2_TIGHT_Z_MARGIN_M = float(os.getenv("QARM_PLACE_VERIFY_V2_TIGHT_Z_MARGIN_M", "0.012"))
PLACE_VERIFY_V2_STACK_XY_MARGIN_M = float(os.getenv("QARM_PLACE_VERIFY_V2_STACK_XY_MARGIN_M", "0.062"))
# Minimal verify-v2 mismatch relaxation to reduce false out-of-margin rejects.
# Applied only at verify scoring time (does not alter tracker/lock behavior).
PLACE_VERIFY_V2_MISMATCH_RELAX_XY_M = float(os.getenv("QARM_PLACE_VERIFY_V2_MISMATCH_RELAX_XY_M", "0.016"))
PLACE_VERIFY_V2_MISMATCH_RELAX_Z_M = float(os.getenv("QARM_PLACE_VERIFY_V2_MISMATCH_RELAX_Z_M", "0.008"))
PLACE_VERIFY_V2_MIN_HITS = int(os.getenv("QARM_PLACE_VERIFY_V2_MIN_HITS", "4"))
PLACE_VERIFY_V2_DELTA_MIN = float(os.getenv("QARM_PLACE_VERIFY_V2_DELTA_MIN", "0.10"))
PLACE_VERIFY_V2_SETTLE_S = float(os.getenv("QARM_PLACE_VERIFY_V2_SETTLE_S", "0.20"))
PLACE_VERIFY_V2_SLOT_SCAN_FIRST = _env_bool("QARM_PLACE_VERIFY_V2_SLOT_SCAN_FIRST", True)
PLACE_VERIFY_V2_HYDRATE_FALLBACK_ENABLED = _env_bool("QARM_PLACE_VERIFY_V2_HYDRATE_FALLBACK_ENABLED", True)
PLACE_VERIFY_V2_EXPECTED_SLOT_RETRIES = int(os.getenv("QARM_PLACE_VERIFY_V2_EXPECTED_SLOT_RETRIES", "1"))
PLACE_VERIFY_V2_TOP_CANDIDATE_CHECKS = int(os.getenv("QARM_PLACE_VERIFY_V2_TOP_CANDIDATE_CHECKS", "2"))
PLACE_VERIFY_V2_DEFER_GENERIC_HANDOFF_TO_HYDRATE = _env_bool(
    "QARM_PLACE_VERIFY_V2_DEFER_GENERIC_HANDOFF_TO_HYDRATE",
    True,
)
PLACE_VERIFY_V2_LADDER_LOGS = _env_bool("QARM_PLACE_VERIFY_V2_LADDER_LOGS", True)
PLACE_VERIFY_V2_ACTIVE_CENTER_ON_WEAK = _env_bool("QARM_PLACE_VERIFY_V2_ACTIVE_CENTER_ON_WEAK", True)
PLACE_VERIFY_V2_ACTIVE_CENTER_TIMEOUT_S = float(os.getenv("QARM_PLACE_VERIFY_V2_ACTIVE_CENTER_TIMEOUT_S", "2.5"))
PLACE_VERIFY_V2_ALWAYS_RECENTER = _env_bool("QARM_PLACE_VERIFY_V2_ALWAYS_RECENTER", True)
PLACE_VERIFY_V2_CUBE_EDGE_M = float(os.getenv("QARM_PLACE_VERIFY_V2_CUBE_EDGE_M", "0.060"))
PLACE_VERIFY_V2_MIN_OVERLAP = float(os.getenv("QARM_PLACE_VERIFY_V2_MIN_OVERLAP", "0.65"))
PLACE_VERIFY_V2_LOOK_MOVE_S = float(os.getenv("QARM_PLACE_VERIFY_V2_LOOK_MOVE_S", "0.55"))
PLACE_VERIFY_V2_STACK_MIN_LAYER_FRAC = float(os.getenv("QARM_PLACE_VERIFY_V2_STACK_MIN_LAYER_FRAC", "0.50"))
PLACE_VERIFY_V2_STACK_PREFER_TOP = _env_bool("QARM_PLACE_VERIFY_V2_STACK_PREFER_TOP", True)
PLACE_VERIFY_V2_RECENTER_PIXEL_ONLY = _env_bool("QARM_PLACE_VERIFY_V2_RECENTER_PIXEL_ONLY", True)
PLACE_VERIFY_V2_RECENTER_PIXEL_TOP = _env_bool("QARM_PLACE_VERIFY_V2_RECENTER_PIXEL_TOP", True)
PLACE_VERIFY_V2_RECENTER_COLOR_FILTER = _env_bool("QARM_PLACE_VERIFY_V2_RECENTER_COLOR_FILTER", True)
PLACE_VERIFY_V2_RECENTER_COLOR_MIN_CONF = float(os.getenv("QARM_PLACE_VERIFY_V2_RECENTER_COLOR_MIN_CONF", "0.52"))
PLACE_VERIFY_V2_COLOR_SAMPLES = int(os.getenv("QARM_PLACE_VERIFY_V2_COLOR_SAMPLES", "4"))
PLACE_VERIFY_V2_COLOR_MIN_HITS = int(os.getenv("QARM_PLACE_VERIFY_V2_COLOR_MIN_HITS", "2"))
PLACE_VERIFY_V2_COLOR_MIN_CONF = float(
    os.getenv(
        "QARM_PLACE_VERIFY_V2_COLOR_MIN_CONF",
        os.getenv("QARM_PLACE_VERIFY_V2_RECENTER_COLOR_MIN_CONF", "0.52"),
    )
)
PLACE_VERIFY_V2_COLOR_COMMIT_CONF = float(os.getenv("QARM_PLACE_VERIFY_V2_COLOR_COMMIT_CONF", "0.56"))
PLACE_VERIFY_V2_RECENTER_DISALLOW_WRONG_COLOR = _env_bool("QARM_PLACE_VERIFY_V2_RECENTER_DISALLOW_WRONG_COLOR", True)
PLACE_VERIFY_V2_RECENTER_WRONG_COLOR_BLACKLIST_PX = float(os.getenv("QARM_PLACE_VERIFY_V2_RECENTER_WRONG_COLOR_BLACKLIST_PX", "80.0"))
PLACE_VERIFY_V2_RECENTER_DISALLOW_WRONG_XY = _env_bool("QARM_PLACE_VERIFY_V2_RECENTER_DISALLOW_WRONG_XY", True)
_PLACE_VERIFY_V2_WRONG_XY_DEFAULT_M = max(float(PLACE_VERIFY_V2_XY_MARGIN_M), float(PLACE_VERIFY_V2_RADIUS_M))
PLACE_VERIFY_V2_RECENTER_WRONG_XY_M = float(
    os.getenv("QARM_PLACE_VERIFY_V2_RECENTER_WRONG_XY_M", f"{_PLACE_VERIFY_V2_WRONG_XY_DEFAULT_M:.3f}")
)
PLACE_VERIFY_V2_RECENTER_WRONG_XY_BLACKLIST_PX = float(
    os.getenv(
        "QARM_PLACE_VERIFY_V2_RECENTER_WRONG_XY_BLACKLIST_PX",
        f"{PLACE_VERIFY_V2_RECENTER_WRONG_COLOR_BLACKLIST_PX:.1f}",
    )
)
PLACE_VERIFY_V2_RECENTER_TRACK_SMOOTH_FRAMES = int(
    os.getenv("QARM_PLACE_VERIFY_V2_RECENTER_TRACK_SMOOTH_FRAMES", "2")
)
PLACE_VERIFY_V2_RECENTER_MAX_CANDIDATE_TRIES = int(
    os.getenv("QARM_PLACE_VERIFY_V2_RECENTER_MAX_CANDIDATE_TRIES", "2")
)
PLACE_VERIFY_V2_RECENTER_DYNAMIC_BLACKLIST_PX = float(
    os.getenv("QARM_PLACE_VERIFY_V2_RECENTER_DYNAMIC_BLACKLIST_PX", "28.0")
)
PLACE_VERIFY_V2_RECENTER_PREBLACKLIST_WRONG_COLOR_TOP_FIRST = _env_bool(
    "QARM_PLACE_VERIFY_V2_RECENTER_PREBLACKLIST_WRONG_COLOR_TOP_FIRST",
    False,
)
PLACE_VERIFY_V2_RECENTER_SHOW_BLACKLIST_OVERLAY = _env_bool(
    "QARM_PLACE_VERIFY_V2_RECENTER_SHOW_BLACKLIST_OVERLAY",
    True,
)
PLACE_VERIFY_V2_RECENTER_PERSIST_WINDOW = _env_bool(
    "QARM_PLACE_VERIFY_V2_RECENTER_PERSIST_WINDOW",
    True,
)
PLACE_VERIFY_V2_RECENTER_LOCK_PAUSE_MS = int(
    os.getenv("QARM_PLACE_VERIFY_V2_RECENTER_LOCK_PAUSE_MS", "1")
)
PLACE_VERIFY_V2_RECENTER_FORCED_TARGET_FRAMES = int(
    os.getenv("QARM_PLACE_VERIFY_V2_RECENTER_FORCED_TARGET_FRAMES", "6")
)
PLACE_VERIFY_V2_WEAK_RECENTER_MAX_PASSES = int(
    os.getenv("QARM_PLACE_VERIFY_V2_WEAK_RECENTER_MAX_PASSES", "1")
)
PLACE_VERIFY_V2_RECENTER_ON_MISMATCH = _env_bool(
    "QARM_PLACE_VERIFY_V2_RECENTER_ON_MISMATCH",
    True,
)
PLACE_VERIFY_V2_MISMATCH_RECENTER_TIMEOUT_S = float(
    os.getenv("QARM_PLACE_VERIFY_V2_MISMATCH_RECENTER_TIMEOUT_S", "1.3")
)
PLACE_VERIFY_V2_MAX_REJECTS = int(
    os.getenv("QARM_PLACE_VERIFY_V2_MAX_REJECTS", "4")
)
PLACE_VERIFY_V2_MIN_REJECTS_PER_SESSION = int(
    os.getenv("QARM_PLACE_VERIFY_V2_MIN_REJECTS_PER_SESSION", "8")
)
PLACE_VERIFY_V2_HARD_TIMEOUT_S = float(
    os.getenv("QARM_PLACE_VERIFY_V2_HARD_TIMEOUT_S", "30.0")
)
TRACK_HANDOFF_NO_CANDIDATE_TIMEOUT_S = float(
    os.getenv("QARM_TRACK_HANDOFF_NO_CANDIDATE_TIMEOUT_S", "5.0")
)
PLACE_VERIFY_V2_DISABLE_STABLE_GATE = _env_bool(
    "QARM_PLACE_VERIFY_V2_DISABLE_STABLE_GATE",
    True,
)
PLACE_VERIFY_V2_TRACK_STABLE_FRAMES = int(
    os.getenv("QARM_PLACE_VERIFY_V2_TRACK_STABLE_FRAMES", "3")
)
PLACE_VERIFY_V2_TRACK_MAX_JUMP_PX = float(
    os.getenv("QARM_PLACE_VERIFY_V2_TRACK_MAX_JUMP_PX", "24.0")
)
PLACE_VERIFY_V2_TRACK_SHIFT_PAUSE_S = float(
    os.getenv("QARM_PLACE_VERIFY_V2_TRACK_SHIFT_PAUSE_S", "0.08")
)
PLACE_VERIFY_V2_SECTION_PIXEL_GATE = _env_bool(
    "QARM_PLACE_VERIFY_V2_SECTION_PIXEL_GATE",
    True,
)
PLACE_VERIFY_V2_SECTION_PIXEL_MARGIN_PX = int(
    os.getenv("QARM_PLACE_VERIFY_V2_SECTION_PIXEL_MARGIN_PX", "18")
)
PLACE_VERIFY_V2_AVOID_NEGATIVE_Y = _env_bool(
    "QARM_PLACE_VERIFY_V2_AVOID_NEGATIVE_Y",
    True,
)
PLACE_VERIFY_V2_MIN_TRACK_Y_M = float(os.getenv("QARM_PLACE_VERIFY_V2_MIN_TRACK_Y_M", "-0.020"))
_PLACE_VERIFY_V2_TARGET_MODE_RAW = os.getenv("QARM_PLACE_VERIFY_V2_TARGET_MODE", "top_first").strip().lower()
PLACE_VERIFY_V2_TARGET_MODE = (
    _PLACE_VERIFY_V2_TARGET_MODE_RAW
    if _PLACE_VERIFY_V2_TARGET_MODE_RAW in {"top_first", "filtered_first"}
    else "top_first"
)

# ============================= Tuning: Tracking =============================
TRACK_ENABLE = _env_bool("QARM_TRACK_ENABLE", True)
TRACK_MATCH_XY_M = float(os.getenv("QARM_TRACK_MATCH_XY_M", "0.055"))
TRACK_MAX_MISS_FRAMES = int(os.getenv("QARM_TRACK_MAX_MISS_FRAMES", "8"))
TRACK_MIN_CONF = float(os.getenv("QARM_TRACK_MIN_CONF", "0.35"))
TRACK_PICK_PREFER_TOP = _env_bool("QARM_TRACK_PICK_PREFER_TOP", True)
TRACK_PICK_TOP_STRICT = _env_bool("QARM_TRACK_PICK_TOP_STRICT", True)
TRACK_PICK_TOP_TIE_Z_M = float(os.getenv("QARM_TRACK_PICK_TOP_TIE_Z_M", "0.012"))
PICK_TOP_EXPOSED_ONLY = _env_bool("QARM_PICK_TOP_EXPOSED_ONLY", True)
PICK_TOP_EXPOSED_X_OVERLAP_MIN = float(os.getenv("QARM_PICK_TOP_EXPOSED_X_OVERLAP_MIN", "0.55"))
PICK_TOP_EXPOSED_Y_GAP_PX = float(os.getenv("QARM_PICK_TOP_EXPOSED_Y_GAP_PX", "8.0"))
_PICK_TOP_EXPOSED_FALLBACK_RAW = os.getenv("QARM_PICK_TOP_EXPOSED_FALLBACK", "closest").strip().lower()
PICK_TOP_EXPOSED_FALLBACK = (
    _PICK_TOP_EXPOSED_FALLBACK_RAW
    if _PICK_TOP_EXPOSED_FALLBACK_RAW in {"closest", "top"}
    else "closest"
)
PICK_OTHER_BLOCK_XY_M = float(os.getenv("QARM_PICK_OTHER_BLOCK_XY_M", "0.055"))
PICK_OTHER_BLOCK_Z_M = float(os.getenv("QARM_PICK_OTHER_BLOCK_Z_M", "0.080"))
PICK_OTHER_BLOCK_UV_PX = float(os.getenv("QARM_PICK_OTHER_BLOCK_UV_PX", "85.0"))
PICK_OTHER_MAX_CANDIDATE_TRIES = int(os.getenv("QARM_PICK_OTHER_MAX_CANDIDATE_TRIES", "3"))
PICK_OTHER_VALIDATE_SAMPLES = int(os.getenv("QARM_PICK_OTHER_VALIDATE_SAMPLES", "4"))
PICK_OTHER_VALIDATE_TIMEOUT_S = float(os.getenv("QARM_PICK_OTHER_VALIDATE_TIMEOUT_S", "1.2"))
PICK_OTHER_MAX_REJECTS = int(os.getenv("QARM_PICK_OTHER_MAX_REJECTS", "4"))
PICK_OTHER_HARD_TIMEOUT_S = float(os.getenv("QARM_PICK_OTHER_HARD_TIMEOUT_S", "25.0"))
PICK_OTHER_PERSIST_BLOCK_MAX = max(1, int(os.getenv("QARM_PICK_OTHER_PERSIST_BLOCK_MAX", "8")))
PICK_OTHER_REJECT_SAME_COLOR = str(os.getenv("QARM_PICK_OTHER_REJECT_SAME_COLOR", "true")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# Hard-lock tracker backend to Bot-SORT + ReID.
# QARM_YOLO_TRACKER is intentionally ignored/deprecated in this mode.
YOLO_TRACKER = "trackers/botsort_reid.yaml"
YOLO_TRACK_PERSIST = _env_bool("QARM_YOLO_TRACK_PERSIST", True)
TRACK_WARN_MISSING_IDS = _env_bool("QARM_TRACK_WARN_MISSING_IDS", True)
TRACK_WARN_MISSING_IDS_INTERVAL_S = float(os.getenv("QARM_TRACK_WARN_MISSING_IDS_INTERVAL_S", "2.0"))

# ============================= Tuning: Stack Readiness Gate =============================
STACK_READINESS_MIN_PLACE_SUCCESS = float(os.getenv("QARM_STACK_READY_MIN_PLACE_SUCCESS", "0.85"))
STACK_READINESS_MAX_REOBSERVE_RATIO = float(os.getenv("QARM_STACK_READY_MAX_REOBSERVE_RATIO", "0.40"))
STACK_READINESS_MAX_RECOVERY_RATIO = float(os.getenv("QARM_STACK_READY_MAX_RECOVERY_RATIO", "0.25"))
STACK_READINESS_MAX_POLICY_INVALID = int(os.getenv("QARM_STACK_READY_MAX_POLICY_INVALID", "0"))

# ============================= Tuning: Verification Correction =============================
STACK_VERIFY_CORRECTION_ENABLED = _env_bool("QARM_STACK_VERIFY_CORRECTION_ENABLED", True)
STACK_VERIFY_REQUIRE_CONFIRMED_FOR_ADVANCE = _env_bool("QARM_STACK_VERIFY_REQUIRE_CONFIRMED_FOR_ADVANCE", True)
STACK_VERIFY_BLOCK_STACK_ON_UNCONFIRMED = _env_bool("QARM_STACK_VERIFY_BLOCK_STACK_ON_UNCONFIRMED", True)
STACK_VERIFY_ALLOW_DOWNWARD_CORRECTION = _env_bool("QARM_STACK_VERIFY_ALLOW_DOWNWARD_CORRECTION", False)
STACK_VERIFY_DOWNWARD_REQUIRE_STABLE_REMEASURE = _env_bool(
    "QARM_STACK_VERIFY_DOWNWARD_REQUIRE_STABLE_REMEASURE",
    True,
)
STACK_REMEASURE_MAX_ATTEMPTS = int(os.getenv("QARM_STACK_REMEASURE_MAX_ATTEMPTS", "3"))
STACK_REMEASURE_REQUIRED_VALID = int(os.getenv("QARM_STACK_REMEASURE_REQUIRED_VALID", "2"))
STACK_REMEASURE_MAX_Z_SPREAD_M = float(os.getenv("QARM_STACK_REMEASURE_MAX_Z_SPREAD_M", "0.025"))
STACK_REMEASURE_PAUSE_S = float(os.getenv("QARM_STACK_REMEASURE_PAUSE_S", "0.05"))

# ============================= Data models =============================
@dataclass
class CycleState:
    cycle_count: int = 0
    picked_count: int = 0
    placed_count: int = 0
    no_pick_miss_count: int = 0
    cycles_without_place_progress: int = 0
    stop_reason: str = "completed"
    skip_final_motion: bool = False
    holding_object: bool = False
    current_hold_grip: float = 0.0
    hold_grip_samples: list[float] = field(default_factory=list)
    blocked_slots: set[int] = field(default_factory=set)
    placed_targets: list[np.ndarray] = field(default_factory=list)
    slot_attempt_count: dict[int, int] = field(default_factory=dict)
    raw_policy_rows: list[dict] = field(default_factory=list)
    policy_step_count: int = 0
    reobserve_requests: int = 0
    reobserve_max_streak: int = 0
    auto_recovery_observes: int = 0
    invalid_precondition_recoveries: int = 0
    policy_invalid_count: int = 0
    place_verify_confirmed_count: int = 0
    place_verify_uncertain_count: int = 0
    pick_stability_fail_count: int = 0
    last_pick_verification: dict = field(default_factory=dict)
    recent_grasp_fail_count: int = 0
    recent_pick_active_miss_count: int = 0
    last_grasp_fail_reason: str = ""
    last_center_failure: dict = field(default_factory=dict)
    last_place_verification: dict = field(default_factory=dict)
    last_return_verify: dict = field(default_factory=dict)
    placed_ledger: list[dict] = field(default_factory=list)
    next_object_id: int = 1
    last_place_verification_v2: dict = field(default_factory=dict)
    track_memory: dict[int, dict] = field(default_factory=dict)
    active_target_track_id: int | None = None
    last_track_snapshot: dict = field(default_factory=dict)
    track_untracked_detections_total: int = 0
    track_untracked_frames: int = 0
    track_last_warn_ms: int = 0
    placed_counts_by_section: dict[str, int] = field(default_factory=lambda: {SECTION_LEFT_NAME: 0, SECTION_RIGHT_NAME: 0})
    returned_count: int = 0
    misplaced_drop_count: int = 0
    last_pick_return_xyz: list[float] | None = None
    last_pick_measured_xyz: list[float] | None = None
    pick_other_block_track_id: int | None = None
    pick_other_block_xyz: list[float] | None = None
    pick_other_block_uv: list[int] | None = None
    pick_other_block_track_ids: list[int] = field(default_factory=list)
    pick_other_block_xyzs: list[list[float]] = field(default_factory=list)
    pick_other_block_uvs: list[list[int]] = field(default_factory=list)
    pick_other_block_source: str = "none"
    last_picked_track_id: int | None = None
    last_picked_uv: list[int] | None = None
    last_verify_lock_uv: list[int] | None = None
    last_verify_lock_xyz: list[float] | None = None
    last_verify_lock_track_id: int | None = None
    last_verify_lock_source: str = "none"
    scene_revision: int = 0
    last_scene_reconcile: dict = field(default_factory=dict)
    scene_snapshot_sections: dict = field(default_factory=dict)
    last_place_space_truth: dict = field(default_factory=dict)
    startup_hydrated_sections: dict = field(default_factory=dict)
    stack_anchor_xyz_by_section: dict[str, list[float]] = field(default_factory=dict)
    stack_anchor_source_by_section: dict[str, str] = field(default_factory=dict)
    last_popped_left_xy: list[float] | None = None
    last_popped_right_xy: list[float] | None = None
    place_space_check_scene_revision: int = -1
    place_space_check_target_track_id: int | None = None
    place_space_check_target_uv: list[int] | None = None
    last_begin_stack_verify: dict = field(default_factory=dict)
    pregrasp_pick_lock_joints: list[float] | None = None
    pregrasp_pick_lock_uv: list[int] | None = None
    pregrasp_pick_lock_track_id: int | None = None
    pregrasp_pick_lock_color: str = "unknown"
    pregrasp_pick_lock_color_conf: float = 0.0
    pick_placed_empty_cooldown_left: int = 0
    pick_placed_empty_cooldown_right: int = 0


@dataclass
class SceneObservation:
    color_frame: object
    depth_frame: object
    image_bgr: np.ndarray
    image_display: np.ndarray
    candidates: list[dict]
    projected_rows: list[dict]
    image_center_uv: tuple[int, int]


# ============================= Current sensing / grip safety =============================
GRIP_CURRENT_LIMITS = GripCurrentLimits(
    grip_detect_a=GRIP_DETECT_A,
    grip_miss_max_a=GRIP_MISS_MAX_A,
    grip_warn_a=GRIP_WARN_A,
    grip_hard_a=GRIP_HARD_A,
    emergency_trip_a=GRIP_EMERGENCY_A,
    transient_ignore_s=GRIP_TRANSIENT_IGNORE_S,
    debounce_samples=GRIP_DEBOUNCE_SAMPLES,
    max_close_s=GRIP_MAX_CLOSE_S,
    final_hold_s=GRIP_FINAL_HOLD_S,
    min_grip=0.2,
    max_grip=MAX_GRIP_CMD,
    grip_step=GRIP_STEP,
    relax_step=GRIP_RELAX_STEP,
    warn_relax_enabled=GRIP_WARN_RELAX_ENABLED,
    warn_relax_step=GRIP_WARN_RELAX_STEP,
    warn_relax_debounce_samples=max(1, int(GRIP_WARN_RELAX_DEBOUNCE)),
    min_detect_grip_cmd=GRIP_MIN_DETECT_CMD,
    min_overcurrent_grip_cmd=GRIP_MIN_OVERCURRENT_CMD,
)
MOTION_SUPERVISION_LIMITS = MotionSupervisionLimits(
    gripper_warn_a=MOTION_GRIP_WARN_A,
    gripper_hard_a=MOTION_GRIP_HARD_A,
    gripper_emergency_a=MOTION_GRIP_EMERGENCY_A,
    total_warn_a=MOTION_TOTAL_WARN_A,
    total_hard_a=MOTION_TOTAL_HARD_A,
    total_emergency_a=MOTION_TOTAL_EMERGENCY_A,
    debounce_samples=MOTION_DEBOUNCE_SAMPLES,
    recover_debounce_samples=max(2, MOTION_DEBOUNCE_SAMPLES // 2),
    freeze_timeout_s=MOTION_FREEZE_TIMEOUT_S,
    relax_step=MOTION_RELAX_STEP,
    min_grip=0.10,
    max_grip=MAX_GRIP_CMD,
    warn_relax_enabled=MOTION_WARN_RELAX_ENABLED,
    warn_relax_step=MOTION_WARN_RELAX_STEP,
    warn_relax_debounce_samples=max(1, int(MOTION_WARN_RELAX_DEBOUNCE)),
)
def get_grasp_z_pick_fraction() -> float:
    return float(GRASP_Z_PICK_FRACTION)


def set_grasp_z_pick_fraction(value: float) -> float:
    global GRASP_Z_PICK_FRACTION
    frac_min = float(GRASP_Z_PICK_FRAC_MIN)
    frac_max = float(GRASP_Z_PICK_FRAC_MAX)
    GRASP_Z_PICK_FRACTION = float(max(frac_min, min(frac_max, float(value))))
    return float(GRASP_Z_PICK_FRACTION)


def get_grip_tune_params() -> dict[str, float]:
    return {
        "grip_detect_a": float(GRIP_CURRENT_LIMITS.grip_detect_a),
        "grip_min_detect_cmd": float(GRIP_CURRENT_LIMITS.min_detect_grip_cmd),
        "grip_miss_max_a": float(GRIP_CURRENT_LIMITS.grip_miss_max_a),
        "grip_step": float(GRIP_CURRENT_LIMITS.grip_step),
        "max_grip_cmd": float(MAX_GRIP_CMD),
    }


def set_grip_tune_params(
    *,
    grip_detect_a: float | None = None,
    grip_min_detect_cmd: float | None = None,
    grip_miss_max_a: float | None = None,
    grip_step: float | None = None,
    max_grip_cmd: float | None = None,
) -> dict[str, float]:
    global GRIP_DETECT_A, GRIP_MIN_DETECT_CMD, GRIP_MISS_MAX_A, GRIP_STEP, MAX_GRIP_CMD, GRIP_DEFAULT
    if max_grip_cmd is not None:
        cap = float(max(0.20, min(0.95, float(max_grip_cmd))))
        MAX_GRIP_CMD = cap
        GRIP_CURRENT_LIMITS.max_grip = cap
        MOTION_SUPERVISION_LIMITS.max_grip = cap
        GRIP_DEFAULT = float(min(float(GRIP_DEFAULT), cap))
    if grip_detect_a is not None:
        detect = float(max(0.01, min(1.00, float(grip_detect_a))))
        GRIP_DETECT_A = detect
        GRIP_CURRENT_LIMITS.grip_detect_a = detect
    if grip_min_detect_cmd is not None:
        min_detect_cmd = float(max(0.20, min(float(MAX_GRIP_CMD), float(grip_min_detect_cmd))))
        GRIP_MIN_DETECT_CMD = min_detect_cmd
        GRIP_CURRENT_LIMITS.min_detect_grip_cmd = min_detect_cmd
    if grip_miss_max_a is not None:
        miss = float(max(0.01, min(1.00, float(grip_miss_max_a))))
        GRIP_MISS_MAX_A = miss
        GRIP_CURRENT_LIMITS.grip_miss_max_a = miss
    if grip_step is not None:
        step = float(max(0.0005, min(0.0500, float(grip_step))))
        GRIP_STEP = step
        GRIP_CURRENT_LIMITS.grip_step = step
    return get_grip_tune_params()


def ensure_tune_profiles_dir() -> Path:
    TUNE_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    return TUNE_PROFILES_DIR


def save_calibration_profile(profile_path: Path, payload: dict) -> Path:
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return profile_path


def load_calibration_profile(profile_path: Path) -> dict | None:
    if not profile_path.exists():
        return None
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def maybe_apply_calibration_profile_from_env(log_prefix: str = "[CalibrationProfile]") -> dict | None:
    profile_path_raw = str(CALIB_PROFILE_PATH or "").strip()
    profile_path: Path | None = None
    profile_source = "env_path"
    if profile_path_raw:
        profile_path = Path(profile_path_raw)
    elif bool(CALIB_PROFILE_AUTO):
        auto_path = ensure_tune_profiles_dir() / str(CALIB_PROFILE_AUTO_NAME)
        if auto_path.exists():
            profile_path = auto_path
            profile_source = "auto_latest"
    if profile_path is None:
        return None
    data = load_calibration_profile(profile_path)
    if not isinstance(data, dict):
        print(f"{log_prefix} load_failed path={profile_path}")
        return None
    try:
        x = float(data.get("cam_off_x_m"))
        y = float(data.get("cam_off_y_m"))
        z = float(data.get("cam_off_z_m"))
    except Exception:
        print(f"{log_prefix} invalid_payload path={profile_path}")
        return None
    old_row = get_cam_offsets()
    old_grasp_frac = float(get_grasp_z_pick_fraction())
    old_grip = get_grip_tune_params()
    set_cam_offsets(cam_off_x_m=x, cam_off_y_m=y, cam_off_z_m=z)
    grasp_frac_loaded = False
    grasp_frac_new = float(old_grasp_frac)
    try:
        if "grasp_z_pick_fraction" in data:
            grasp_frac_new = float(set_grasp_z_pick_fraction(float(data.get("grasp_z_pick_fraction"))))
            grasp_frac_loaded = True
    except Exception:
        grasp_frac_loaded = False
    grip_loaded = False
    grip_loaded_keys: list[str] = []
    try:
        grip_updates: dict[str, float] = {}
        if "grip_detect_a" in data:
            grip_updates["grip_detect_a"] = float(data.get("grip_detect_a"))
            grip_loaded_keys.append("grip_detect_a")
        if "grip_min_detect_cmd" in data:
            grip_updates["grip_min_detect_cmd"] = float(data.get("grip_min_detect_cmd"))
            grip_loaded_keys.append("grip_min_detect_cmd")
        if "grip_miss_max_a" in data:
            grip_updates["grip_miss_max_a"] = float(data.get("grip_miss_max_a"))
            grip_loaded_keys.append("grip_miss_max_a")
        if "grip_step" in data:
            grip_updates["grip_step"] = float(data.get("grip_step"))
            grip_loaded_keys.append("grip_step")
        if "max_grip_cmd" in data:
            grip_updates["max_grip_cmd"] = float(data.get("max_grip_cmd"))
            grip_loaded_keys.append("max_grip_cmd")
        if grip_updates:
            set_grip_tune_params(**grip_updates)
            grip_loaded = True
    except Exception:
        grip_loaded = False
    new_row = get_cam_offsets()
    new_grip = get_grip_tune_params()
    print(
        f"{log_prefix} applied path={profile_path} source={profile_source} "
        f"old=({old_row['cam_off_x_m']:.5f},{old_row['cam_off_y_m']:.5f},{old_row['cam_off_z_m']:.5f}) "
        f"new=({new_row['cam_off_x_m']:.5f},{new_row['cam_off_y_m']:.5f},{new_row['cam_off_z_m']:.5f}) "
        f"grasp_z_pick_fraction={grasp_frac_new:.4f} "
        f"(loaded={grasp_frac_loaded}) "
        f"grip=detect:{new_grip['grip_detect_a']:.3f} miss:{new_grip['grip_miss_max_a']:.3f} "
        f"min_detect_cmd:{new_grip['grip_min_detect_cmd']:.3f} "
        f"step:{new_grip['grip_step']:.4f} max:{new_grip['max_grip_cmd']:.3f} "
        f"(loaded={grip_loaded} keys={grip_loaded_keys})"
    )
    return {
        "path": str(profile_path),
        "source": str(profile_source),
        "old": old_row,
        "new": new_row,
        "grasp_z_pick_fraction_old": float(old_grasp_frac),
        "grasp_z_pick_fraction_new": float(grasp_frac_new),
        "grasp_z_pick_fraction_loaded": bool(grasp_frac_loaded),
        "grip_old": dict(old_grip),
        "grip_new": dict(new_grip),
        "grip_loaded": bool(grip_loaded),
        "grip_loaded_keys": list(grip_loaded_keys),
    }

def save_localization_capture(
    capture_root: str,
    capture_tag: str,
    raw_bgr: np.ndarray,
    annotated_bgr: np.ndarray | None,
    lock_u: int,
    lock_v: int,
    lock_conf: float,
):
    root = Path(capture_root)
    raw_dir = root / "raw"
    ann_dir = root / "annotated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if annotated_bgr is not None:
        ann_dir.mkdir(parents=True, exist_ok=True)
    ts_ms = int(time.time() * 1000)
    stem = f"{capture_tag}_{ts_ms}"
    raw_name = f"{stem}_raw.png"
    ann_name = f"{stem}_annotated.png"
    raw_path = raw_dir / raw_name
    cv2.imwrite(str(raw_path), raw_bgr)
    ann_rel = ""
    if annotated_bgr is not None:
        ann_path = ann_dir / ann_name
        cv2.imwrite(str(ann_path), annotated_bgr)
        ann_rel = str(Path("annotated") / ann_name)
    csv_path = root / "captures.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8") as fp:
        if write_header:
            fp.write("capture_id,timestamp_ms,lock_u,lock_v,lock_conf,raw_image,annotated_image\n")
        fp.write(
            f"{stem},{ts_ms},{int(lock_u)},{int(lock_v)},{float(lock_conf):.6f},"
            f"{str(Path('raw') / raw_name)},{ann_rel}\n"
        )
    return str(raw_path)
# ============================= Arm motion helpers =============================
class Arm:
    def __init__(self):
        self.arm = QArm(hardware=1)
        self.util = QArmUtilities()
        self.sample_time = 1.0 / SAMPLE_RATE_HZ
        self._grip_hold = 0.5  # remember the last grip command we sent
        # tolerances for "did we reach the pose?"
        self.joint_tol_deg = float(os.getenv("QARM_JOINT_TOL_DEG", "2.2"))
        self.settle_time_s = 2.0
        self.last_motion_reason = "ok"
        self.last_motion_diag: dict[str, object] = {}
        self.guard_event_counts = {
            "warn": 0,
            "warn_relax": 0,
            "freeze": 0,
            "recover": 0,
            "unrecoverable": 0,
        }
    def _write(self, phi_cmd: np.ndarray, grip_cmd: float, led=(0, 1, 1)):
        ledCmd = np.array(led, dtype=np.float64)
        g = max(0.0, min(float(grip_cmd), MAX_GRIP_CMD))
        self._grip_hold = g
        # Try common calling conventions across PAL versions
        try:
            # Most robust: positional
            self.arm.read_write_std(phi_cmd, g, ledCmd)
            return
        except TypeError:
            pass
        # Keyword fallbacks (different library versions use different names)
        for kwargs in (
            {"phiCMD": phi_cmd, "grpCMD": g, "baseLED": ledCmd},
            {"phiCMD": phi_cmd, "grpCmd": g, "baseLED": ledCmd},
            {"phiCMD": phi_cmd, "gripCMD": g, "baseLED": ledCmd},
            {"phiCMD": phi_cmd, "gripCmd": g, "baseLED": ledCmd},
            {"phiCMD": phi_cmd, "gripperCMD": g, "baseLED": ledCmd},
            {"phiCMD": phi_cmd, "gripperCmd": g, "baseLED": ledCmd},
        ):
            try:
                self.arm.read_write_std(**kwargs)
                return
            except TypeError:
                continue
        # If we get here, print the signature to help diagnose
        import inspect
        sig = inspect.signature(self.arm.read_write_std)
        raise TypeError(f"Could not call read_write_std with known argument names. Signature is: {sig}")
    def ik(self, xyz: np.ndarray, wrist_angle: float = 0.0) -> np.ndarray:
        """xyz: np.array([x,y,z]) -> returns 4x joint vector."""
        seed = self.arm.measJointPosition[0:4]
        if hasattr(self.util, "qarm_inverse_kinematics"):
            _, phiCmd = self.util.qarm_inverse_kinematics(xyz, wrist_angle, seed)
        else:
            _, phiCmd = self.util.inverse_kinematics(xyz, wrist_angle, seed)
        return phiCmd
    def nudge_joints(self, dphi: np.ndarray):
        """Small joint-space nudge during scanning; keeps gripper where it was."""
        phi = self.arm.measJointPosition[0:4].astype(float) + dphi
        self._write(phi, self._grip_hold)   # reuse last commanded grip
        time.sleep(self.sample_time)
    def tick_hold(self, grip: float | None = None):
        """Refresh the current command once to keep gripper clamp while waiting."""
        g = self._grip_hold if grip is None else clamp_grip_cmd(grip)
        phi_now = self.arm.measJointPosition[0:4].astype(float)
        self._write(phi_now, g)
    @staticmethod
    def _cubic_blend(alpha: float) -> float:
        alpha = max(0.0, min(1.0, float(alpha)))
        return 3.0 * alpha * alpha - 2.0 * alpha * alpha * alpha

    def _apply_motion_supervisor(
        self,
        motion_supervisor: MotionGripSupervisor | None,
        grip_cmd: float,
        label: str,
    ):
        grip_cmd = float(maybe_bleed_carry_grip_on_move(self, grip_cmd, str(label)))
        if motion_supervisor is None:
            return "ok", clamp_grip_cmd(grip_cmd)
        top_phase_tokens = ("align_high", "release_clearance", "retreat_vertical", "home")
        relax_scale = float(MOTION_RELAX_TOP_MULT) if any(tok in str(label) for tok in top_phase_tokens) else 1.0
        g_cur = read_gripper_current_abs(self)
        t_cur = read_total_arm_current_abs(self)
        update = motion_supervisor.update(
            gripper_current_a=g_cur,
            total_current_a=t_cur,
            grip_cmd=grip_cmd,
            relax_scale=relax_scale,
        )
        evt = update.get("event", None)
        if evt in self.guard_event_counts:
            self.guard_event_counts[evt] += 1
        if evt == "warn_relax":
            g_val = float(update.get("gripper_current_a", float("nan")))
            t_val = float(update.get("total_current_a", float("nan")))
            print(
                f"[CurrentGuard] {label}: warn_relax grip->{float(update.get('grip_cmd', grip_cmd)):.3f} "
                f"Ig={g_val:.3f}A Itotal={t_val:.3f}A"
            )
        if evt == "freeze":
            reason = str(update.get("reason", ""))
            g_val = float(update.get("gripper_current_a", float("nan")))
            t_val = float(update.get("total_current_a", float("nan")))
            limits = motion_supervisor.limits
            exceeded_bits: list[str] = []
            if reason == "emergency_threshold":
                if np.isfinite(g_val) and g_val >= float(limits.gripper_emergency_a):
                    exceeded_bits.append(
                        f"Ig={g_val:.3f}A >= grip_emergency={float(limits.gripper_emergency_a):.3f}A"
                    )
                if np.isfinite(t_val) and t_val >= float(limits.total_emergency_a):
                    exceeded_bits.append(
                        f"Itotal={t_val:.3f}A >= total_emergency={float(limits.total_emergency_a):.3f}A"
                    )
            elif reason == "hard_threshold":
                if np.isfinite(g_val) and g_val >= float(limits.gripper_hard_a):
                    exceeded_bits.append(f"Ig={g_val:.3f}A >= grip_hard={float(limits.gripper_hard_a):.3f}A")
                if np.isfinite(t_val) and t_val >= float(limits.total_hard_a):
                    exceeded_bits.append(f"Itotal={t_val:.3f}A >= total_hard={float(limits.total_hard_a):.3f}A")
            exceeded_str = "; ".join(exceeded_bits) if exceeded_bits else "threshold source ambiguous"
            print(
                f"[CurrentGuard] {label}: event={evt}, state={update.get('state')}, "
                f"reason={update.get('reason')}, grip={float(update.get('grip_cmd', grip_cmd)):.3f}, "
                f"Ig={g_val:.3f} A, Itotal={t_val:.3f} A | exceeded: {exceeded_str}"
            )
        return str(update.get("state", "ok")), clamp_grip_cmd(update.get("grip_cmd", grip_cmd))
    def _hold_joint_target(
        self,
        phi_target: np.ndarray,
        grip: float,
        label: str = "",
        motion_supervisor: MotionGripSupervisor | None = None,
    ) -> bool:
        label_text = str(label or "")
        label_text_norm = str(label_text).strip().lower()
        is_release_phase = ("release_clearance" in label_text_norm) or ("retreat_vertical" in label_text_norm)
        is_grasp_lift_phase = ("lift_object_cubic" in label_text_norm)
        if is_release_phase:
            tol_deg_used = float(PLACE_RELEASE_JOINT_TOL_DEG)
        elif is_grasp_lift_phase:
            tol_deg_used = float(max(float(self.joint_tol_deg), float(GRASP_LIFT_JOINT_TOL_DEG)))
        else:
            tol_deg_used = float(self.joint_tol_deg)
        settle_time_s_used = float(PLACE_RELEASE_SETTLE_S) if is_release_phase else float(self.settle_time_s)
        tol_rad = math.radians(float(tol_deg_used))
        t0 = time.time()
        reached = False
        max_err = float("inf")
        grip_cmd = clamp_grip_cmd(grip)
        self.last_motion_reason = "ok"
        motion_state = "ok"
        while (time.time() - t0) < float(max(0.05, settle_time_s_used)):
            self._write(phi_target, grip_cmd)
            state, grip_cmd = self._apply_motion_supervisor(motion_supervisor, grip_cmd, f"{label}_settle")
            motion_state = str(state)
            if state == "unrecoverable":
                self.last_motion_reason = "move_overcurrent_unrecoverable"
                self.last_motion_diag = {
                    "label": label_text,
                    "reached": False,
                    "max_err_deg": float("inf"),
                    "tol_deg_used": float(tol_deg_used),
                    "settle_time_s": float(settle_time_s_used),
                    "motion_state": str(motion_state),
                    "last_motion_reason": str(self.last_motion_reason),
                    "timestamp_ms": int(time.time() * 1000),
                }
                return False
            phi_meas = self.arm.measJointPosition[0:4].astype(float)
            err = phi_target - phi_meas
            max_err = float(np.max(np.abs(err[:3])))
            if max_err <= tol_rad and state != "freeze_recovering":
                reached = True
                break
            time.sleep(self.sample_time)
        max_err_deg = math.degrees(max_err) if np.isfinite(max_err) else float("inf")
        if not reached:
            print(
                f"[WARN] Joint target not fully reached for '{label}' "
                f"(max error ~{max_err_deg:.2f} deg)"
            )
            if self.last_motion_reason == "ok":
                self.last_motion_reason = "joint_settle_timeout"
        self.last_motion_diag = {
            "label": label_text,
            "reached": bool(reached),
            "max_err_deg": float(max_err_deg),
            "tol_deg_used": float(tol_deg_used),
            "settle_time_s": float(settle_time_s_used),
            "motion_state": str(motion_state),
            "last_motion_reason": str(self.last_motion_reason),
            "timestamp_ms": int(time.time() * 1000),
        }
        return reached
    def goto_joints_blocking(
        self,
        phi_target: np.ndarray,
        grip: float | None = None,
        duration: float = 1.0,
        steps: int = 60,
        label: str = "",
        motion_supervisor: MotionGripSupervisor | None = None,
    ) -> bool:
        if grip is None:
            grip = self._grip_hold
        phi_target = np.array(phi_target, dtype=float)
        phi_start = self.arm.measJointPosition[0:4].astype(float)
        # 1) Cubic interpolation (smooth start/stop)
        delta = phi_target - phi_start
        grip_cmd = clamp_grip_cmd(grip)
        self.last_motion_reason = "ok"
        k = 1
        while k <= steps:
            t = k / steps
            blend = self._cubic_blend(t)
            phi_cmd = phi_start + blend * delta
            self._write(phi_cmd, grip_cmd)
            state, grip_cmd = self._apply_motion_supervisor(motion_supervisor, grip_cmd, f"{label}_interp")
            if state == "unrecoverable":
                self.last_motion_reason = "move_overcurrent_unrecoverable"
                return False
            if state != "freeze_recovering":
                k += 1
            time.sleep(duration / max(1, steps))
        return self._hold_joint_target(
            phi_target,
            float(grip_cmd),
            label,
            motion_supervisor=motion_supervisor,
        )
    def goto_joints_waypoints_cubic(
        self,
        phi_waypoints: list[np.ndarray],
        grip_waypoints: list[float] | None = None,
        segment_duration: float = 1.0,
        steps_per_segment: int = 50,
        label: str = "",
        motion_supervisor: MotionGripSupervisor | None = None,
    ) -> bool:
        if not phi_waypoints:
            return False
        q_list = [np.array(self.arm.measJointPosition[0:4], dtype=float)]
        q_list.extend(np.array(q, dtype=float) for q in phi_waypoints)
        if grip_waypoints is None:
            g_list = [float(self._grip_hold)] + [float(self._grip_hold)] * len(phi_waypoints)
        else:
            if len(grip_waypoints) != len(phi_waypoints):
                raise ValueError("grip_waypoints must match phi_waypoints length.")
            g_list = [float(self._grip_hold)] + [clamp_grip_cmd(g) for g in grip_waypoints]
        n = len(q_list)
        tangents = [np.zeros(4, dtype=float) for _ in range(n)]
        for i in range(1, n - 1):
            tangents[i] = 0.5 * (q_list[i + 1] - q_list[i - 1])
        self.last_motion_reason = "ok"
        grip_cmd = float(g_list[0])
        for seg in range(n - 1):
            q0 = q_list[seg]
            q1 = q_list[seg + 1]
            m0 = tangents[seg]
            m1 = tangents[seg + 1]
            g0 = g_list[seg]
            g1 = g_list[seg + 1]
            k = 1
            while k <= steps_per_segment:
                t = k / steps_per_segment
                t2 = t * t
                t3 = t2 * t
                h00 = 2 * t3 - 3 * t2 + 1
                h10 = t3 - 2 * t2 + t
                h01 = -2 * t3 + 3 * t2
                h11 = t3 - t2
                phi_cmd = h00 * q0 + h10 * m0 + h01 * q1 + h11 * m1
                grip_cmd = (1 - t) * g0 + t * g1 if motion_supervisor is None else grip_cmd
                self._write(phi_cmd, grip_cmd)
                state, grip_cmd = self._apply_motion_supervisor(motion_supervisor, grip_cmd, f"{label}_seg{seg}")
                if state == "unrecoverable":
                    self.last_motion_reason = "move_overcurrent_unrecoverable"
                    return False
                if state != "freeze_recovering":
                    k += 1
                time.sleep(segment_duration / steps_per_segment)
        return self._hold_joint_target(
            q_list[-1],
            grip_cmd,
            label or "joint_waypoints_cubic",
            motion_supervisor=motion_supervisor,
        )

    def goto_task_space(
        self,
        pos: np.ndarray,
        duration: float = 1.0,
        steps: int = 100,
        label: str = "",
        motion_supervisor: MotionGripSupervisor | None = None,
    ) -> bool:
        target_xyz = pos[:3].astype(float)
        target_grip = float(pos[3]) if pos.size > 3 else self._grip_hold
        phi_target = self.ik(target_xyz, 0.0)
        return self.goto_joints_blocking(
            phi_target,
            grip=target_grip,
            duration=duration,
            steps=steps,
            label=label or "task_space_move",
            motion_supervisor=motion_supervisor,
        )
    def goto_task_waypoints_cubic(
        self,
        poses: list[np.ndarray],
        segment_duration: float = 1.0,
        steps_per_segment: int = 60,
        label: str = "",
        motion_supervisor: MotionGripSupervisor | None = None,
    ) -> bool:
        if not poses:
            return False
        phi_waypoints = []
        grip_waypoints = []
        for pose in poses:
            target_xyz = np.array(pose[:3], dtype=float)
            target_grip = float(pose[3]) if len(pose) > 3 else float(self._grip_hold)
            phi_waypoints.append(self.ik(target_xyz, 0.0))
            grip_waypoints.append(target_grip)
        return self.goto_joints_waypoints_cubic(
            phi_waypoints=phi_waypoints,
            grip_waypoints=grip_waypoints,
            segment_duration=segment_duration,
            steps_per_segment=steps_per_segment,
            label=label or "task_waypoints_cubic",
            motion_supervisor=motion_supervisor,
        )

# RealSense D4xx: 640x480 @ 30 fps is widely supported; native 640x640 is not (raises
# "Couldn't resolve requests"). Square 640 infer for YOLO uses QARM_YOLO_STRETCH_* instead.
REALSENSE_COLOR_WIDTH = int(os.getenv("QARM_REALSENSE_WIDTH", "640"))
REALSENSE_COLOR_HEIGHT = int(os.getenv("QARM_REALSENSE_HEIGHT", "480"))
REALSENSE_FPS = int(os.getenv("QARM_REALSENSE_FPS", "30"))


# ============================= Perception pipeline =============================
class Perception:
    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ):
        width_i = int(REALSENSE_COLOR_WIDTH if width is None else width)
        height_i = int(REALSENSE_COLOR_HEIGHT if height is None else height)
        fps_i = int(REALSENSE_FPS if fps is None else fps)
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, width_i, height_i, rs.format.z16, fps_i)
        cfg.enable_stream(rs.stream.color, width_i, height_i, rs.format.bgr8, fps_i)
        try:
            profile = self.pipeline.start(cfg)
        except RuntimeError as exc:
            raise RuntimeError(
                f"RealSense pipeline start failed for {width_i}x{height_i}@{fps_i}fps: {exc}. "
                "D4xx cameras typically support 640x480@30; 640x640 is not a valid stream mode. "
                "Use QARM_REALSENSE_WIDTH/HEIGHT=640/480 and QARM_YOLO_STRETCH_SIZE=640 for square detect."
            ) from exc
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        # Warmup frames
        for _ in range(6):
            self.pipeline.wait_for_frames()
        # Cache intrinsics from color
        frames = self.align.process(self.pipeline.wait_for_frames())
        color = frames.get_color_frame()
        self.intr = rs.video_stream_profile(color.profile).get_intrinsics()
    def get_frames(self):
        frames = self.align.process(self.pipeline.wait_for_frames())
        return frames.get_color_frame(), frames.get_depth_frame()
    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
# ============================= Current sensing / grip safety =============================
def clamp_grip_cmd(value: float) -> float:
    return max(0.0, min(float(value), MAX_GRIP_CMD))


def apply_place_command_xy_offset(xyz: np.ndarray, *, skip: bool = False) -> np.ndarray:
    """Shift commanded place XY only (vision/kinematic bias); Z is never modified."""
    out = np.array(xyz, dtype=float).reshape(-1).copy()
    if bool(skip) or not bool(PLACE_CMD_XY_OFFSET_ENABLED):
        return out
    if out.size >= 1 and np.isfinite(out[0]):
        out[0] = float(out[0]) + float(PLACE_CMD_X_OFFSET_M)
    if out.size >= 2 and np.isfinite(out[1]):
        out[1] = float(out[1]) + float(PLACE_CMD_Y_OFFSET_M)
    return out


def apply_place_pick_bias_compensate(xyz: np.ndarray) -> tuple[np.ndarray, float, float, bool]:
    """Subtract scaled grasp pick XY bias from place command (inverse of pick close offset)."""
    out = np.array(xyz, dtype=float).reshape(-1).copy()
    if not bool(PLACE_PICK_BIAS_COMPENSATE_ENABLED):
        return out, 0.0, 0.0, False
    scale = max(0.0, float(PLACE_PICK_BIAS_COMPENSATE_SCALE))
    dx = -float(scale) * float(GRASP_PICK_X_BIAS_M)
    dy = -float(scale) * float(GRASP_PICK_Y_BIAS_M)
    if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
        return out, 0.0, 0.0, False
    if out.size >= 1 and np.isfinite(out[0]):
        out[0] = float(out[0]) + float(dx)
    if out.size >= 2 and np.isfinite(out[1]):
        out[1] = float(out[1]) + float(dy)
    return out, float(dx), float(dy), True


def compute_stack_pick_x_offset(pick_xyz) -> tuple[float, dict]:
    """Command-only stack X offset derived from the successful pick's measured X."""
    meta = {
        "enabled": bool(STACK_PICK_X_OFFSET_ENABLED),
        "applied": False,
        "reason": "disabled",
        "pick_xyz": None,
        "pick_x_m": None,
        "near_x_m": float(STACK_PICK_X_NEAR_M),
        "far_x_m": float(STACK_PICK_X_FAR_M),
        "near_offset_m": float(STACK_PICK_X_OFFSET_NEAR_M),
        "far_offset_m": float(STACK_PICK_X_OFFSET_FAR_M),
        "clamp_min_m": float(STACK_PICK_X_OFFSET_CLAMP_MIN_M),
        "clamp_max_m": float(STACK_PICK_X_OFFSET_CLAMP_MAX_M),
        "t": 0.0,
        "raw_offset_m": 0.0,
        "offset_m": 0.0,
    }
    if not bool(STACK_PICK_X_OFFSET_ENABLED):
        return 0.0, meta
    try:
        vals = list(pick_xyz)
        if len(vals) < 3:
            meta["reason"] = "missing_pick_xyz"
            return 0.0, meta
        pick_arr = np.array([float(vals[0]), float(vals[1]), float(vals[2])], dtype=float)
    except Exception:
        meta["reason"] = "invalid_pick_xyz"
        return 0.0, meta
    if not np.all(np.isfinite(pick_arr[:3])):
        meta["reason"] = "invalid_pick_xyz"
        return 0.0, meta
    near_x = float(STACK_PICK_X_NEAR_M)
    far_x = float(STACK_PICK_X_FAR_M)
    span = float(far_x - near_x)
    if not np.isfinite(span) or abs(span) <= 1e-9:
        meta["reason"] = "invalid_pick_x_range"
        meta["pick_xyz"] = [float(pick_arr[0]), float(pick_arr[1]), float(pick_arr[2])]
        meta["pick_x_m"] = float(pick_arr[0])
        return 0.0, meta
    t = (float(pick_arr[0]) - near_x) / span
    t = min(1.0, max(0.0, float(t)))
    raw_dx = float(STACK_PICK_X_OFFSET_NEAR_M) + t * (
        float(STACK_PICK_X_OFFSET_FAR_M) - float(STACK_PICK_X_OFFSET_NEAR_M)
    )
    clamp_min = min(float(STACK_PICK_X_OFFSET_CLAMP_MIN_M), float(STACK_PICK_X_OFFSET_CLAMP_MAX_M))
    clamp_max = max(float(STACK_PICK_X_OFFSET_CLAMP_MIN_M), float(STACK_PICK_X_OFFSET_CLAMP_MAX_M))
    dx = min(clamp_max, max(clamp_min, float(raw_dx)))
    meta.update(
        {
            "applied": abs(float(dx)) > 1e-9,
            "reason": "ok",
            "pick_xyz": [float(pick_arr[0]), float(pick_arr[1]), float(pick_arr[2])],
            "pick_x_m": float(pick_arr[0]),
            "t": float(t),
            "raw_offset_m": float(raw_dx),
            "offset_m": float(dx),
        }
    )
    return float(dx), meta


def read_gripper_current_abs(arm: Arm) -> float:
    try:
        currents = np.asarray(getattr(arm.arm, "measJointCurrent", []), dtype=float).reshape(-1)
    except Exception:
        return float("nan")
    if currents.size >= 5 and np.isfinite(currents[4]):
        return float(abs(currents[4]))
    return float("nan")
def read_total_arm_current_abs(arm: Arm) -> float:
    try:
        total = read_total_arm_current(arm.arm)
    except Exception:
        return float("nan")
    return float(total) if np.isfinite(total) else float("nan")
def make_motion_supervisor(initial_grip: float, label: str) -> MotionGripSupervisor | None:
    if not MOTION_GUARD_ENABLED:
        return None
    return MotionGripSupervisor(
        limits=MOTION_SUPERVISION_LIMITS,
        initial_grip=clamp_grip_cmd(initial_grip),
        label=label,
    )


def carry_bleed_should_apply_for_label(label: str) -> bool:
    label_norm = str(label or "").strip().lower()
    if not label_norm:
        return False
    carry_tokens = ("_carry", "pick_misplaced", "misplaced", "correction", "grasp_lift", "lift_object")
    return any(tok in label_norm for tok in carry_tokens)


def maybe_bleed_carry_grip_on_move(arm: Arm, grip_cmd: float, label: str) -> float:
    """One relax step during carry when instantaneous grip current exceeds carry max."""
    if not bool(POST_LIFT_CARRY_BLEED_ENABLED) or not bool(POST_LIFT_CARRY_BLEED_MOVE_TIME):
        return float(clamp_grip_cmd(grip_cmd))
    if not carry_bleed_should_apply_for_label(str(label)):
        return float(clamp_grip_cmd(grip_cmd))
    grip = float(clamp_grip_cmd(grip_cmd))
    grip_floor = float(clamp_grip_cmd(POST_LIFT_CARRY_MIN_GRIP_CMD))
    if grip <= float(grip_floor) + 1e-6:
        return grip
    try:
        i_now = float(read_gripper_current_abs(arm))
    except Exception:
        return grip
    if not np.isfinite(i_now) or float(i_now) <= float(POST_LIFT_CARRY_MAX_A):
        return grip
    step = max(0.0005, float(POST_LIFT_CARRY_TUNE_STEP))
    next_grip = float(clamp_grip_cmd(max(grip_floor, grip - step)))
    if next_grip < grip - 1e-6:
        print(
            f"[CarryBleed] move-time i={float(i_now):.3f} > max={float(POST_LIFT_CARRY_MAX_A):.3f} "
            f"grip {grip:.3f}->{next_grip:.3f} label={str(label)}"
        )
        arm.tick_hold(grip=next_grip)
    return next_grip


def extract_projected_detections(
    det: YOLODetector | None,
    arm: Arm,
    per: Perception | None,
    bgr_img: np.ndarray,
    depth_frame,
    min_conf: float = TRACK_MIN_CONF,
) -> list[dict]:
    if det is None or per is None:
        return []
    _img_unused, candidates = det.detect_candidates_and_draw(bgr_img, draw=False)
    if bool(YOLO_BBOX_SPLIT_ENABLED):
        candidates = split_merged_stack_candidates(
            candidates,
            arm=arm,
            per=per,
            depth_frame=depth_frame,
            layer_dz_m=float(STACK_LEVEL_DZ_M),
            min_height_m=float(YOLO_BBOX_SPLIT_MIN_HEIGHT_M),
            min_aspect=float(YOLO_BBOX_SPLIT_MIN_ASPECT),
            max_cubes=int(YOLO_BBOX_SPLIT_MAX_CUBES),
        )
    return project_candidates_to_base(
        arm=arm,
        per=per,
        depth_frame=depth_frame,
        candidates=candidates,
        min_conf=min_conf,
    )


def observe_scene_frame(
    det: YOLODetector | None,
    arm: Arm,
    per: Perception | None,
    *,
    draw: bool = False,
    projected_min_conf: float = TRACK_MIN_CONF,
    state: CycleState | None = None,
    update_tracks: bool = False,
) -> SceneObservation | None:
    if det is None or per is None:
        return None
    color, depth = per.get_frames()
    img = np.asanyarray(color.get_data())
    cx, cy = per.intr.width // 2, per.intr.height // 2
    img_display, candidates = det.detect_candidates_and_draw(img, draw=draw)
    if bool(YOLO_BBOX_SPLIT_ENABLED):
        candidates = split_merged_stack_candidates(
            candidates,
            arm=arm,
            per=per,
            depth_frame=depth,
            layer_dz_m=float(STACK_LEVEL_DZ_M),
            min_height_m=float(YOLO_BBOX_SPLIT_MIN_HEIGHT_M),
            min_aspect=float(YOLO_BBOX_SPLIT_MIN_ASPECT),
            max_cubes=int(YOLO_BBOX_SPLIT_MAX_CUBES),
        )
    projected_rows = project_candidates_to_base(
        arm=arm,
        per=per,
        depth_frame=depth,
        candidates=candidates,
        min_conf=float(projected_min_conf),
    )
    if state is not None and TRACK_ENABLE and update_tracks:
        update_cube_tracks(
            state=state,
            detections=projected_rows,
            max_miss_frames=TRACK_MAX_MISS_FRAMES,
            image_center_uv=(int(cx), int(cy)),
        )
    return SceneObservation(
        color_frame=color,
        depth_frame=depth,
        image_bgr=img,
        image_display=img_display,
        candidates=candidates,
        projected_rows=projected_rows,
        image_center_uv=(int(cx), int(cy)),
    )


def render_operator_overlay(
    frame: np.ndarray,
    state: CycleState | None,
    ui_mode: str,
    tracks: dict[int, dict],
    active_track_id: int | None,
    cx: int,
    cy: int,
    selected_uv: tuple[int, int] | None = None,
    status_line: str = "",
) -> np.ndarray:
    img = frame.copy()
    mode = str(ui_mode or "minimal").strip().lower()
    h, w = img.shape[:2]
    cv2.drawMarker(img, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.circle(img, (cx, cy), PX_TOL, (0, 255, 255), 2)

    if mode == "debug":
        for track_id, row in tracks.items():
            uv = row.get("uv", None)
            if not isinstance(uv, (list, tuple)) or len(uv) < 2:
                continue
            u = int(uv[0])
            v = int(uv[1])
            miss = int(row.get("miss_frames", 0))
            conf = float(row.get("conf", 0.0))
            color = (0, 255, 255) if int(track_id) == int(active_track_id or -1) else ((0, 220, 0) if miss == 0 else (120, 120, 120))
            cv2.circle(img, (u, v), 6, color, -1)
            cv2.putText(
                img,
                f"id={int(track_id)} m={miss} c={conf:.2f}",
                (u + 6, v - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
    elif UI_SHOW_TRACK_IDS:
        # Minimal-mode visibility: show active target track id by default.
        rows_to_draw: list[tuple[int, dict]] = []
        if active_track_id is not None and active_track_id in tracks:
            rows_to_draw.append((int(active_track_id), tracks[int(active_track_id)]))
        if UI_SHOW_ALL_TRACK_IDS_MINIMAL:
            for track_id, row in tracks.items():
                if int(row.get("miss_frames", 0)) != 0:
                    continue
                if active_track_id is not None and int(track_id) == int(active_track_id):
                    continue
                rows_to_draw.append((int(track_id), row))
        for track_id, row in rows_to_draw:
            uv = row.get("uv", None)
            if not isinstance(uv, (list, tuple)) or len(uv) < 2:
                continue
            u = int(uv[0])
            v = int(uv[1])
            miss = int(row.get("miss_frames", 0))
            conf = float(row.get("conf", 0.0))
            is_active = (active_track_id is not None and int(track_id) == int(active_track_id))
            color = (0, 255, 255) if is_active else (0, 220, 0)
            radius = 7 if is_active else 5
            cv2.circle(img, (u, v), radius, color, 2)
            cv2.putText(
                img,
                f"id={int(track_id)} c={conf:.2f} m={miss}",
                (u + 8, v - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
            )

    if selected_uv is not None:
        cv2.circle(img, (int(selected_uv[0]), int(selected_uv[1])), 9, (255, 180, 0), 2)

    visible_count = sum(1 for row in tracks.values() if int(row.get("miss_frames", 0)) == 0)
    active_text = "none" if active_track_id is None else str(int(active_track_id))
    lines = [f"tracks={visible_count} active={active_text} mode={mode}"]
    if status_line:
        lines.append(status_line)
    if state is not None and state.placed_counts_by_section:
        left_n = int(state.placed_counts_by_section.get(SECTION_LEFT_NAME, 0))
        right_n = int(state.placed_counts_by_section.get(SECTION_RIGHT_NAME, 0))
        lines.append(f"placed {SECTION_LEFT_NAME}:{left_n} {SECTION_RIGHT_NAME}:{right_n}")
        vis = state.last_track_snapshot.get("visible_section_counts", {})
        left_v = int(vis.get(SECTION_LEFT_NAME, 0))
        right_v = int(vis.get(SECTION_RIGHT_NAME, 0))
        lines.append(f"visible {SECTION_LEFT_NAME}:{left_v} {SECTION_RIGHT_NAME}:{right_v}")

    # Keep tracking text in a bottom panel to avoid overlap with centering prompts.
    panel_line_h = 20
    panel_pad = 8
    panel_h = panel_pad * 2 + panel_line_h * len(lines)
    panel_y0 = max(0, h - panel_h - 6)
    panel_x0 = 6
    panel_x1 = min(w - 6, 470)
    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x0, panel_y0), (panel_x1, panel_y0 + panel_h), (20, 20, 20), -1)
    alpha = 0.55
    img = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0)

    for idx, line in enumerate(lines):
        y = panel_y0 + panel_pad + 14 + idx * panel_line_h
        color = (245, 245, 245) if idx == 0 else (215, 215, 215)
        cv2.putText(
            img,
            line,
            (panel_x0 + panel_pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            1,
        )
    return img


def _finite_xyz_or_none(xyz: np.ndarray | list[float] | None) -> list[float] | None:
    if xyz is None:
        return None
    arr = np.array(xyz, dtype=float).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


# ============================= Runtime utilities =============================
def _match_projected_row_by_uv(projected_rows: list[dict], u: int, v: int, max_px: float = 16.0) -> dict | None:
    if not projected_rows:
        return None
    max_d2 = float(max_px) * float(max_px)
    best = None
    best_d2 = float("inf")
    for row in projected_rows:
        ru = int(row.get("u", 0))
        rv = int(row.get("v", 0))
        d2 = float((ru - int(u)) ** 2 + (rv - int(v)) ** 2)
        if d2 < best_d2:
            best_d2 = d2
            best = row
    if best is None or best_d2 > max_d2:
        return None
    return best


# ============================= Current sensing / grip safety =============================
def verify_grasp_in_air_current(
    arm: Arm,
    hold_grip: float,
    samples: int = POST_LIFT_VERIFY_SAMPLES,
    min_hits: int = POST_LIFT_VERIFY_MIN_HITS,
    min_current_a: float = POST_LIFT_VERIFY_MIN_CURRENT_A,
):
    samples = max(1, int(samples))
    min_hits = max(1, min(int(min_hits), samples))
    hits = 0
    values: list[float] = []
    t_end = time.time() + max(0.0, float(POST_LIFT_VERIFY_HOLD_S))
    sample_idx = 0
    while sample_idx < samples or time.time() < t_end:
        arm.tick_hold(grip=hold_grip)
        current_a = read_gripper_current_abs(arm)
        if np.isfinite(current_a):
            values.append(float(current_a))
            if current_a >= float(min_current_a):
                hits += 1
        sample_idx += 1
        time.sleep(max(0.0, arm.sample_time))
    median_a = float(np.median(values)) if values else float("nan")
    max_a = float(np.max(values)) if values else float("nan")
    min_a = float(np.min(values)) if values else float("nan")
    ok = hits >= min_hits
    print(
        f"[PostLiftVerify] {'PASS' if ok else 'FAIL'}: "
        f"hits={hits}/{max(1, len(values))} above {min_current_a:.3f} A | "
        f"median={median_a:.3f} A, min={min_a:.3f} A, max={max_a:.3f} A"
    )
    return ok, median_a, max_a

def tune_hold_grip_to_current_target(
    arm: Arm,
    hold_grip: float,
    target_min_a: float = POST_LIFT_TUNE_TARGET_A,
    target_max_a: float | None = None,
    *,
    relax_only: bool = False,
    max_steps_override: int | None = None,
    tune_step_override: float | None = None,
    min_grip_cmd: float | None = None,
):
    if not POST_LIFT_TUNE_ENABLED:
        return clamp_grip_cmd(hold_grip), {"status": "disabled"}
    grip = clamp_grip_cmd(hold_grip)
    target_min = max(0.01, float(target_min_a))
    target_max = (
        float(target_max_a)
        if target_max_a is not None
        else float(GRIP_CURRENT_LIMITS.grip_warn_a)
    )
    max_steps = max(1, int(max_steps_override if max_steps_override is not None else POST_LIFT_TUNE_MAX_STEPS))
    tune_step = max(
        0.0005,
        float(tune_step_override if tune_step_override is not None else POST_LIFT_TUNE_STEP),
    )
    tune_samples = max(1, int(POST_LIFT_TUNE_SAMPLES))
    grip_floor = clamp_grip_cmd(
        float(POST_LIFT_CARRY_MIN_GRIP_CMD if min_grip_cmd is None else min_grip_cmd)
    )
    hard_a = float(GRIP_CURRENT_LIMITS.grip_hard_a)
    last_median = float("nan")
    last_peak = float("nan")

    for step_i in range(max_steps):
        values: list[float] = []
        for _ in range(tune_samples):
            arm.tick_hold(grip=grip)
            current_a = read_gripper_current_abs(arm)
            if np.isfinite(current_a):
                values.append(float(current_a))
            time.sleep(max(0.0, arm.sample_time))
        if values:
            last_median = float(np.median(values))
            last_peak = float(np.max(values))
        else:
            last_median = float("nan")
            last_peak = float("nan")

        current_in_band = bool(
            np.isfinite(last_median) and float(last_median) >= target_min and float(last_median) <= target_max
        )
        at_grip_floor = bool(float(grip) <= float(grip_floor) + 1e-6)

        if bool(relax_only):
            if current_in_band and at_grip_floor:
                return grip, {
                    "status": "target_reached",
                    "steps": int(step_i + 1),
                    "median_a": float(last_median),
                    "peak_a": float(last_peak),
                }
            if np.isfinite(last_median) and float(last_median) >= hard_a:
                next_grip = clamp_grip_cmd(max(grip_floor, grip - max(tune_step, float(GRIP_RELAX_STEP))))
            elif np.isfinite(last_median) and float(last_median) > target_max:
                next_grip = clamp_grip_cmd(max(grip_floor, grip - tune_step))
            elif float(grip) > float(grip_floor) + 1e-6:
                next_grip = clamp_grip_cmd(max(grip_floor, grip - tune_step))
            else:
                return grip, {
                    "status": "clamped",
                    "steps": int(step_i + 1),
                    "median_a": float(last_median),
                    "peak_a": float(last_peak),
                }
        else:
            if current_in_band:
                return grip, {
                    "status": "target_reached",
                    "steps": int(step_i + 1),
                    "median_a": float(last_median),
                    "peak_a": float(last_peak),
                }
            if np.isfinite(last_median) and last_median >= hard_a:
                next_grip = clamp_grip_cmd(grip - max(tune_step, float(GRIP_RELAX_STEP)))
            elif np.isfinite(last_median) and last_median > target_max:
                next_grip = clamp_grip_cmd(grip - tune_step)
            else:
                next_grip = clamp_grip_cmd(grip + tune_step)

        if abs(float(next_grip) - float(grip)) <= 1e-6:
            return grip, {
                "status": "clamped",
                "steps": int(step_i + 1),
                "median_a": float(last_median),
                "peak_a": float(last_peak),
            }
        grip = float(next_grip)

    return grip, {
        "status": "max_steps",
        "steps": int(max_steps),
        "median_a": float(last_median),
        "peak_a": float(last_peak),
    }

def verify_pick_stability_signal(arm: Arm, hold_grip: float) -> dict:
    if not PICK_STABILITY_RECHECK_ENABLED:
        return {"status": "disabled", "ok": True}
    time.sleep(max(0.0, float(PICK_STABILITY_RECHECK_DELAY_S)))
    ok, median_a, max_a = verify_grasp_in_air_current(
        arm=arm,
        hold_grip=hold_grip,
        samples=max(1, int(PICK_STABILITY_RECHECK_SAMPLES)),
        min_hits=max(1, int(PICK_STABILITY_RECHECK_MIN_HITS)),
        min_current_a=float(PICK_STABILITY_RECHECK_MIN_CURRENT_A),
    )
    status = "stable" if ok else "unstable"
    return {
        "status": status,
        "ok": bool(ok),
        "median_a": float(median_a),
        "max_a": float(max_a),
        "threshold_a": float(PICK_STABILITY_RECHECK_MIN_CURRENT_A),
        "samples": int(PICK_STABILITY_RECHECK_SAMPLES),
        "min_hits": int(PICK_STABILITY_RECHECK_MIN_HITS),
    }

def classify_cube_color_patch(
    bgr_img: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int] | None = None,
    center_uv: tuple[int, int] | None = None,
    patch_size: int = 72,
    bbox_core_ratio: float = 1.0,
    bbox_core_y_bias: float = 0.68,
) -> tuple[str, float]:
    """
    Lightweight orange/blue classifier for live routing.
    Uses HSV hue concentration while suppressing dark/low-saturation pixels.
    Tuned for orange + light-cyan/blue paper cubes under lab lighting.
    """
    h, w = bgr_img.shape[:2]
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        core_ratio = float(max(0.20, min(1.0, bbox_core_ratio)))
        if core_ratio < 1.0:
            bw = max(1, int(x2 - x1))
            bh = max(1, int(y2 - y1))
            core_w = max(1, int(round(float(bw) * core_ratio)))
            core_h = max(1, int(round(float(bh) * core_ratio)))
            cx = int(round((x1 + x2) * 0.5))
            y_bias = float(max(0.0, min(1.0, bbox_core_y_bias)))
            cy = int(round(float(y1) + (float(bh) - 1.0) * y_bias))
            x1 = max(0, min(w - 1, int(cx - (core_w // 2))))
            y1 = max(0, min(h - 1, int(cy - (core_h // 2))))
            x2 = max(x1 + 1, min(w, int(cx + ((core_w + 1) // 2))))
            y2 = max(y1 + 1, min(h, int(cy + ((core_h + 1) // 2))))
        crop = bgr_img[y1:y2, x1:x2]
    elif center_uv is not None:
        u, v = [int(x) for x in center_uv]
        half = max(10, int(patch_size // 2))
        x1, x2 = max(0, u - half), min(w, u + half)
        y1, y2 = max(0, v - half), min(h, v + half)
        crop = bgr_img[y1:y2, x1:x2]
    else:
        crop = bgr_img
    if crop.size == 0:
        return "unknown", 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.float32)
    sat = hsv[..., 1].astype(np.float32)
    val = hsv[..., 2].astype(np.float32)
    valid_mask = (sat >= float(COLOR_MIN_SAT)) & (val >= float(COLOR_MIN_VAL))
    if np.count_nonzero(valid_mask) < int(COLOR_MIN_VALID_PIXELS):
        return "unknown", 0.0
    hv = hue[valid_mask]
    # OpenCV hue range: [0,179].
    orange_mask = ((hv >= float(COLOR_ORANGE_H_MIN)) & (hv <= float(COLOR_ORANGE_H_MAX)))
    blue_mask = ((hv >= float(COLOR_BLUE_H_MIN)) & (hv <= float(COLOR_BLUE_H_MAX)))
    orange_score = float(np.mean(orange_mask))
    blue_score = float(np.mean(blue_mask))
    score_sum = orange_score + blue_score
    if score_sum <= 1e-6:
        return "unknown", 0.0
    conf = float(max(orange_score, blue_score) / score_sum)
    if orange_score > blue_score:
        return "orange", conf
    return "blue", conf

def measure_base_point_from_uv(arm: Arm, per: Perception, u: int, v: int, n=N_MEAS, win=MEAS_MEDIAN_WIN):
    xs, ys, zs = [], [], []
    print(f"  Taking {n} depth measurements at pixel ({u}, {v})...")
    for i in range(max(1, n)):
        _, depth = per.get_frames()
        Z_cam = robust_depth_m(depth, u, v, per.depth_scale, win=MEAS_MEDIAN_WIN, percentile=95)
        if not np.isfinite(Z_cam) or Z_cam <= 0:
            print(f"    Sample {i+1}/{n}: Invalid depth (Z={Z_cam})")
            continue
        Xc, Yc, Zc = uvz_to_xyz_cam(u, v, Z_cam, per.intr)
        if not np.all(np.isfinite([Xc, Yc, Zc])):
            print(f"    Sample {i+1}/{n}: Invalid camera coords")
            continue
        yaw, shoulder, elbow = arm.arm.measJointPosition[0:3].astype(float)
        T_cam_to_base = base_to_camera_T(yaw, shoulder, elbow)
        p_cam  = np.array([Xc, Yc, Zc, 1.0], dtype=float)
        p_base = T_cam_to_base @ p_cam
        bx, by, bz = map(float, p_base[:3])
        xs.append(bx)
        ys.append(by)
        zs.append(bz)
        print(f"    Sample {i+1}/{n}: Z_cam={Z_cam:.4f}m -> cam=({Xc:.4f}, {Yc:.4f}, {Zc:.4f}) "
              f"-> base=({bx:.4f}, {by:.4f}, {bz:.4f})")
        print(f"               Joints: yaw={np.rad2deg(yaw):.1f}deg, "
              f"shoulder={np.rad2deg(shoulder):.1f}deg, elbow={np.rad2deg(elbow):.1f}deg")
        time.sleep(0.02)
    if not xs:
        print("  FAILED: No valid depth measurements obtained")
        return np.nan, np.nan, np.nan
    avg_x = float(np.mean(xs))
    avg_y = float(np.mean(ys))
    avg_z = float(np.mean(zs))
    std_x = float(np.std(xs))
    std_y = float(np.std(ys))
    std_z = float(np.std(zs))
    print(f"  OK Average BASE coords: x={avg_x:.4f}m, y={avg_y:.4f}m, z={avg_z:.4f}m")
    print(f"    Standard deviation:  x={std_x:.4f}m, y={std_y:.4f}m, z={std_z:.4f}m")
    if avg_z < 0.0:
        print(f"  WARNING: Z is still negative ({avg_z:.3f}m); check DH / camera mount conventions")
    return avg_x, avg_y, avg_z

# ============================= Candidate filtering =============================
def pick_workspace_y_max_m() -> float:
    default_y_max = float(PICK_MAX_BASE_Y_M) + max(0.0, float(SCENE_RECON_PLACE_Y_MARGIN_M))
    return float(os.getenv("QARM_PICK_WORKSPACE_Y_MAX_M", str(default_y_max)))


def xyz_in_pick_workspace(xyz) -> bool:
    arr = np.array(xyz, dtype=float).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return False
    return float(arr[1]) <= float(pick_workspace_y_max_m())


def pick_workspace_reject_reason(xyz) -> str | None:
    arr = np.array(xyz, dtype=float).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return "candidate_pick_space_rejected invalid_xyz"
    y_val = float(arr[1])
    y_max = float(pick_workspace_y_max_m())
    if y_val <= y_max:
        return None
    return (
        f"candidate_pick_space_rejected y={y_val:.3f} "
        f"max={y_max:.3f} workspace=pick_only"
    )


def _filter_pick_candidates_by_base_y(
    candidates: list[dict],
    arm: Arm,
    per: Perception,
    depth_frame,
) -> tuple[list[dict], int]:
    if not PICK_FILTER_BY_BASE_Y or not candidates:
        return candidates, 0
    filtered: list[dict] = []
    rejected_count = 0
    for candidate in candidates:
        est_xyz = estimate_base_xyz_from_uv_fast(
            arm=arm,
            per=per,
            depth_frame=depth_frame,
            u=int(candidate["u"]),
            v=int(candidate["v"]),
        )
        if xyz_in_pick_workspace(est_xyz):
            filtered.append(candidate)
        else:
            rejected_count += 1
    return filtered, rejected_count


def _candidate_track_id_or_none(candidate: dict) -> int | None:
    raw_tid = candidate.get("track_id", None)
    if raw_tid is None:
        return None
    try:
        return int(raw_tid)
    except (TypeError, ValueError):
        return None


def _candidate_bbox_xyxy_or_none(candidate: dict) -> tuple[float, float, float, float] | None:
    raw = candidate.get("bbox_xyxy", None)
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        x1 = float(raw[0])
        y1 = float(raw[1])
        x2 = float(raw[2])
        y2 = float(raw[3])
    except (TypeError, ValueError):
        return None
    if not np.isfinite([x1, y1, x2, y2]).all():
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_x_overlap_ratio(a_xyxy: tuple[float, float, float, float], b_xyxy: tuple[float, float, float, float]) -> float:
    ax1, _ay1, ax2, _ay2 = a_xyxy
    bx1, _by1, bx2, _by2 = b_xyxy
    aw = max(1.0, float(ax2 - ax1))
    bw = max(1.0, float(bx2 - bx1))
    overlap_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    return float(overlap_w / max(1.0, min(aw, bw)))


def select_pick_candidate_stack_top(
    candidates: list[dict],
    cx: int,
    cy: int,
    *,
    min_conf: float = DETECT_CONF,
    require_track_id: bool = False,
    blocked_track_ids: set[int] | None = None,
    blocked_uvs: list[list[int]] | None = None,
    top_exposed_only: bool = PICK_TOP_EXPOSED_ONLY,
    x_overlap_min: float = PICK_TOP_EXPOSED_X_OVERLAP_MIN,
    y_gap_px: float = PICK_TOP_EXPOSED_Y_GAP_PX,
    fallback_mode: str = PICK_TOP_EXPOSED_FALLBACK,
) -> tuple[dict | None, dict]:
    blocked_ids = set() if blocked_track_ids is None else {int(tid) for tid in blocked_track_ids}
    blocked_uvs_norm: list[tuple[int, int]] = []
    if blocked_uvs:
        for uv in blocked_uvs:
            if not isinstance(uv, (list, tuple)) or len(uv) < 2:
                continue
            try:
                blocked_uvs_norm.append((int(uv[0]), int(uv[1])))
            except (TypeError, ValueError):
                continue

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

    rows: list[dict] = []
    conf_considered = 0
    for c in candidates:
        conf = float(c.get("conf", 0.0))
        if conf < float(min_conf):
            continue
        conf_considered += 1
        u = int(c.get("u", 0))
        v = int(c.get("v", 0))
        tid = _candidate_track_id_or_none(c)
        if bool(require_track_id) and tid is None:
            continue
        if tid is not None and int(tid) in blocked_ids:
            continue
        if _is_uv_blocked(u, v):
            continue
        row = dict(c)
        row["u"] = int(u)
        row["v"] = int(v)
        row["conf"] = float(conf)
        row["d2_px"] = float((int(u) - int(cx)) ** 2 + (int(v) - int(cy)) ** 2)
        if tid is not None:
            row["track_id"] = int(tid)
        bbox = _candidate_bbox_xyxy_or_none(row)
        if bbox is not None:
            row["_bbox_xyxy"] = bbox
            row["_bbox_y1"] = float(bbox[1])
        rows.append(row)

    diag = {
        "selector_mode": "none",
        "candidate_count": int(len(candidates)),
        "conf_considered": int(conf_considered),
        "eligible_count": int(len(rows)),
        "exposed_top_count": 0,
        "fallback_used": False,
        "selected_track_id": None,
        "selected_uv": None,
    }
    if not rows:
        return None, diag

    fallback_used = False
    selector_mode = "closest"
    pick_pool = list(rows)

    if bool(top_exposed_only):
        bbox_rows = [row for row in rows if row.get("_bbox_xyxy", None) is not None]
        if bbox_rows:
            n = len(bbox_rows)
            parent = list(range(n))

            def _find(i: int) -> int:
                root = i
                while parent[root] != root:
                    root = parent[root]
                while parent[i] != i:
                    nxt = parent[i]
                    parent[i] = root
                    i = nxt
                return root

            def _union(i: int, j: int) -> None:
                ri = _find(i)
                rj = _find(j)
                if ri != rj:
                    parent[rj] = ri

            x_overlap_thr = max(0.0, min(1.0, float(x_overlap_min)))
            for i in range(n):
                ai = bbox_rows[i].get("_bbox_xyxy", None)
                if ai is None:
                    continue
                for j in range(i + 1, n):
                    aj = bbox_rows[j].get("_bbox_xyxy", None)
                    if aj is None:
                        continue
                    if _bbox_x_overlap_ratio(ai, aj) >= x_overlap_thr:
                        _union(i, j)

            comp: dict[int, list[dict]] = {}
            for idx, row in enumerate(bbox_rows):
                root = _find(idx)
                comp.setdefault(root, []).append(row)

            top_rows: list[dict] = []
            y_gap = max(0.0, float(y_gap_px))
            for group in comp.values():
                group_sorted = sorted(
                    group,
                    key=lambda r: (
                        float(r.get("_bbox_y1", float(r.get("v", 0)))),
                        float(r.get("d2_px", float("inf"))),
                        -float(r.get("conf", 0.0)),
                    ),
                )
                top = group_sorted[0]
                if y_gap > 0.0:
                    top_y = float(top.get("_bbox_y1", float(top.get("v", 0))))
                    for alt in group_sorted[1:]:
                        alt_y = float(alt.get("_bbox_y1", float(alt.get("v", 0))))
                        if (alt_y - top_y) <= y_gap and float(alt.get("d2_px", float("inf"))) < float(
                            top.get("d2_px", float("inf"))
                        ):
                            top = alt
                top_rows.append(top)
            if top_rows:
                pick_pool = top_rows
                selector_mode = "top_exposed_by_overlap"
            else:
                fallback_used = True
                selector_mode = f"fallback_{fallback_mode}"
        else:
            fallback_used = True
            selector_mode = f"fallback_{fallback_mode}_no_bbox"

    if selector_mode.startswith("fallback_top"):
        pick_pool.sort(key=lambda r: (int(r.get("v", 0)), float(r.get("d2_px", float("inf"))), -float(r.get("conf", 0.0))))
    else:
        pick_pool.sort(key=lambda r: (float(r.get("d2_px", float("inf"))), -float(r.get("conf", 0.0)), int(r.get("v", 0))))

    selected = pick_pool[0] if pick_pool else None
    diag["selector_mode"] = str(selector_mode)
    diag["fallback_used"] = bool(fallback_used)
    diag["exposed_top_count"] = int(len(pick_pool))
    if selected is not None:
        sel_tid = _candidate_track_id_or_none(selected)
        diag["selected_track_id"] = (None if sel_tid is None else int(sel_tid))
        diag["selected_uv"] = [int(selected.get("u", 0)), int(selected.get("v", 0))]
    return selected, diag


# ============================= Pick / grasp logic =============================
@dataclass(frozen=True)
class GraspPlan:
    measured_xyz: np.ndarray
    pre_pose: np.ndarray
    target_pose: np.ndarray
    lift_pose: np.ndarray
    grip_cmd: float
    measured_reach: float
    command_reach: float
    z_pick: float
    dynamic_stack_forward_x: float


def _clamp_grasp_z_pick_fraction(frac: float) -> float:
    frac_min = float(GRASP_Z_PICK_FRAC_MIN)
    frac_max = float(GRASP_Z_PICK_FRAC_MAX)
    return float(max(frac_min, min(frac_max, float(frac))))


def _compute_pick_z_depth_from_top_m(
    measured_z: float,
    top_cube_frac: float,
    cube_edge_m: float,
) -> tuple[float, float]:
    """
    Depth below measured top face for grasp Z.

    depth = min((1-frac)*edge + optional_flat, 0.5*edge)
    z_pick = max(measured_z - depth, TABLE_Z_SAT_M)

    frac=0.50 targets the vertical midpoint of the detected top cube.
    """
    frac = _clamp_grasp_z_pick_fraction(float(top_cube_frac))
    edge = max(0.01, float(cube_edge_m))
    depth_flat = max(0.0, float(GRASP_Z_DEPTH_FROM_TOP_M))
    depth_frac = max(0.0, (1.0 - frac) * edge)
    depth_center_cap = 0.5 * edge
    depth_m = min(depth_frac + depth_flat, depth_center_cap)
    z_est = float(measured_z) - float(depth_m)
    z_pick = float(max(z_est, TABLE_Z_SAT_M))
    return z_pick, float(depth_m)


def _compute_pick_z_from_top_cube(
    measured_z: float,
    top_cube_frac: float,
    cube_edge_m: float,
) -> float:
    z_pick, _depth_m = _compute_pick_z_depth_from_top_m(
        measured_z,
        top_cube_frac,
        cube_edge_m,
    )
    return float(z_pick)


def _compute_dynamic_stack_forward_x_bias(measured_z: float) -> float:
    if not bool(GRASP_STACK_FORWARD_ENABLE):
        return 0.0
    try:
        z_val = float(measured_z)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(z_val):
        return 0.0
    z_start = float(max(TABLE_Z_SAT_M, GRASP_STACK_FORWARD_Z_START_M))
    max_extra = float(max(0.0, GRASP_STACK_FORWARD_MAX_M))
    if float(z_val) <= float(z_start) + 1e-6:
        return 0.0
    per_level = float(GRASP_STACK_FORWARD_PER_LEVEL_M)
    if per_level > 0.0:
        dz_step = float(max(1e-6, STACK_LEVEL_DZ_M))
        level = int(math.floor((float(z_val) - float(z_start)) / dz_step))
        level = max(0, min(int(level), max(0, int(GRASP_STACK_FORWARD_MAX_LEVELS))))
        return float(min(max_extra, float(level) * per_level))
    dz = float(max(0.0, z_val - z_start))
    gain = float(max(0.0, GRASP_STACK_FORWARD_GAIN_X_PER_M))
    return float(min(max_extra, dz * gain))


def _compute_grasp_far_xy_z_lift(x_m: float, y_m: float) -> tuple[float, float]:
    if not bool(GRASP_FAR_XY_Z_LIFT_ENABLED):
        return 0.0, float("nan")
    try:
        xy_reach_m = math.hypot(float(x_m), float(y_m))
    except (TypeError, ValueError):
        return 0.0, float("nan")
    if not np.isfinite(xy_reach_m):
        return 0.0, float("nan")
    threshold_m = max(0.0, float(GRASP_FAR_XY_Z_LIFT_REACH_M))
    if float(xy_reach_m) < threshold_m:
        return 0.0, float(xy_reach_m)
    return max(0.0, float(GRASP_FAR_XY_Z_LIFT_M)), float(xy_reach_m)


def _build_grasp_plan(
    arm: Arm,
    per: Perception,
    cx: int,
    cy: int,
    grip_default: float,
    safe_pick_reach_m: float | None,
    extra_x_offset_m: float = 0.0,
    extra_y_offset_m: float = 0.0,
    extra_z_offset_m: float = 0.0,
) -> tuple[GraspPlan | None, str]:
    print("[Grasp 1/5] Measuring depth at centered target...")
    x, y, z = measure_base_point_from_uv(arm, per, cx, cy, n=N_MEAS)
    if not np.all(np.isfinite([x, y, z])):
        print("Depth invalid near center - grasp aborted.")
        return None, "depth_invalid"
    r_meas = math.sqrt(x * x + y * y + z * z)
    z_pick_frac = _clamp_grasp_z_pick_fraction(float(GRASP_Z_PICK_FRACTION))
    z_pick, z_depth_from_top_m = _compute_pick_z_depth_from_top_m(
        measured_z=float(z),
        top_cube_frac=float(z_pick_frac),
        cube_edge_m=float(GRASP_CUBE_EDGE_M),
    )
    far_xy_z_lift_m, pick_xy_reach_m = _compute_grasp_far_xy_z_lift(float(x), float(y))
    z_pick = float(z_pick) + float(far_xy_z_lift_m)
    dynamic_stack_forward_x = _compute_dynamic_stack_forward_x_bias(float(z))
    extra_x_offset = float(extra_x_offset_m)
    extra_y_offset = float(extra_y_offset_m)
    extra_z_offset = float(extra_z_offset_m)
    y_cmd = y + float(GRASP_PICK_Y_BIAS_M) + float(extra_y_offset)
    x_pick = x + float(GRASP_PICK_X_BIAS_M) + float(dynamic_stack_forward_x) + float(extra_x_offset)
    z_pick = float(z_pick) + float(extra_z_offset)
    r_cmd = math.sqrt(x_pick * x_pick + y_cmd * y_cmd + z_pick * z_pick)
    safe_reach = MAX_REACH_M if safe_pick_reach_m is None else min(MAX_REACH_M, float(safe_pick_reach_m))
    print(f"Measured object at BASE frame: ({x:.3f}, {y:.3f}, {z:.3f}) m")
    if r_meas > MAX_REACH_M:
        print(f"Object out of reach (distance={r_meas:.3f} m > {MAX_REACH_M:.2f} m).")
        return None, "measured_out_of_reach"
    if r_cmd > MAX_REACH_M:
        print(f"Commanded pose out of reach (distance={r_cmd:.3f} m > {MAX_REACH_M:.2f} m).")
        return None, "commanded_pose_out_of_reach"
    if r_meas > safe_reach or r_cmd > safe_reach:
        print(f"Pick rejected by safety reach cap (meas={r_meas:.3f} m, cmd={r_cmd:.3f} m, cap={safe_reach:.3f} m).")
        return None, "pick_reach_torque_risk"
    grip_cmd = clamp_grip_cmd(grip_default)
    approach_grip = clamp_grip_cmd(GRASP_APPROACH_GRIP)
    print("OK Safe to grasp:")
    print(f"  - Measured reach: {r_meas:.3f} m <= {MAX_REACH_M:.2f} m")
    print(f"  - Command reach: {r_cmd:.3f} m <= {MAX_REACH_M:.2f} m")
    print(f"  - Z saturated: {z_pick:.3f} m (>= {TABLE_Z_SAT_M:.3f} m)")
    print(
        f"  - Z pick model: z_meas - depth_from_top = {z_pick:.3f} "
        f"(depth={float(z_depth_from_top_m):.3f} m, frac={float(z_pick_frac):.2f}, "
        f"(1-frac)*edge={(1.0 - z_pick_frac) * float(GRASP_CUBE_EDGE_M):.3f}, "
        f"flat_add={float(GRASP_Z_DEPTH_FROM_TOP_M):.3f}, "
        f"center_cap={0.5 * float(GRASP_CUBE_EDGE_M):.3f}, edge={float(GRASP_CUBE_EDGE_M):.3f} m, "
        f"far_xy_lift={float(far_xy_z_lift_m):+.4f} "
        f"extra_z={float(extra_z_offset):+.4f} "
        f"@xy_reach={float(pick_xy_reach_m):.3f}/{float(GRASP_FAR_XY_Z_LIFT_REACH_M):.3f})"
    )
    print(
        f"  - X command: cmd_x={x_pick:.3f} "
        f"(base_bias={float(GRASP_PICK_X_BIAS_M):+.3f}, "
        f"stack_fwd={float(dynamic_stack_forward_x):+.3f}, "
        f"extra_x={float(extra_x_offset):+.3f}, "
        f"z_start={float(max(TABLE_Z_SAT_M, GRASP_STACK_FORWARD_Z_START_M)):.3f})"
    )
    print(
        f"  - Y command: cmd_y={y_cmd:.3f} "
        f"(bias={float(GRASP_PICK_Y_BIAS_M):+.3f}, extra_y={float(extra_y_offset):+.3f})"
    )
    print(f"  - Approach grip: {approach_grip:.3f}")
    print(f"  - Gripper clamp: {grip_cmd:.3f} (max {MAX_GRIP_CMD:.2f})")
    return (
        GraspPlan(
            measured_xyz=np.array([x, y, z], dtype=float),
            pre_pose=np.array([x_pick, y_cmd, z_pick + PREGRASP_LIFT, approach_grip]),
            target_pose=np.array([x_pick, y_cmd, z_pick, approach_grip]),
            lift_pose=np.array([x_pick, y_cmd, z_pick + PREGRASP_LIFT, grip_cmd]),
            grip_cmd=grip_cmd,
            measured_reach=r_meas,
            command_reach=r_cmd,
            z_pick=z_pick,
            dynamic_stack_forward_x=float(dynamic_stack_forward_x),
        ),
        "ok",
    )

def _settle_before_grasp_close(arm: Arm):
    settle_s = max(0.0, float(GRASP_SETTLE_BEFORE_CLOSE_S))
    if settle_s <= 1e-6:
        return
    t_end = time.time() + settle_s
    settle_grip = clamp_grip_cmd(GRASP_APPROACH_GRIP)
    while time.time() < t_end:
        arm.tick_hold(grip=settle_grip)
        time.sleep(max(0.0, arm.sample_time))


def _settle_before_lift(arm: Arm, hold_grip: float) -> None:
    settle_s = max(0.0, float(GRASP_SETTLE_BEFORE_LIFT_S))
    if settle_s <= 1e-6:
        return
    t_end = time.time() + settle_s
    grip_hold = float(clamp_grip_cmd(hold_grip))
    while time.time() < t_end:
        arm.tick_hold(grip=grip_hold)
        time.sleep(max(0.0, arm.sample_time))


def _pre_lift_bleed_grip(
    arm: Arm,
    hold_grip: float,
    *,
    close_peak_a: float | None = None,
) -> tuple[float, dict]:
    """Relax grip toward carry band before a loaded lift to reduce joint stall risk."""
    grip_in = float(clamp_grip_cmd(hold_grip))
    if not bool(GRASP_PRE_LIFT_BLEED_ENABLED):
        return grip_in, {"status": "disabled"}
    peak_a = float(close_peak_a) if close_peak_a is not None else float("nan")
    peak_high = bool(np.isfinite(peak_a) and float(peak_a) >= float(GRASP_PRE_LIFT_BLEED_PEAK_A))
    grip_high = bool(float(grip_in) >= float(GRASP_PRE_LIFT_BLEED_GRIP_TRIGGER) - 1e-6)
    if not (peak_high or grip_high):
        return grip_in, {"status": "skipped", "peak_a": peak_a, "grip_in": grip_in}
    bled_grip, meta = tune_hold_grip_to_current_target(
        arm=arm,
        hold_grip=grip_in,
        target_min_a=float(POST_LIFT_CARRY_TARGET_A),
        target_max_a=float(POST_LIFT_CARRY_MAX_A),
        relax_only=True,
        max_steps_override=int(GRASP_PRE_LIFT_BLEED_MAX_STEPS),
        tune_step_override=float(GRASP_PRE_LIFT_BLEED_TUNE_STEP),
        min_grip_cmd=float(POST_LIFT_CARRY_MIN_GRIP_CMD),
    )
    bled_grip = float(clamp_grip_cmd(bled_grip))
    peak_text = f"{float(peak_a):.3f}A" if np.isfinite(peak_a) else "nan"
    print(
        f"[GraspPreLiftBleed] grip {grip_in:.3f}->{bled_grip:.3f} "
        f"close_peak={peak_text} status={meta.get('status')} steps={meta.get('steps')} "
        f"median={meta.get('median_a')} peak_tune={meta.get('peak_a')}"
    )
    out_meta = dict(meta)
    out_meta["peak_a"] = peak_a
    out_meta["grip_in"] = float(grip_in)
    out_meta["grip_out"] = float(bled_grip)
    return bled_grip, out_meta


def _close_for_grasp(
    arm: Arm,
    plan: GraspPlan,
    *,
    grip_step_override: float | None = None,
) -> tuple[bool, float, str, float]:
    close_limits = GRIP_CURRENT_LIMITS
    if grip_step_override is not None:
        grip_step = float(max(0.0005, min(0.0500, float(grip_step_override))))
        close_limits = replace(GRIP_CURRENT_LIMITS, grip_step=grip_step)
    print(
        f"[Grasp 4/5] Closing gripper with current guard "
        f"(target {plan.grip_cmd:.2f}, step={float(close_limits.grip_step):.4f})..."
    )
    close_result = close_gripper_with_current_guard(
        arm_wrapper=arm,
        target_xyz=plan.target_pose[:3],
        grip_start=float(clamp_grip_cmd(plan.target_pose[3])),
        grip_target=plan.grip_cmd,
        limits=close_limits,
    )
    close_status = str(close_result.get("status", "unknown"))
    hold_grip = clamp_grip_cmd(close_result.get("final_grip", plan.grip_cmd))
    peak_a = float(close_result.get("gripper_current_peak_a", float("nan")))
    last_a = float(close_result.get("gripper_current_last_a", float("nan")))
    first_status = str(close_status)
    first_hold_grip = float(hold_grip)
    first_peak_a = float(peak_a)
    first_last_a = float(last_a)
    print(f"[GripCurrent] status={close_status}, final_grip={hold_grip:.3f}, peak={peak_a:.3f} A, last={last_a:.3f} A")
    if close_status in ("miss", "timeout") and bool(GRIP_SOFT_RETRY_ENABLED):
        near_contact = bool(np.isfinite(peak_a) and float(peak_a) >= float(GRIP_SOFT_RETRY_NEAR_CONTACT_A))
        near_target = bool(float(hold_grip) >= float(clamp_grip_cmd(plan.grip_cmd)) - 0.002)
        if near_contact or near_target:
            retry_start = clamp_grip_cmd(
                max(
                    float(clamp_grip_cmd(plan.target_pose[3])),
                    float(hold_grip) - max(0.0, float(GRIP_SOFT_RETRY_REOPEN_DELTA)),
                )
            )
            retry_limits = GripCurrentLimits(
                grip_detect_a=max(0.01, float(close_limits.grip_detect_a) * max(0.70, float(GRIP_SOFT_RETRY_DETECT_SCALE))),
                grip_miss_max_a=float(close_limits.grip_miss_max_a),
                grip_warn_a=float(close_limits.grip_warn_a),
                grip_hard_a=float(close_limits.grip_hard_a),
                emergency_trip_a=float(close_limits.emergency_trip_a),
                transient_ignore_s=min(float(close_limits.transient_ignore_s), max(0.0, float(GRIP_SOFT_RETRY_TRANSIENT_IGNORE_S))),
                debounce_samples=max(1, int(close_limits.debounce_samples)),
                max_close_s=float(close_limits.max_close_s) + max(0.0, float(GRIP_SOFT_RETRY_MAX_CLOSE_EXTRA_S)),
                final_hold_s=float(close_limits.final_hold_s),
                min_grip=float(close_limits.min_grip),
                max_grip=float(close_limits.max_grip),
                grip_step=max(0.0005, float(close_limits.grip_step) * max(0.50, float(GRIP_SOFT_RETRY_STEP_SCALE))),
                relax_step=float(close_limits.relax_step),
                min_detect_grip_cmd=float(close_limits.min_detect_grip_cmd),
            )
            print(
                "[GripCurrent] soft_retry engaged "
                f"(near_contact={near_contact} near_target={near_target} "
                f"start={retry_start:.3f} detect={retry_limits.grip_detect_a:.3f}A "
                f"step={retry_limits.grip_step:.4f} close_s={retry_limits.max_close_s:.2f})"
            )
            retry_result = close_gripper_with_current_guard(
                arm_wrapper=arm,
                target_xyz=plan.target_pose[:3],
                grip_start=float(retry_start),
                grip_target=plan.grip_cmd,
                limits=retry_limits,
            )
            close_result = retry_result
            close_status = str(retry_result.get("status", "unknown"))
            hold_grip = clamp_grip_cmd(retry_result.get("final_grip", hold_grip))
            peak_a = float(retry_result.get("gripper_current_peak_a", peak_a))
            last_a = float(retry_result.get("gripper_current_last_a", last_a))
            print(
                f"[GripCurrentRetry] status={close_status}, final_grip={hold_grip:.3f}, "
                f"peak={peak_a:.3f} A, last={last_a:.3f} A"
            )
    if close_status in ("miss", "timeout"):
        first_near_contact = bool(
            np.isfinite(first_peak_a) and float(first_peak_a) >= float(GRIP_SOFT_RETRY_NEAR_CONTACT_A)
        )
        if first_near_contact and str(first_status) in {"miss", "timeout"}:
            print(
                "[GripCurrent] accepting first-pass contact evidence despite retry failure "
                f"(first_status={first_status}, first_peak={first_peak_a:.3f}A, "
                f"first_last={first_last_a:.3f}A, retry_status={close_status}, retry_peak={peak_a:.3f}A)"
            )
            return True, float(first_hold_grip), "ok_first_contact_fallback", float(first_peak_a)
    if close_status in ("miss", "timeout"):
        print("FAILED: Current verification indicates no secured grasp; retrying pick.")
        return False, 0.0, f"current_{close_status}", float(peak_a)
    if close_status == "overcurrent_emergency":
        print("FAILED: Emergency overcurrent while closing; retrying pick with safer approach.")
        return False, 0.0, "current_overcurrent_emergency", float(peak_a)
    hold_floor = clamp_grip_cmd(max(0.0, float(GRIP_MIN_SUCCESS_HOLD_CMD)))
    if float(hold_grip) < float(hold_floor):
        hold_grip_before = float(hold_grip)
        hold_grip = float(hold_floor)
        print(
            f"[GripCurrent] hold_floor_applied grip={hold_grip_before:.3f}->{hold_grip:.3f} "
            f"floor={hold_floor:.3f}"
        )
    return True, hold_grip, "ok", float(peak_a)


def _format_grasp_lift_fail_diag(arm: Arm) -> str:
    diag = getattr(arm, "last_motion_diag", None)
    if not isinstance(diag, dict):
        return "diag=none"
    return (
        f"diag label={diag.get('label')} reached={diag.get('reached')} "
        f"max_err_deg={diag.get('max_err_deg')} tol_deg={diag.get('tol_used_deg')} "
        f"motion={diag.get('motion_state')} reason={diag.get('last_motion_reason')}"
    )


def _lift_verify_and_tune_grasp(
    arm: Arm,
    plan: GraspPlan,
    hold_grip: float,
    *,
    close_peak_a: float | None = None,
) -> tuple[bool, float, str]:
    hold_grip = float(clamp_grip_cmd(hold_grip))
    hold_grip, _pre_lift_meta = _pre_lift_bleed_grip(
        arm,
        hold_grip,
        close_peak_a=close_peak_a,
    )
    _settle_before_lift(arm, hold_grip)
    lift_segment_s = float(max(0.8, float(GRASP_LIFT_SEGMENT_S)))
    lift_steps = max(30, int(GRASP_LIFT_STEPS))
    print(
        f"[Grasp 5/5] Lifting object grip={float(hold_grip):.3f} "
        f"segment_s={lift_segment_s:.2f} steps={int(lift_steps)}"
    )
    lift_pose = plan.lift_pose.copy()
    lift_pose[3] = hold_grip
    lift_supervisor = make_motion_supervisor(hold_grip, label="grasp_lift")
    if not arm.goto_task_waypoints_cubic(
        [lift_pose],
        segment_duration=float(lift_segment_s),
        steps_per_segment=int(lift_steps),
        label="lift_object_cubic",
        motion_supervisor=lift_supervisor,
    ):
        if arm.last_motion_reason == "move_overcurrent_unrecoverable":
            print("FAILED: Lift aborted due to unrecoverable move-time overcurrent.")
            return False, float(hold_grip), "move_overcurrent_unrecoverable"
        print(
            f"FAILED: Failed to reach lift pose; aborting grasp. {_format_grasp_lift_fail_diag(arm)}"
        )
        return False, float(hold_grip), "lift_failed"
    verify_ok, verify_med_a, _ = verify_grasp_in_air_current(
        arm,
        hold_grip=hold_grip,
        samples=POST_LIFT_VERIFY_SAMPLES,
        min_hits=POST_LIFT_VERIFY_MIN_HITS,
        min_current_a=POST_LIFT_VERIFY_MIN_CURRENT_A,
    )
    if not verify_ok and POST_LIFT_TUNE_ENABLED:
        tuned_grip, tune_meta = tune_hold_grip_to_current_target(
            arm=arm,
            hold_grip=hold_grip,
            target_min_a=float(POST_LIFT_TUNE_TARGET_A),
        )
        print(
            f"[PostLiftTune] status={tune_meta.get('status')} steps={tune_meta.get('steps')} "
            f"grip {float(hold_grip):.3f}->{float(tuned_grip):.3f} "
            f"median={float(tune_meta.get('median_a', float('nan'))):.3f} A "
            f"peak={float(tune_meta.get('peak_a', float('nan'))):.3f} A"
        )
        hold_grip = float(tuned_grip)
        verify_ok, verify_med_a, _ = verify_grasp_in_air_current(
            arm,
            hold_grip=hold_grip,
            samples=POST_LIFT_VERIFY_SAMPLES,
            min_hits=POST_LIFT_VERIFY_MIN_HITS,
            min_current_a=POST_LIFT_VERIFY_MIN_CURRENT_A,
        )
    if not verify_ok:
        print(f"FAILED: Post-lift verification failed (median current {verify_med_a:.3f} A). Object likely slipped or was never secured.")
        # Keep grip on failure so caller can retreat/home without forced drop.
        return False, float(hold_grip), "post_lift_verify_failed"
    hold_grip_clamped = float(clamp_grip_cmd(hold_grip))
    carry_current_high = bool(np.isfinite(verify_med_a) and float(verify_med_a) > float(POST_LIFT_CARRY_MAX_A))
    carry_grip_trigger = float(max(float(POST_LIFT_CARRY_GRIP_TRIGGER_CMD), float(MAX_GRIP_CMD)))
    carry_grip_high = bool(float(hold_grip_clamped) >= float(carry_grip_trigger) - 1e-6)
    carry_bleed_reasons: list[str] = []
    if carry_current_high:
        carry_bleed_reasons.append("current")
    if carry_grip_high:
        carry_bleed_reasons.append("grip_cmd")
    if (
        bool(POST_LIFT_CARRY_BLEED_ENABLED)
        and bool(verify_ok)
        and carry_bleed_reasons
    ):
        pre_grip = float(hold_grip_clamped)
        pre_med = float(verify_med_a) if np.isfinite(verify_med_a) else float("nan")
        bled_grip, bleed_meta = tune_hold_grip_to_current_target(
            arm=arm,
            hold_grip=hold_grip_clamped,
            target_min_a=float(POST_LIFT_CARRY_TARGET_A),
            target_max_a=float(POST_LIFT_CARRY_MAX_A),
            relax_only=True,
            max_steps_override=int(POST_LIFT_CARRY_MAX_STEPS),
            tune_step_override=float(POST_LIFT_CARRY_TUNE_STEP),
            min_grip_cmd=float(POST_LIFT_CARRY_MIN_GRIP_CMD),
        )
        bled_grip = float(clamp_grip_cmd(bled_grip))
        reverify_ok, reverify_med, _ = verify_grasp_in_air_current(
            arm=arm,
            hold_grip=bled_grip,
            samples=POST_LIFT_VERIFY_SAMPLES,
            min_hits=POST_LIFT_VERIFY_MIN_HITS,
            min_current_a=POST_LIFT_VERIFY_MIN_CURRENT_A,
        )
        trigger_text = "+".join(carry_bleed_reasons)
        if bool(reverify_ok):
            hold_grip = float(bled_grip)
            print(
                f"[CarryBleed] trigger={trigger_text} status={bleed_meta.get('status')} "
                f"steps={bleed_meta.get('steps')} grip {pre_grip:.3f}->{float(hold_grip):.3f} "
                f"median {pre_med:.3f}->{float(reverify_med):.3f} A "
                f"max_grip_cmd={float(MAX_GRIP_CMD):.3f}"
            )
        elif (
            np.isfinite(reverify_med)
            and float(reverify_med) >= float(POST_LIFT_VERIFY_MIN_CURRENT_A)
            and float(bled_grip) < float(pre_grip) - 1e-6
        ):
            hold_grip = float(bled_grip)
            arm.tick_hold(grip=hold_grip)
            print(
                f"[CarryBleed] trigger={trigger_text} partial_accept "
                f"status={bleed_meta.get('status')} steps={bleed_meta.get('steps')} "
                f"grip {pre_grip:.3f}->{float(hold_grip):.3f} "
                f"median {pre_med:.3f}->{float(reverify_med):.3f} A "
                f"(reverify_hits_low min={float(POST_LIFT_VERIFY_MIN_CURRENT_A):.3f}A) "
                f"max_grip_cmd={float(MAX_GRIP_CMD):.3f}"
            )
        else:
            print(
                f"[CarryBleed] trigger={trigger_text} reject reverify_failed "
                f"median={float(reverify_med):.3f}A reverting grip->{pre_grip:.3f} "
                f"max_grip_cmd={float(MAX_GRIP_CMD):.3f}"
            )
            arm.tick_hold(grip=clamp_grip_cmd(pre_grip))
            hold_grip = float(pre_grip)
    return True, hold_grip, "ok"

def safe_grasp(
    arm: Arm,
    per: Perception,
    cx: int,
    cy: int,
    grip_default: float | None = None,
    safe_pick_reach_m: float | None = None,
    grip_step_override: float | None = None,
    extra_x_offset_m: float = 0.0,
    extra_y_offset_m: float = 0.0,
    extra_z_offset_m: float = 0.0,
)-> tuple[bool, float, str, dict]:
    grip_default = GRIP_DEFAULT if grip_default is None else grip_default
    plan, plan_reason = _build_grasp_plan(
        arm=arm,
        per=per,
        cx=cx,
        cy=cy,
        grip_default=grip_default,
        safe_pick_reach_m=safe_pick_reach_m,
        extra_x_offset_m=float(extra_x_offset_m),
        extra_y_offset_m=float(extra_y_offset_m),
        extra_z_offset_m=float(extra_z_offset_m),
    )
    if plan is None:
        return False, 0.0, plan_reason, {}
    grasp_info = {
        "pick_measured_xyz": [float(plan.measured_xyz[0]), float(plan.measured_xyz[1]), float(plan.measured_xyz[2])],
        "pick_target_xyz": [float(plan.target_pose[0]), float(plan.target_pose[1]), float(plan.target_pose[2])],
        "lift_pose_xyzg": [float(plan.lift_pose[0]), float(plan.lift_pose[1]), float(plan.lift_pose[2]), float(plan.lift_pose[3])],
        "z_pick": float(plan.z_pick),
        "dynamic_stack_forward_x_m": float(plan.dynamic_stack_forward_x),
        "extra_x_offset_m": float(extra_x_offset_m),
        "extra_y_offset_m": float(extra_y_offset_m),
        "extra_z_offset_m": float(extra_z_offset_m),
        "measured_reach_m": float(plan.measured_reach),
        "command_reach_m": float(plan.command_reach),
    }
    print("[Grasp 3/5] Approaching and descending to grasp height...")
    if not arm.goto_task_waypoints_cubic(
        [plan.pre_pose, plan.target_pose],
        segment_duration=max(0.6, float(GRASP_APPROACH_SEGMENT_S)),
        steps_per_segment=max(30, int(GRASP_APPROACH_STEPS)),
        label="pre_to_grasp_cubic",
    ):
        print("FAILED: Failed to reach grasp pose via cubic path; aborting grasp.")
        return False, 0.0, "approach_failed", dict(grasp_info)
    _settle_before_grasp_close(arm)
    close_ok, hold_grip, close_reason, close_peak_a = _close_for_grasp(
        arm,
        plan,
        grip_step_override=grip_step_override,
    )
    if not close_ok:
        return False, 0.0, close_reason, dict(grasp_info)
    lift_ok, tuned_hold_grip, lift_reason = _lift_verify_and_tune_grasp(
        arm,
        plan,
        hold_grip,
        close_peak_a=float(close_peak_a),
    )
    if not lift_ok:
        return False, float(tuned_hold_grip), lift_reason, dict(grasp_info)
    return True, tuned_hold_grip, "ok", grasp_info
# ============================= Runtime utilities =============================
def log_stop_reason(reason: str):
    print(f"[Stop] {reason}.")

def log_cycle_header(cycle_count: int, title: str):
    print(f"\n{'-' * 20} {title} {cycle_count} {'-' * 20}")

def log_startup_config(mode: str, safe_slots: list[np.ndarray] | None = None, section_groups: dict[str, list[int]] | None = None):
    cam_offsets = get_cam_offsets()
    print(
        f"[Calibration] CAM_OFF_X_M={cam_offsets['cam_off_x_m']:.5f}, "
        f"CAM_OFF_Y_M={cam_offsets['cam_off_y_m']:.5f}, CAM_OFF_Z_M={cam_offsets['cam_off_z_m']:.5f}"
    )
    print(
        f"[GripTune] hard={GRIP_HARD_A:.3f} A, warn={GRIP_WARN_A:.3f} A, "
        f"emergency={GRIP_EMERGENCY_A:.3f} A, detect={GRIP_DETECT_A:.3f} A, "
        f"min_detect_cmd={GRIP_MIN_DETECT_CMD:.3f}, "
        f"miss_max={GRIP_MISS_MAX_A:.3f} A, debounce={GRIP_DEBOUNCE_SAMPLES}"
    )
    print(
        f"[GripTune] default_grip={clamp_grip_cmd(GRIP_DEFAULT):.3f}, "
        f"max_grip={MAX_GRIP_CMD:.3f}"
    )
    print(
        f"[GripClose] transient_ignore={GRIP_TRANSIENT_IGNORE_S:.2f}s "
        f"max_close={GRIP_MAX_CLOSE_S:.2f}s step={GRIP_STEP:.4f} "
        f"relax={GRIP_RELAX_STEP:.4f} "
        f"warn_relax={GRIP_WARN_RELAX_ENABLED}:{GRIP_WARN_RELAX_STEP:.4f}/{int(max(1, GRIP_WARN_RELAX_DEBOUNCE))} "
        f"final_hold={GRIP_FINAL_HOLD_S:.2f}s"
    )
    print(
        f"[GripRetry] enabled={GRIP_SOFT_RETRY_ENABLED} "
        f"near_contact={GRIP_SOFT_RETRY_NEAR_CONTACT_A:.3f}A "
        f"reopen_delta={GRIP_SOFT_RETRY_REOPEN_DELTA:.3f} "
        f"extra_close_s={GRIP_SOFT_RETRY_MAX_CLOSE_EXTRA_S:.2f} "
        f"step_scale={GRIP_SOFT_RETRY_STEP_SCALE:.2f} "
        f"detect_scale={GRIP_SOFT_RETRY_DETECT_SCALE:.2f}"
    )
    print(
        f"[GraspTune] approach_grip={clamp_grip_cmd(GRASP_APPROACH_GRIP):.3f} "
        f"z_pick_frac={_clamp_grasp_z_pick_fraction(float(GRASP_Z_PICK_FRACTION)):.2f} "
        f"z_flat_add={float(GRASP_Z_DEPTH_FROM_TOP_M):.3f} m "
        f"far_xy_lift={bool(GRASP_FAR_XY_Z_LIFT_ENABLED)}:"
        f"{float(GRASP_FAR_XY_Z_LIFT_M):.4f}@{float(GRASP_FAR_XY_Z_LIFT_REACH_M):.3f}m "
        f"cube_edge={float(GRASP_CUBE_EDGE_M):.3f} m "
        f"pick_x_bias={float(GRASP_PICK_X_BIAS_M):+.3f} m "
        f"pick_y_bias={float(GRASP_PICK_Y_BIAS_M):+.3f} m "
        f"misplaced_extra=({float(PICK_MISPLACED_GRASP_X_OFFSET_M):+.3f},"
        f"{float(PICK_MISPLACED_GRASP_Y_OFFSET_M):+.3f},"
        f"{float(PICK_MISPLACED_GRASP_Z_OFFSET_M):+.3f}) m "
        f"misplaced_y_per_level={float(PICK_MISPLACED_GRASP_Y_PER_LEVEL_M):+.3f} "
        f"cap={float(PICK_MISPLACED_GRASP_Y_MAX_M):+.3f}"
    )
    print(
        f"[GraspTune] stack_fwd={GRASP_STACK_FORWARD_ENABLE} "
        f"z_start={float(GRASP_STACK_FORWARD_Z_START_M):.3f} m "
        f"per_level={float(GRASP_STACK_FORWARD_PER_LEVEL_M):+.4f} m "
        f"max_levels={int(GRASP_STACK_FORWARD_MAX_LEVELS)} "
        f"cap={float(GRASP_STACK_FORWARD_MAX_M):.3f} m "
        f"(step={float(STACK_LEVEL_DZ_M):.3f} m)"
    )
    print(
        f"[GraspSettle] joint_tol_default={float(os.getenv('QARM_JOINT_TOL_DEG', '2.2')):.2f} deg "
        f"lift_joint_tol={float(GRASP_LIFT_JOINT_TOL_DEG):.2f} deg"
    )
    print(
        f"[GraspLift] segment_s={float(GRASP_LIFT_SEGMENT_S):.2f} steps={int(GRASP_LIFT_STEPS)} "
        f"settle_before_lift_s={float(GRASP_SETTLE_BEFORE_LIFT_S):.2f} | "
        f"pre_lift_bleed={bool(GRASP_PRE_LIFT_BLEED_ENABLED)} "
        f"grip_trigger>={float(GRASP_PRE_LIFT_BLEED_GRIP_TRIGGER):.2f} "
        f"peak_trigger>={float(GRASP_PRE_LIFT_BLEED_PEAK_A):.2f}A "
        f"step={float(GRASP_PRE_LIFT_BLEED_TUNE_STEP):.4f} "
        f"max_steps={int(GRASP_PRE_LIFT_BLEED_MAX_STEPS)}"
    )
    print(
        f"[PushCfg] target=({PUSH_TARGET_X_M:.3f},{PUSH_TARGET_Y_M:.3f},{PUSH_TARGET_Z_M:.3f}) "
        f"grip_ratio={PUSH_GRIP_RATIO:.2f} approach_offset={PUSH_APPROACH_OFFSET_M:.3f} "
        f"step={PUSH_STEP_M:.3f} max_steps={PUSH_MAX_STEPS} "
        f"tol_xy={PUSH_FINAL_XY_TOL_M:.3f} z_clearance={PUSH_Z_CLEARANCE_M:.3f} "
        f"corridor_half_width={PUSH_CORRIDOR_HALF_WIDTH_M:.3f}"
    )
    if GRIP_HARD_SWEEP_HINT_A:
        print(f"[GripTune] sweep_hint={GRIP_HARD_SWEEP_HINT_A}")
    print(
        f"[CurrentGuard] enabled={MOTION_GUARD_ENABLED}, "
        f"grip(w/h/e)=({MOTION_GRIP_WARN_A:.2f}/{MOTION_GRIP_HARD_A:.2f}/{MOTION_GRIP_EMERGENCY_A:.2f}) A, "
        f"total(w/h/e)=({MOTION_TOTAL_WARN_A:.2f}/{MOTION_TOTAL_HARD_A:.2f}/{MOTION_TOTAL_EMERGENCY_A:.2f}) A, "
        f"freeze_timeout={MOTION_FREEZE_TIMEOUT_S:.2f}s "
        f"relax_step={MOTION_RELAX_STEP:.4f} top_mult={MOTION_RELAX_TOP_MULT:.2f} "
        f"warn_relax={MOTION_WARN_RELAX_ENABLED}:{MOTION_WARN_RELAX_STEP:.4f}/{int(max(1, MOTION_WARN_RELAX_DEBOUNCE))}"
    )
    print(
        f"[PostLiftTune] enabled={POST_LIFT_TUNE_ENABLED} "
        f"target={POST_LIFT_TUNE_TARGET_A:.3f}A step={POST_LIFT_TUNE_STEP:.4f} "
        f"samples={POST_LIFT_TUNE_SAMPLES} max_steps={POST_LIFT_TUNE_MAX_STEPS}"
    )
    print(
        f"[CarryBleed] enabled={POST_LIFT_CARRY_BLEED_ENABLED} "
        f"target={POST_LIFT_CARRY_TARGET_A:.3f}A max={POST_LIFT_CARRY_MAX_A:.3f}A "
        f"grip_trigger>={POST_LIFT_CARRY_GRIP_TRIGGER_CMD:.3f} "
        f"step={POST_LIFT_CARRY_TUNE_STEP:.4f} max_steps={int(POST_LIFT_CARRY_MAX_STEPS)} "
        f"min_grip={POST_LIFT_CARRY_MIN_GRIP_CMD:.3f} relax_only=yes "
        f"max_grip_cmd={MAX_GRIP_CMD:.3f}"
    )
    print(
        f"[PlaceGrid] dx={PLACE_GRID_DX_M:.3f} m, dy={PLACE_GRID_DY_M:.3f} m, "
        f"center=({PLACE_GRID_CENTER_X_M:.3f}, {PLACE_GRID_CENTER_Y_M:.3f}), "
        f"min_sep={MIN_PLACE_SLOT_SEPARATION_M:.3f} m"
    )
    print(
        f"[PlaceTune] place_xy_bias=+0.000,{float(PLACE_Y_BIAS_M):+.3f} m "
        f"cmd_xy_offset={bool(PLACE_CMD_XY_OFFSET_ENABLED)} "
        f"({float(PLACE_CMD_X_OFFSET_M):+.3f},{float(PLACE_CMD_Y_OFFSET_M):+.3f}) "
        f"skip_stack_anchor={bool(PLACE_CMD_OFFSET_SKIP_STACK_ANCHOR)} "
        f"pick_bias_comp={bool(PLACE_PICK_BIAS_COMPENSATE_ENABLED)} "
        f"scale={float(PLACE_PICK_BIAS_COMPENSATE_SCALE):.2f} "
        f"stack_anchor_pick_comp={bool(PLACE_PICK_BIAS_COMPENSATE_STACK_ANCHOR_ENABLED)} "
        f"stack_level_x_offset={bool(STACK_X_LEVEL_OFFSET_ENABLED)} "
        f"(L1={float(STACK_X_LEVEL1_OFFSET):+.3f},L2={float(STACK_X_LEVEL2_OFFSET):+.3f}) "
        f"stack_pick_x_offset={bool(STACK_PICK_X_OFFSET_ENABLED)} "
        f"require_pick_x={bool(STACK_PICK_X_OFFSET_REQUIRE_PICK_X)} "
        f"pick_x_range=({float(STACK_PICK_X_NEAR_M):.3f},{float(STACK_PICK_X_FAR_M):.3f}) "
        f"pick_x_dx=({float(STACK_PICK_X_OFFSET_NEAR_M):+.3f},{float(STACK_PICK_X_OFFSET_FAR_M):+.3f}) "
        f"(delta=-pick_bias) z_unchanged=yes px_tol={int(PX_TOL)}"
    )
    print(
        f"[PlaceMotion] touch_open={clamp_grip_cmd(PLACE_RELEASE_TOUCH_OPEN_GRIP):.3f}, "
        f"post_open={clamp_grip_cmd(PLACE_RELEASE_OPEN_GRIP):.3f}, release_z={PLACE_RELEASE_Z_M:.3f}, "
        f"release_clearance={PLACE_RELEASE_CLEARANCE_M:.3f}, near_offset={PLACE_NEAR_DESCENT_OFFSET_M:.3f}"
    )
    print(
        f"[PlaceSettle] release_joint_tol={PLACE_RELEASE_JOINT_TOL_DEG:.2f} deg "
        f"release_settle={PLACE_RELEASE_SETTLE_S:.2f}s continue_reasons={sorted(list(PLACE_FAIL_CONTINUE_REASONS))}"
    )
    print(
        f"[PlaceTransit] stack_start_level={PLACE_TRANSIT_STACK_START_LEVEL} | "
        f"dz_per_level={PLACE_TRANSIT_STACK_DZ_M:.3f} m | "
        f"place_stack_dz={PLACE_STACK_LEVEL_DZ_M:.3f} m | "
        f"upper_extra_z={PLACE_STACK_UPPER_EXTRA_Z_M:.3f} m | "
        f"level3_extra_z={PLACE_STACK_LEVEL3_EXTRA_Z_M:.3f} m | "
        f"max_extra={PLACE_TRANSIT_STACK_MAX_EXTRA_M:.3f} m"
    )
    if PICK_FILTER_BY_BASE_Y:
        print(f"[PickFilter] base_y <= {PICK_MAX_BASE_Y_M:.3f} m")
    print(
        f"[Tracking] enabled={TRACK_ENABLE} match_xy={TRACK_MATCH_XY_M:.3f} m "
        f"max_miss={TRACK_MAX_MISS_FRAMES} min_conf={TRACK_MIN_CONF:.2f} "
        f"pick_top={TRACK_PICK_PREFER_TOP} strict_top={TRACK_PICK_TOP_STRICT} tie_z={TRACK_PICK_TOP_TIE_Z_M:.3f} "
        f"exposed_only={PICK_TOP_EXPOSED_ONLY} x_overlap_min={PICK_TOP_EXPOSED_X_OVERLAP_MIN:.2f} "
        f"y_gap_px={PICK_TOP_EXPOSED_Y_GAP_PX:.1f} fallback={PICK_TOP_EXPOSED_FALLBACK} "
        f"tracker={YOLO_TRACKER} (hard_locked) persist={YOLO_TRACK_PERSIST} "
        f"ui_mode={UI_MODE} draw_all_boxes={UI_DRAW_ALL_BOXES} "
        f"show_track_ids={UI_SHOW_TRACK_IDS} show_all_ids_minimal={UI_SHOW_ALL_TRACK_IDS_MINIMAL}"
    )
    print(
        f"[DetectProfile] iou_nms={YOLO_IOU_NMS:.2f} "
        f"bbox_split_enabled={YOLO_BBOX_SPLIT_ENABLED} "
        f"bbox_split_aspect>={YOLO_BBOX_SPLIT_MIN_ASPECT:.2f} "
        f"bbox_split_height_m>={YOLO_BBOX_SPLIT_MIN_HEIGHT_M:.3f} "
        f"stretch_square={YOLO_STRETCH_SQUARE} stretch_size={int(YOLO_STRETCH_SIZE)}"
    )
    section_centers_xy = _verify_section_xy_centers()
    section_centers_xy_dbg = {
        str(name): [round(float(center[0]), 4), round(float(center[1]), 4)]
        for name, center in dict(section_centers_xy).items()
    }
    print(
        f"[SectionMap] label_mirror={bool(SECTION_LABEL_MIRROR)} "
        f"centers_xy={section_centers_xy_dbg}"
    )
    if AUTO_CAPTURE_LOCALIZATION_IMAGES:
        capture_root = Path(LOCALIZATION_CAPTURE_ROOT)
        print(f"[DataCapture] enabled root={capture_root} (raw images -> {capture_root / 'raw'})")
    if safe_slots:
        print(
            f"[PlaceGrid] configured safe slots={len(safe_slots)} "
            f"(requested grid cols x rows {PLACE_GRID_COLS}x{PLACE_GRID_ROWS}={PLACE_GRID_ROWS * PLACE_GRID_COLS})"
        )
        print(
            f"[PlaceGrid] first slot=({safe_slots[0][0]:.3f}, {safe_slots[0][1]:.3f}, {safe_slots[0][2]:.3f}) "
            f"last slot=({safe_slots[-1][0]:.3f}, {safe_slots[-1][1]:.3f}, {safe_slots[-1][2]:.3f})"
        )
    if mode == "prompted" and section_groups is not None:
        print(
            f"[Sections] {SECTION_LEFT_NAME}={section_groups.get(SECTION_LEFT_NAME, [])} | "
            f"{SECTION_RIGHT_NAME}={section_groups.get(SECTION_RIGHT_NAME, [])} | "
            f"stack_enabled={ENABLE_STACK_ACTIONS} | column_only={PROMPTED_COLUMN_ONLY}"
        )
        print(f"[Mode] prompted | backend={LLM_POLICY_BACKEND} | model={LLM_POLICY_MODEL}")
        print(f"[Mode] mission_prompt={MISSION_PROMPT}")
        print(f"[Mode] prompt_file={LLM_POLICY_PROMPT_PATH}")
        print(
            f"[PolicyTune] timeout={LLM_POLICY_TIMEOUT_S:.1f}s | "
            f"fallback=disabled (strict LLM-only) | "
            f"think={LLM_POLICY_THINK} | "
            f"num_predict={int(LLM_POLICY_NUM_PREDICT)} | "
            f"print_raw={POLICY_PRINT_RAW} | "
            f"print_raw_block={POLICY_PRINT_RAW_BLOCK}"
        )
        print(
            f"[VerifyHooks] pick_stability={PICK_STABILITY_RECHECK_ENABLED} "
            f"(thr={PICK_STABILITY_RECHECK_MIN_CURRENT_A:.3f} A) | "
            f"place_verify_v2={PLACE_VERIFY_V2_ENABLED} "
            f"(pre/post={PLACE_VERIFY_V2_SAMPLES_PRE}/{PLACE_VERIFY_V2_SAMPLES_POST}, "
            f"r={PLACE_VERIFY_V2_RADIUS_M:.3f}, xy={PLACE_VERIFY_V2_XY_MARGIN_M:.3f}, "
            f"z={PLACE_VERIFY_V2_Z_MARGIN_M:.3f}, "
            f"mismatch_relax_xy={PLACE_VERIFY_V2_MISMATCH_RELAX_XY_M:.3f}, "
            f"mismatch_relax_z={PLACE_VERIFY_V2_MISMATCH_RELAX_Z_M:.3f}, "
            f"x_ref_offset={PLACE_VERIFY_V2_EXPECTED_X_OFFSET_M:.3f}, "
            f"y_ref_offset={PLACE_VERIFY_V2_EXPECTED_Y_OFFSET_M:.3f}, "
            f"z_ref_offset={PLACE_VERIFY_V2_EXPECTED_Z_OFFSET_M:.3f}, "
            f"surface_z_offset={PLACE_VERIFY_V2_SURFACE_Z_OFFSET_M:.3f}, "
            f"eval_offsets={PLACE_VERIFY_V2_EXPECTED_EVAL_USE_OFFSETS}, "
            f"scan_xy_offset={bool(SCAN_BASE_XY_OFFSET_ENABLED)} "
            f"({float(SCAN_BASE_X_OFFSET_M):+.3f},{float(SCAN_BASE_Y_OFFSET_M):+.3f}) z_unchanged=yes, "
            f"stack_xy={PLACE_VERIFY_V2_STACK_XY_MARGIN_M:.3f}, "
            f"hits>={PLACE_VERIFY_V2_MIN_HITS}, delta>={PLACE_VERIFY_V2_DELTA_MIN:.2f}, "
            f"slot_scan_first={bool(PLACE_VERIFY_V2_SLOT_SCAN_FIRST)}, "
            f"hydrate_fallback={bool(PLACE_VERIFY_V2_HYDRATE_FALLBACK_ENABLED)}, "
            f"expected_slot_retries={int(PLACE_VERIFY_V2_EXPECTED_SLOT_RETRIES)}, "
            f"top_candidate_checks={int(PLACE_VERIFY_V2_TOP_CANDIDATE_CHECKS)}, "
            f"defer_generic_to_hydrate={bool(PLACE_VERIFY_V2_DEFER_GENERIC_HANDOFF_TO_HYDRATE)}, "
            f"ladder_logs={bool(PLACE_VERIFY_V2_LADDER_LOGS)}, "
            f"always_recenter={PLACE_VERIFY_V2_ALWAYS_RECENTER}, overlap>={PLACE_VERIFY_V2_MIN_OVERLAP:.2f})"
        )
        print(
            f"[VerifyStack] prefer_top={PLACE_VERIFY_V2_STACK_PREFER_TOP} | "
            f"min_layer_frac={PLACE_VERIFY_V2_STACK_MIN_LAYER_FRAC:.2f} "
            f"(stack_step={STACK_LEVEL_DZ_M:.3f} m)"
        )
        print(
            f"[VerifyRecenter] target_mode={PLACE_VERIFY_V2_TARGET_MODE} | "
            f"pixel_only={PLACE_VERIFY_V2_RECENTER_PIXEL_ONLY} | "
            f"pixel_top={PLACE_VERIFY_V2_RECENTER_PIXEL_TOP} | "
            f"max_candidate_tries={PLACE_VERIFY_V2_RECENTER_MAX_CANDIDATE_TRIES} | "
            f"track_smooth_frames={PLACE_VERIFY_V2_RECENTER_TRACK_SMOOTH_FRAMES} | "
            f"color_filter={PLACE_VERIFY_V2_RECENTER_COLOR_FILTER} "
            f"(min_conf={PLACE_VERIFY_V2_RECENTER_COLOR_MIN_CONF:.2f}) | "
            f"disallow_wrong_color={PLACE_VERIFY_V2_RECENTER_DISALLOW_WRONG_COLOR} "
            f"(blacklist_px={PLACE_VERIFY_V2_RECENTER_WRONG_COLOR_BLACKLIST_PX:.1f}, "
            f"preblacklist_top_first={PLACE_VERIFY_V2_RECENTER_PREBLACKLIST_WRONG_COLOR_TOP_FIRST}) | "
            f"disallow_wrong_xy={PLACE_VERIFY_V2_RECENTER_DISALLOW_WRONG_XY} "
            f"(xy_m={PLACE_VERIFY_V2_RECENTER_WRONG_XY_M:.3f}, "
            f"blacklist_px={PLACE_VERIFY_V2_RECENTER_WRONG_XY_BLACKLIST_PX:.1f}) | "
            f"dynamic_blacklist_px={PLACE_VERIFY_V2_RECENTER_DYNAMIC_BLACKLIST_PX:.1f} | "
            f"show_blacklist_overlay={PLACE_VERIFY_V2_RECENTER_SHOW_BLACKLIST_OVERLAY} | "
            f"persist_window={PLACE_VERIFY_V2_RECENTER_PERSIST_WINDOW} (deprecated; verify_v2 owns window lifecycle) | "
            f"lock_pause_ms={PLACE_VERIFY_V2_RECENTER_LOCK_PAUSE_MS} | "
            f"forced_target_frames={PLACE_VERIFY_V2_RECENTER_FORCED_TARGET_FRAMES} | "
            f"weak_recenter_max_passes={PLACE_VERIFY_V2_WEAK_RECENTER_MAX_PASSES} | "
            f"mismatch_recenter={PLACE_VERIFY_V2_RECENTER_ON_MISMATCH} "
            f"(timeout_s={PLACE_VERIFY_V2_MISMATCH_RECENTER_TIMEOUT_S:.2f}) | "
            f"max_rejects={PLACE_VERIFY_V2_MAX_REJECTS} "
            f"(min_session={PLACE_VERIFY_V2_MIN_REJECTS_PER_SESSION}) | "
            f"session_timeout_s={PLACE_VERIFY_V2_HARD_TIMEOUT_S:.1f} | "
            f"disable_stable_gate={PLACE_VERIFY_V2_DISABLE_STABLE_GATE} | "
            f"track_stable_frames={PLACE_VERIFY_V2_TRACK_STABLE_FRAMES} | "
            f"track_jump_px={PLACE_VERIFY_V2_TRACK_MAX_JUMP_PX:.1f} | "
            f"track_shift_pause_s={PLACE_VERIFY_V2_TRACK_SHIFT_PAUSE_S:.2f} | "
            f"no_track_candidate_timeout_s={TRACK_HANDOFF_NO_CANDIDATE_TIMEOUT_S:.1f} | "
            f"section_pixel_gate={PLACE_VERIFY_V2_SECTION_PIXEL_GATE} "
            f"(margin_px={PLACE_VERIFY_V2_SECTION_PIXEL_MARGIN_PX}) | "
            f"avoid_negative_y={PLACE_VERIFY_V2_AVOID_NEGATIVE_Y} "
            f"(min_y={PLACE_VERIFY_V2_MIN_TRACK_Y_M:+.3f})"
        )
        print(
            f"[PickMisplacedHeightCfg] step_m={MISPLACED_PICK_HEIGHT_STEP_M:.3f} | "
            f"tol_m={MISPLACED_PICK_HEIGHT_TOL_M:.3f} | "
            f"top_tol_m={MISPLACED_PICK_TOP_HEIGHT_TOL_M:.3f} | "
            f"require_top_match={bool(MISPLACED_PICK_REQUIRE_TOP_LEVEL_MATCH)} | "
            f"pick_placed_handoff_section_hard_filter={bool(PICK_PLACED_HANDOFF_SECTION_HARD_FILTER)} | "
            f"pick_placed_verify_strikes={int(PICK_PLACED_VERIFY_STRIKES)} | "
            f"correction_retreat_home={bool(CORRECTION_RETREAT_HOME_ENABLED)} | "
            f"correction_drop_transit={bool(CORRECTION_DROP_TRANSIT_ENABLED)} | "
            f"carry_bleed_move_time={bool(POST_LIFT_CARRY_BLEED_MOVE_TIME)} | "
            f"pick_placed_sole_track_retries={int(PICK_PLACED_SOLE_TRACK_RETRIES)} | "
            f"section_max_dist_m={float(MISPLACED_PICK_SECTION_MAX_DIST_M):.3f} | "
            f"pick_placed_lock_timeout_s={float(PICK_PLACED_LOCK_TIMEOUT_S):.1f} "
            f"pick_placed_max_lock_s={float(PICK_PLACED_MAX_TOTAL_LOCK_TIME_S):.1f}"
        )
        print(
            f"[StartupHydrateCfg] discovery_timeout_s={STARTUP_STACK_DISCOVERY_TIMEOUT_S:.1f} | "
            f"no_det_timeout_s={STARTUP_STACK_DISCOVERY_NO_DET_TIMEOUT_S:.1f} | "
            f"lock_frames={STARTUP_STACK_LOCK_FRAMES} lock_timeout_s={STARTUP_STACK_LOCK_TIMEOUT_S:.2f} "
            f"lock_stage_timeout_s={STARTUP_STACK_LOCK_STAGE_TIMEOUT_S:.1f} | "
            f"lock_assign_xy_margin_m={STARTUP_STACK_LOCK_ASSIGN_XY_MARGIN_M:.3f} | "
            f"measure_samples={STARTUP_STACK_MEASURE_SAMPLES} measure_min_hits={STARTUP_STACK_MEASURE_MIN_HITS} "
            f"center_ey_scale={STARTUP_STACK_CENTER_EY_SCALE:.2f} | "
            f"target_min_conf={STARTUP_TARGET_MIN_CONF:.2f} | "
            f"z_predict_enabled={STARTUP_STACK_Z_PREDICT_ENABLED} | "
            f"refresh_pass_enabled={STARTUP_REFRESH_PASS_ENABLED} | "
            f"layer_scan_frames={STARTUP_STACK_LAYER_SCAN_FRAMES} "
            f"layer_vote_min_hits={STARTUP_STACK_LAYER_VOTE_MIN_HITS} "
            f"layer_match_xy={STARTUP_STACK_LAYER_MATCH_XY_M:.3f} "
            f"layer_match_z={STARTUP_STACK_LAYER_MATCH_Z_M:.3f} | "
            f"require_expected_layers={bool(STARTUP_STACK_REQUIRE_EXPECTED_LAYERS)} "
            f"bootstrap_max_passes={int(STARTUP_STACK_BOOTSTRAP_MAX_PASSES)} | "
            f"side_full_rescan={bool(STARTUP_STACK_SIDE_FULL_RESCAN_ENABLED)} "
            f"rescan_min_expected={int(STARTUP_STACK_SIDE_FULL_RESCAN_MIN_EXPECTED)} "
            f"rescan_frames={int(STARTUP_STACK_SIDE_FULL_RESCAN_FRAMES)} | "
            f"exit_when_sides_full={bool(STARTUP_STACK_EXIT_WHEN_SIDES_FULL)}"
        )
        print(
            f"[VerifyCorrection] enabled={STACK_VERIFY_CORRECTION_ENABLED} | "
            f"require_confirmed_for_stack_advance={STACK_VERIFY_REQUIRE_CONFIRMED_FOR_ADVANCE} | "
            f"block_stack_on_unconfirmed={STACK_VERIFY_BLOCK_STACK_ON_UNCONFIRMED} | "
            f"fail_hydrate_refresh={PICK_CORRECTION_FAIL_HYDRATE_REFRESH_ENABLED} | "
            f"allow_downward={STACK_VERIFY_ALLOW_DOWNWARD_CORRECTION} "
            f"(require_stable_remeasure={STACK_VERIFY_DOWNWARD_REQUIRE_STABLE_REMEASURE}) | "
            f"remeasure(attempts={STACK_REMEASURE_MAX_ATTEMPTS}, required={STACK_REMEASURE_REQUIRED_VALID}, "
            f"max_z_spread={STACK_REMEASURE_MAX_Z_SPREAD_M:.3f} m)"
        )

def log_summary(mode: str, state: CycleState, arm: Arm):
    avg_hold = float(np.mean(state.hold_grip_samples)) if state.hold_grip_samples else float("nan")
    freeze_events = int(arm.guard_event_counts.get("freeze", 0))
    recoveries = int(arm.guard_event_counts.get("recover", 0))
    unrecoverable_events = int(arm.guard_event_counts.get("unrecoverable", 0))
    place_success_rate = float(state.placed_count) / max(1, int(state.picked_count))
    reobserve_ratio = float(state.reobserve_requests) / max(1, int(state.policy_step_count))
    recovery_ratio = float(state.invalid_precondition_recoveries) / max(1, int(state.policy_step_count))
    stop_reason_stable = not str(state.stop_reason).startswith("policy_invalid:")
    stack_readiness_ok = (
        place_success_rate >= float(STACK_READINESS_MIN_PLACE_SUCCESS)
        and reobserve_ratio <= float(STACK_READINESS_MAX_REOBSERVE_RATIO)
        and recovery_ratio <= float(STACK_READINESS_MAX_RECOVERY_RATIO)
        and int(state.policy_invalid_count) <= int(STACK_READINESS_MAX_POLICY_INVALID)
        and stop_reason_stable
    )
    print("\n" + "=" * 60)
    if mode == "prompted":
        print(
            f"OK Prompted mission complete! Picked={state.picked_count} | Placed={state.placed_count} | "
            f"stop_reason={state.stop_reason} | freeze_events={freeze_events} | "
            f"recoveries={recoveries} | unrecoverable_events={unrecoverable_events} | avg_hold_grip={avg_hold:.3f}"
        )
        print(
            f"[PromptedStats] steps={state.policy_step_count} | reobserve={state.reobserve_requests} "
            f"({reobserve_ratio:.2f}) max_streak={state.reobserve_max_streak} | "
            f"returned={state.returned_count} | "
            f"auto_recovery_observes={state.auto_recovery_observes} | "
            f"invalid_recoveries={state.invalid_precondition_recoveries} "
            f"({recovery_ratio:.2f}) | policy_invalid={state.policy_invalid_count}"
        )
        print(
            f"[VerifyStats] pick_unstable={state.pick_stability_fail_count} | "
            f"place_confirmed={state.place_verify_confirmed_count} | place_uncertain={state.place_verify_uncertain_count}"
        )
        print(
            f"[TrackStats] missing_id_rows_total={state.track_untracked_detections_total} | "
            f"missing_id_frames={state.track_untracked_frames}"
        )
        print(
            f"[StackReadyGate] {'PASS' if stack_readiness_ok else 'HOLD'} "
            f"(place_success={place_success_rate:.2f} target>={STACK_READINESS_MIN_PLACE_SUCCESS:.2f}, "
            f"reobserve_ratio<={STACK_READINESS_MAX_REOBSERVE_RATIO:.2f}, "
            f"recovery_ratio<={STACK_READINESS_MAX_RECOVERY_RATIO:.2f}, "
            f"policy_invalid<={STACK_READINESS_MAX_POLICY_INVALID}, stop_reason_stable={stop_reason_stable})"
        )
    else:
        print(
            f"OK Mission complete! Picked cubes={state.picked_count} | Placed cubes={state.placed_count} | "
            f"stop_reason={state.stop_reason} | freeze_events={freeze_events} | "
            f"recoveries={recoveries} | unrecoverable_events={unrecoverable_events} | avg_hold_grip={avg_hold:.3f}"
        )
    print("=" * 60)


def finalize_run_home(arm: Arm, state: CycleState, final_label: str):
    if state.skip_final_motion:
        print("[WARN] Skipping final HOME move due to unrecoverable overcurrent condition.")
        return
    final_home = HOME.copy()
    if state.holding_object:
        print("[WARN] Ending while still holding a cube; keeping gripper closed at final HOME.")
        final_home[3] = clamp_grip_cmd(state.current_hold_grip)
    arm.goto_task_space(final_home, duration=1.2, label=final_label)

def shutdown_runtime(per: Perception, arm: Arm):
    cv2.destroyAllWindows()
    per.stop()
    arm.arm.terminate()
    print("Program terminated.")


# ============================= Pick / grasp logic =============================
def execute_push_cube_action(
    *,
    state: CycleState,
    arm: Arm,
    det: YOLODetector,
    per: Perception,
    centered_pos: tuple[int, int] | None,
    label_prefix: str,
) -> tuple[bool, str, dict | None]:
    if bool(state.holding_object):
        return False, "holding_object", None
    if centered_pos is None:
        return False, "push_requires_centered_target", None

    push_target_xyz = np.array(
        [float(PUSH_TARGET_X_M), float(PUSH_TARGET_Y_M), float(PUSH_TARGET_Z_M)],
        dtype=float,
    ).reshape(-1)
    push_target_xyz[2] = max(float(push_target_xyz[2]), float(TABLE_Z_SAT_M) + 0.005)
    push_grip = clamp_grip_cmd(float(MAX_GRIP_CMD) * float(PUSH_GRIP_RATIO))
    push_step_m = float(max(0.005, PUSH_STEP_M))
    push_approach_offset_m = float(max(0.005, PUSH_APPROACH_OFFSET_M))
    push_max_steps = int(max(1, PUSH_MAX_STEPS))
    push_tol_m = float(max(0.010, PUSH_FINAL_XY_TOL_M))
    push_z_clearance_m = float(max(0.004, PUSH_Z_CLEARANCE_M))
    push_corridor_half_width_m = float(max(0.010, PUSH_CORRIDOR_HALF_WIDTH_M))
    pick_scope_y_max = float(PICK_MAX_BASE_Y_M + max(0.015, SCENE_RECON_PLACE_Y_MARGIN_M))

    def _measure_cube_near_xy(
        *,
        ref_xy: tuple[float, float],
        fallback_uv: tuple[int, int] | None,
        timeout_s: float = 1.2,
        max_samples: int = 5,
    ) -> dict:
        rows_xyz: list[np.ndarray] = []
        rows_uv: list[tuple[int, int]] = []
        t0 = float(time.time())
        while (time.time() - t0) < float(max(0.4, timeout_s)) and len(rows_xyz) < int(max(1, max_samples)):
            obs = observe_scene_frame(
                det=det,
                arm=arm,
                per=per,
                draw=False,
                projected_min_conf=float(max(0.0, PLACE_VERIFY_MIN_CONF)),
                state=state,
                update_tracks=True,
            )
            if obs is None:
                break
            best_row = None
            best_score = float("inf")
            for prow in list(obs.projected_rows):
                try:
                    conf = float(prow.get("conf", 0.0))
                except Exception:
                    conf = 0.0
                if conf < float(PLACE_VERIFY_MIN_CONF):
                    continue
                xyz = np.array(prow.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
                if xyz.size < 3 or not np.all(np.isfinite(xyz[:3])):
                    continue
                if float(xyz[1]) > float(pick_scope_y_max):
                    continue
                d_xy = float(math.hypot(float(xyz[0]) - float(ref_xy[0]), float(xyz[1]) - float(ref_xy[1])))
                score = float(d_xy - (0.02 * conf))
                if score < best_score:
                    best_score = score
                    best_row = prow
            if best_row is None and isinstance(fallback_uv, (list, tuple)) and len(fallback_uv) >= 2:
                u_f = int(fallback_uv[0])
                v_f = int(fallback_uv[1])
                proj = _match_projected_row_by_uv(obs.projected_rows, u=u_f, v=v_f)
                if proj is not None:
                    best_row = dict(proj)
                    best_row["u"] = int(u_f)
                    best_row["v"] = int(v_f)
            if best_row is None:
                time.sleep(max(0.0, float(arm.sample_time)))
                continue
            xyz_best = np.array(best_row.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
            if xyz_best.size < 3 or not np.all(np.isfinite(xyz_best[:3])):
                u_fast = int(best_row.get("u", 0))
                v_fast = int(best_row.get("v", 0))
                fast_xyz = estimate_base_xyz_from_uv_fast(
                    arm=arm,
                    per=per,
                    depth_frame=obs.depth_frame,
                    u=u_fast,
                    v=v_fast,
                )
                fast_xyz = np.array(fast_xyz, dtype=float).reshape(-1)
                if fast_xyz.size < 3 or not np.all(np.isfinite(fast_xyz[:3])):
                    time.sleep(max(0.0, float(arm.sample_time)))
                    continue
                xyz_best = fast_xyz
            rows_xyz.append(np.array([float(xyz_best[0]), float(xyz_best[1]), float(xyz_best[2])], dtype=float))
            rows_uv.append((int(best_row.get("u", 0)), int(best_row.get("v", 0))))
            time.sleep(max(0.0, float(arm.sample_time)))

        if not rows_xyz:
            return {"hits": 0, "median_xyz": None, "median_uv": None}
        xyz_med = np.median(np.array(rows_xyz, dtype=float), axis=0)
        uv_med = np.median(np.array(rows_uv, dtype=float), axis=0) if rows_uv else np.array([np.nan, np.nan], dtype=float)
        uv_pair = None
        if uv_med.size >= 2 and np.all(np.isfinite(uv_med[:2])):
            uv_pair = [int(round(float(uv_med[0]))), int(round(float(uv_med[1])))]
        return {
            "hits": int(len(rows_xyz)),
            "median_xyz": [float(xyz_med[0]), float(xyz_med[1]), float(xyz_med[2])],
            "median_uv": uv_pair,
        }

    def _path_blocked_by_other_cube(
        *,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        target_xyz: np.ndarray,
    ) -> tuple[bool, dict | None]:
        obs = observe_scene_frame(
            det=det,
            arm=arm,
            per=per,
            draw=False,
            projected_min_conf=float(max(0.0, PLACE_VERIFY_MIN_CONF)),
            state=state,
            update_tracks=True,
        )
        if obs is None:
            return False, None
        sx, sy = float(start_xy[0]), float(start_xy[1])
        ex, ey = float(end_xy[0]), float(end_xy[1])
        vx, vy = float(ex - sx), float(ey - sy)
        seg_len2 = float((vx * vx) + (vy * vy))
        if seg_len2 <= 1e-9:
            return False, None
        for prow in list(obs.projected_rows):
            xyz = np.array(prow.get("xyz", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
            if xyz.size < 3 or not np.all(np.isfinite(xyz[:3])):
                continue
            if float(xyz[1]) > float(pick_scope_y_max):
                continue
            d_to_target = float(
                math.hypot(float(xyz[0]) - float(target_xyz[0]), float(xyz[1]) - float(target_xyz[1]))
            )
            if d_to_target <= float(max(0.030, push_corridor_half_width_m * 0.60)):
                # likely the pushed cube itself
                continue
            t_unclamped = float(((float(xyz[0]) - sx) * vx + (float(xyz[1]) - sy) * vy) / seg_len2)
            if t_unclamped < -0.08 or t_unclamped > 1.20:
                continue
            t_seg = float(max(0.0, min(1.0, t_unclamped)))
            cx = float(sx + (t_seg * vx))
            cy = float(sy + (t_seg * vy))
            d_perp = float(math.hypot(float(xyz[0]) - cx, float(xyz[1]) - cy))
            d_z = float(abs(float(xyz[2]) - float(target_xyz[2])))
            if d_perp <= float(push_corridor_half_width_m) and d_z <= float(max(0.08, STACK_LEVEL_DZ_M * 1.3)):
                return True, {
                    "blocked_xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                    "blocked_perp_xy_m": float(d_perp),
                    "blocked_t": float(t_unclamped),
                }
        return False, None

    measure0 = collect_track_measurement_verify_style(
        state=state,
        arm=arm,
        per=per,
        det=det,
        track_id=(None if state.active_target_track_id is None else int(state.active_target_track_id)),
        lock_uv=(int(centered_pos[0]), int(centered_pos[1])),
        sample_count=max(3, min(8, int(STARTUP_STACK_MEASURE_SAMPLES))),
        timeout_s=1.2,
        min_conf=float(PLACE_VERIFY_MIN_CONF),
        show_window=False,
        status_prefix="push_measure_start",
    )
    hits0 = int(measure0.get("hits", 0))
    xyz0 = _finite_xyz_or_none(measure0.get("median_xyz", None))
    uv0 = measure0.get("median_uv", None)
    if xyz0 is None or hits0 < 2:
        fallback = _measure_cube_near_xy(
            ref_xy=(float(push_target_xyz[0]), float(push_target_xyz[1])),
            fallback_uv=(int(centered_pos[0]), int(centered_pos[1])),
            timeout_s=1.2,
            max_samples=5,
        )
        xyz0 = _finite_xyz_or_none(fallback.get("median_xyz", None))
        uv0 = fallback.get("median_uv", None)
        hits0 = int(fallback.get("hits", 0))
    if xyz0 is None or hits0 < 2:
        return False, "push_no_target_measurement", {
            "status": "push_no_target_measurement",
            "hits": int(hits0),
            "samples": int(max(3, min(8, int(STARTUP_STACK_MEASURE_SAMPLES)))),
            "target_xyz": _finite_xyz_or_none(push_target_xyz),
        }
    if float(xyz0[1]) > float(pick_scope_y_max):
        return False, "push_out_of_pick_space", {
            "status": "push_out_of_pick_space",
            "start_xyz": list(xyz0),
            "pick_scope_y_max": float(pick_scope_y_max),
        }

    dist_before = float(math.hypot(float(push_target_xyz[0]) - float(xyz0[0]), float(push_target_xyz[1]) - float(xyz0[1])))
    if dist_before <= float(push_tol_m):
        return True, "ok_already_at_target", {
            "status": "ok_already_at_target",
            "start_xyz": list(xyz0),
            "end_xyz": list(xyz0),
            "steps": 0,
            "distance_before_m": float(dist_before),
            "distance_after_m": float(dist_before),
            "target_xyz": _finite_xyz_or_none(push_target_xyz),
            "push_grip": float(push_grip),
        }

    current_xyz = np.array([float(xyz0[0]), float(xyz0[1]), float(xyz0[2])], dtype=float).reshape(-1)
    current_uv = (
        (int(uv0[0]), int(uv0[1]))
        if isinstance(uv0, (list, tuple)) and len(uv0) >= 2
        else (int(centered_pos[0]), int(centered_pos[1]))
    )
    push_supervisor = make_motion_supervisor(float(push_grip), label=f"{label_prefix}_push")
    step_count = 0
    blocked_context = None
    motion_fail_stage = ""

    for step_i in range(int(push_max_steps)):
        dxy_x = float(push_target_xyz[0]) - float(current_xyz[0])
        dxy_y = float(push_target_xyz[1]) - float(current_xyz[1])
        dist_now = float(math.hypot(dxy_x, dxy_y))
        if dist_now <= float(push_tol_m):
            break
        vec_x = float(dxy_x / max(1e-9, dist_now))
        vec_y = float(dxy_y / max(1e-9, dist_now))
        step_dist = float(min(push_step_m, dist_now))
        contact_x = float(current_xyz[0] - (vec_x * push_approach_offset_m))
        contact_y = float(current_xyz[1] - (vec_y * push_approach_offset_m))
        push_x = float(current_xyz[0] + (vec_x * step_dist))
        push_y = float(current_xyz[1] + (vec_y * step_dist))
        push_z = float(max(float(TABLE_Z_SAT_M) + float(push_z_clearance_m), min(float(current_xyz[2]), float(PUSH_TARGET_Z_M))))
        push_z_high = float(push_z + max(0.010, float(push_z_clearance_m)))

        is_blocked, block_ctx = _path_blocked_by_other_cube(
            start_xy=(float(current_xyz[0]), float(current_xyz[1])),
            end_xy=(push_x, push_y),
            target_xyz=current_xyz,
        )
        if is_blocked:
            blocked_context = dict(block_ctx) if isinstance(block_ctx, dict) else None
            return False, "push_path_blocked", {
                "status": "push_path_blocked",
                "start_xyz": _finite_xyz_or_none(current_xyz),
                "end_xyz": _finite_xyz_or_none(current_xyz),
                "steps": int(step_count),
                "distance_before_m": float(dist_before),
                "distance_after_m": float(dist_now),
                "target_xyz": _finite_xyz_or_none(push_target_xyz),
                "blocked": blocked_context,
            }

        step_poses = [
            np.array([contact_x, contact_y, push_z_high, push_grip], dtype=float),
            np.array([contact_x, contact_y, push_z, push_grip], dtype=float),
            np.array([push_x, push_y, push_z, push_grip], dtype=float),
            np.array([push_x, push_y, push_z_high, push_grip], dtype=float),
        ]
        step_labels = ["pre_contact_high", "contact_down", "push_forward", "post_push_lift"]
        for pose_i, pose in enumerate(step_poses):
            if not arm.goto_task_space(
                pose,
                duration=0.55 if pose_i in {1, 2} else 0.45,
                label=f"{label_prefix}_push_{step_labels[pose_i]}_{step_i+1}",
                motion_supervisor=push_supervisor,
            ):
                motion_fail_stage = str(step_labels[pose_i])
                if str(getattr(arm, "last_motion_reason", "")) == "move_overcurrent_unrecoverable":
                    return False, "move_overcurrent_unrecoverable", {
                        "status": "move_overcurrent_unrecoverable",
                        "stage": str(motion_fail_stage),
                        "steps": int(step_count),
                        "distance_before_m": float(dist_before),
                        "target_xyz": _finite_xyz_or_none(push_target_xyz),
                    }
                return False, "push_motion_failed", {
                    "status": "push_motion_failed",
                    "stage": str(motion_fail_stage),
                    "steps": int(step_count),
                    "distance_before_m": float(dist_before),
                    "distance_after_m": float(dist_now),
                    "target_xyz": _finite_xyz_or_none(push_target_xyz),
                    "motion_reason": str(getattr(arm, "last_motion_reason", "") or ""),
                }

        step_count += 1
        m_row = _measure_cube_near_xy(
            ref_xy=(push_x, push_y),
            fallback_uv=current_uv,
            timeout_s=1.0,
            max_samples=4,
        )
        m_xyz = _finite_xyz_or_none(m_row.get("median_xyz", None))
        if m_xyz is None or int(m_row.get("hits", 0)) < 2:
            break
        current_xyz = np.array([float(m_xyz[0]), float(m_xyz[1]), float(m_xyz[2])], dtype=float).reshape(-1)
        m_uv = m_row.get("median_uv", None)
        if isinstance(m_uv, (list, tuple)) and len(m_uv) >= 2:
            current_uv = (int(m_uv[0]), int(m_uv[1]))
        if float(current_xyz[1]) > float(pick_scope_y_max):
            return False, "push_out_of_pick_space", {
                "status": "push_out_of_pick_space",
                "start_xyz": _finite_xyz_or_none(xyz0),
                "end_xyz": _finite_xyz_or_none(current_xyz),
                "steps": int(step_count),
                "pick_scope_y_max": float(pick_scope_y_max),
            }

    dist_after = float(
        math.hypot(float(push_target_xyz[0]) - float(current_xyz[0]), float(push_target_xyz[1]) - float(current_xyz[1]))
    )
    status = "ok" if dist_after <= float(push_tol_m) else "partial_timeout"
    progress_m = float(max(0.0, dist_before - dist_after))
    context = {
        "status": str(status),
        "start_xyz": _finite_xyz_or_none(xyz0),
        "end_xyz": _finite_xyz_or_none(current_xyz),
        "steps": int(step_count),
        "distance_before_m": float(dist_before),
        "distance_after_m": float(dist_after),
        "progress_m": float(progress_m),
        "target_xyz": _finite_xyz_or_none(push_target_xyz),
        "push_grip": float(push_grip),
        "push_tol_m": float(push_tol_m),
        "max_steps": int(push_max_steps),
    }
    if status == "ok":
        return True, "ok", context
    return False, "partial_timeout", context

# ============================= Return / drop-zone logic =============================
def main_prompted():
    """Deprecated compatibility shim.

    Active prompted orchestration lives in runtime_loop.main_prompted().
    Keep this shim so accidental direct runtime_core execution still works,
    but ensure there is a single authoritative prompted loop implementation.
    """
    from runtime_loop import main_prompted as _runtime_loop_main_prompted

    print("[RuntimeCore] main_prompted is deprecated; delegating to runtime_loop.main_prompted")
    return _runtime_loop_main_prompted()

def main():
    main_prompted()

if __name__ == "__main__":
    main()
