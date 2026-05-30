# Code Inventory (Current Runtime)

## Scope and Notes

- Requested files inspected: `runtime_loop.py`, `runtime_core.py`, `current_sensing.py` (not present), `planner/llm_policy.py` (not present).
- Active equivalents discovered:
  - `current_sensing_grip_limiting.py` (imported by `runtime_core.py`)
  - `llm_commander/planner/live_policy_brain.py` (policy engine used by `runtime_core.build_live_policy_brain()`)
- No behavior changes were made.

Legend for recommendation column:
- `stay`: keep primary ownership where it currently is.
- `move`: candidate to relocate in a future refactor.
- `reuse`: centralized logic already exists; callers should reuse it.
- `left alone`: stable enough; no immediate movement/change pressure.

---

## 1) YOLO detection

| File | Function/Class/Constant | Responsibility | Inputs / Outputs | Where Called | Similar Logic Elsewhere? | Recommendation |
|---|---|---|---|---|---|---|
| `vision_runtime.py` / `runtime_core.py` | `YOLO_MODEL_PATH`, `TARGET_CLASSES`, `YOLO_CONF`, `YOLO_TRACKER`, `YOLO_TRACK_PERSIST`, `YOLO_IOU_NMS` | Runtime detection/tracking config; duplicate/overlap behavior is tuned only through Ultralytics NMS IoU | Env/defaults -> module constants | Used by `YOLODetector.__init__`; instantiated in `runtime_loop.main_prompted()` and `runtime_loop.main_tune()` | Detection constants are centralized in `vision_runtime.py` and imported by `runtime_core.py` for runtime wiring/logs | stay |
| `vision_runtime.py` | `class YOLODetector` | Wraps Ultralytics `model.track`, class filtering, target candidate extraction | In: BGR frame (+ tracker config); Out: `img_display`, candidate dicts (`u,v,conf,bbox,track_id`) | Called broadly (`observe_scene_frame`, `extract_projected_detections`, direct classify/return-handoff probes in `runtime_loop`) | Candidate selection logic also appears downstream in centering/handoff functions, but raw detection is centralized here | stay |
| `vision_runtime.py` | `YOLODetector.detect_candidates_and_draw` | Performs tracked detection and builds bbox-based target candidate records | In: `bgr`, `draw`; Out: annotated image + target candidate list | Called by `observe_scene_frame`, `extract_projected_detections`, classify path in `runtime_loop`, and return handoff classify in core | No equivalent full detection pipeline elsewhere | reuse |

---

## 2) candidate filtering

| File | Function/Class/Constant | Responsibility | Inputs / Outputs | Where Called | Similar Logic Elsewhere? | Recommendation |
|---|---|---|---|---|---|---|
| `runtime_core.py` | `_filter_pick_candidates_by_base_y` + `PICK_FILTER_BY_BASE_Y`, `PICK_MAX_BASE_Y_M` | Drops candidates outside pick-side base-Y gate | In: raw candidates + depth/projection context; Out: filtered list + reject count | Called by `center_object_slowly` | Related section gating exists in verify/reconcile flows | reuse |
| `runtime_core.py` | `select_pick_candidate_stack_top` + `PICK_TOP_EXPOSED_*`, `PICK_OTHER_BLOCK_UV_PX` | Chooses top/exposed candidate with overlap grouping, fallback modes, blocked track/UV support | In: candidates + center + gates; Out: selected candidate + selector diagnostics | Called in `center_object_slowly`, `acquire_and_center_intended_cube` | Track-handoff and verify flows have separate candidate-selection rules | stay |
| `runtime_core.py` | `_filter_verify_candidates` + `PLACE_VERIFY_V2_RECENTER_*` | Verify-time candidate filtering by section/color/expected XY proximity | In: raw candidates + projected rows + expected section/color/xyz; Out: filtered candidates + filter stats | Called by `center_object_on_expected_slot` | Similar color/section checks exist in `reconcile_scene`, but purpose differs (verify lock vs scene summary) | reuse |
| `runtime_core.py` | `_filter_projected_slot_candidates` | Geometry filter around expected slot radius / z bounds | In: projected rows + expected xyz + tolerances; Out: sorted ROI rows + projected-valid count | Called by `collect_slot_observations` | Overlaps with `score_place_geometry` stage but not redundant | left alone |
| `runtime_core.py` | `choose_candidate_near_uv`, `choose_track_candidate_near_uv` | Utility selectors for nearest candidate in UV or nearest-with-track-id | In: candidates + UV (+ track id); Out: chosen candidate or `None` | Used by measurement and handoff/classify helpers | Similar nearest-track logic exists in `nearest_visible_track_by_uv` for track memory, not raw candidate lists | reuse |

---

## 3) projection from UV/depth to base XYZ

| File | Function/Class/Constant | Responsibility | Inputs / Outputs | Where Called | Similar Logic Elsewhere? | Recommendation |
|---|---|---|---|---|---|---|
| `runtime_core.py` | `robust_depth_m` | Robust depth sampling from local patch percentile | In: depth frame, `(u,v)`, scale/window; Out: depth meters (`float`/`nan`) | Used by `estimate_base_xyz_from_uv_fast`, `measure_base_point_from_uv` | No duplicate depth sampler in inspected files | stay |
| `runtime_core.py` | `uvz_to_xyz_cam` | Projects pixel + depth into camera-frame XYZ | In: `u,v,Z,intrinsics`; Out: `(X,Y,Z)` camera coords | Used by `estimate_base_xyz_from_uv_fast`, `measure_base_point_from_uv` | N/A in inspected files | reuse |
| `runtime_core.py` | `estimate_base_xyz_from_uv_fast` | Fast single-pass UV/depth -> base XYZ projection using current joint pose | In: arm/per/depth/UV; Out: `np.ndarray([x,y,z])` | Used in projection, verify fallback, classify seed, push/return/pick helpers | `measure_base_point_from_uv` does multi-sample version of same transform chain | reuse |
| `runtime_core.py` | `project_candidates_to_base` | Adds base XYZ to detection candidates | In: candidates + depth/perception; Out: projected row dict list | Used by `observe_scene_frame`, `extract_projected_detections` | N/A | stay |
| `runtime_core.py` | `extract_projected_detections` | Convenience: detect candidates then project immediately | In: detector/arm/per/frame/depth; Out: projected detections | Used by downstream observation/measurement paths | Similar call chain appears in `observe_scene_frame` but with richer output bundle | left alone |
| `runtime_core.py` | `measure_base_point_from_uv` + `N_MEAS`, `MEAS_MEDIAN_WIN` | Multi-sample measured base XYZ with debug diagnostics | In: arm/per/UV/sample count; Out: averaged `(x,y,z)` floats | Used by grasp planning and stack remeasure paths | Duplicates transform math with `estimate_base_xyz_from_uv_fast` by design (precision vs speed modes) | left alone |

---

## 4) track_id usage

| File | Function/Class/Constant | Responsibility | Inputs / Outputs | Where Called | Similar Logic Elsewhere? | Recommendation |
|---|---|---|---|---|---|---|
| `runtime_core.py` | `TRACK_*` constants (`TRACK_ENABLE`, `TRACK_MIN_CONF`, `TRACK_PICK_*`, etc.) | Global track behavior/tuning knobs | Env/defaults -> module constants | Used across track update, selection, handoff, centering | Tracking IDs also flow from YOLO rows directly | stay |
| `runtime_core.py` | `CycleState.track_memory`, `active_target_track_id`, `last_track_snapshot`, warning counters | In-memory authoritative track state for runtime loop | Stateful fields updated/read across cycle | Updated in `update_cube_tracks`; consumed by pick/verify/return flows and planner gating | Some flows also re-query frame candidates without state memory | stay |
| `runtime_core.py` | `update_cube_tracks` | Maintains visible/missing track lifecycle, snapshot, missing-ID warnings | In: projected detections; Out: updated `state.track_memory` + snapshot | Called in `observe_scene_frame(update_tracks=True)` and pick prime path | No alternate track-memory manager found | reuse |
| `runtime_core.py` | `select_intended_track_for_pick` | Chooses next target track (top-first/center fallback) with blocklists | In: state (+ optional blocked ids/xyzs); Out: chosen track id | Used in `acquire_and_center_intended_cube` | `run_track_handoff_session` also chooses locks, but for specialized sessions | stay |
| `runtime_core.py` | `nearest_visible_track_by_uv` | Links UV lock to nearest visible tracked target | In: state + `u,v`; Out: nearest track id or `None` | Used in centering and classify/handoff glue (`runtime_loop` + core verify flows) | Candidate-nearest utilities exist (`choose_*near_uv`) but operate on raw candidate lists | reuse |
| `runtime_core.py` | `run_track_handoff_session` | Generic track lock/reject/accept loop shared by pick_other, verify, return, misplaced flows | In: state, detector/perception, gating callbacks; Out: session dict (`status`, selected track/uv/xyz, rejects) | Called by `run_pick_other_session`, `run_return_verify_stage`, `run_return_handoff_stage`, `execute_pick_misplaced_cube_action`, verify-v2 | Strong overlap with ad hoc centering loops; this is already the reusable core | reuse |

---

## 5) cube/stack identity

| File | Function/Class/Constant | Responsibility | Inputs / Outputs | Where Called | Similar Logic Elsewhere? | Recommendation |
|---|---|---|---|---|---|---|
| `runtime_core.py` | `SECTION_LEFT_NAME`, `SECTION_RIGHT_NAME`, `MAX_STACK_LEVELS_PER_SECTION`, `STACK_LEVEL_DZ_M` | Canonical section naming and stack capacity geometry | Env/defaults -> constants | Used throughout planner-state, hydrate, verify, correction, placement | No competing section-name source found in inspected files | stay |
| `runtime_core.py` | `_startup_default_hydrated_section_row`, `_normalize_hydrated_section_row`, `_merge_hydrated_section_row_keep_known` | Normalizes and merges startup-hydrated stack identity rows | In: row dicts; Out: normalized authoritative rows | Used by startup/hydration pipeline | Ledger-derived identity also exists (`_ledger_section_truth_row`) but this path is startup authority | reuse |
| `runtime_core.py` | `run_startup_stack_identity_pass` | Startup discovery pass to infer per-side stacks/colors/tracks | In: detector/perception/state; Out: hydration summary dict | Called by `runtime_loop.main_prompted` startup bootstrap | `reconcile_scene` also observes stacks, but startup pass does lock/assignment semantics | stay |
| `runtime_core.py` | `apply_startup_stack_hydration`, `get_startup_hydrated_section_row` | Applies startup identity into runtime authoritative state | In: startup row; Out: synced levels + state mutation | Called in `runtime_loop` bootstrap sync | Similar section status assembled for planner via `_planner_section_row_unified` | reuse |
| `runtime_core.py` | `_set_authoritative_section_sequence`, `append_authoritative_stack_cube`, `pop_authoritative_stack_top`, `get_authoritative_stack_levels` | Maintains authoritative stack sequence through place/pick corrections | In: state + section/color/removal xyz; Out: update summaries + state mutation | Called from `runtime_loop` after place verify and from correction flows | Overlaps with `placed_ledger`; this is explicit authoritative layer separate from historical ledger | stay |
| `runtime_core.py` | `reconcile_scene`, `run_place_space_truth_pass` | Scene-to-state reconciliation, drift/collision detection, section summaries | In: scans + side mode/target xyz; Out: reconcile/truth dict | Called in startup, pre/post return/place/correction flows and `runtime_loop` | Related color/section inference exists in verify filters; reconcile is broader state audit | stay |
| `runtime_core.py` | `_planner_section_row_unified`, `get_section_confirmed_color_sequence_bottom_to_top` | Shapes authoritative section identity for planner input | In: state (+ hints); Out: compact section row | Used by planner state builders and correction eligibility checks | Similar compacting helpers exist (`_compact_section_truth_row`) | reuse |

---

## 6) centering controller

| File | Function/Class/Constant | Responsibility | Inputs / Outputs | Where Called | Similar Logic Elsewhere? | Recommendation |
|---|---|---|---|---|---|---|
| `runtime_core.py` | `KYAW`, `KSHOULDER`, `KELBOW`, `CENTER_*`, `PX_TOL` | Pixel-error -> joint nudge gain and stability thresholds | Env/defaults -> constants | Used by centering/verify/handoff loops | No separate centering config module in inspected files | stay |
| `runtime_core.py` | `_compute_centering_nudge`, `_maybe_apply_centering_nudge`, integrator helpers | Core P+I nudge controller and anti-windup behavior | In: pixel errors/confidence; Out: joint delta or frame-advance decisions | Used by `center_object_slowly`, `center_object_on_expected_slot`, `run_track_handoff_session` | Similar per-loop logic exists, but these helpers are central | reuse |
| `runtime_core.py` | `center_object_slowly` | Main pick-side centering loop with selector hook and track overlay | In: detector/arm/per (+ selector context); Out: locked `(u,v)` or `None` | Called by `acquire_and_center_intended_cube` | Verify centering uses separate function (`center_object_on_expected_slot`) with different gates | stay |
| `runtime_core.py` | `center_object_on_expected_slot` | Verify-slot centering around expected XYZ with color/section filtering and blacklist behavior | In: expected xyz + verify filters; Out: locked `(u,v)` or `None` | Called by stack remeasure helper and verify flows | Shares control primitives with `center_object_slowly`; specialized logic should remain separate | left alone |
| `runtime_core.py` | `acquire_and_center_intended_cube`, `run_pick_center_cycle` | Orchestrates target-track selection + centering for standard pick flow | In: state/arm/per/det + blocks; Out: status + centered UV | Called by `runtime_loop` observe and tune flow | Handoff sessions also perform centering-like loops for special stages | stay |

---

## 7) pick/grasp flow

| File | Function/Class/Constant | Responsibility | Inputs / Outputs | Where Called | Similar Logic Elsewhere? | Recommendation |
|---|---|---|---|---|---|---|
| `runtime_core.py` | `GraspPlan`, `_build_grasp_plan` | Build measured grasp trajectory and safety checks | In: centered UV + arm/perception + grip config; Out: `GraspPlan` or fail reason | Used by `safe_grasp` | Motion execution pieces reused in place flow but plan semantics differ | stay |
| `runtime_core.py` | `_close_for_grasp` | Close gripper with current guard + optional soft retry + hold-floor clamp | In: `Arm`, `GraspPlan`; Out: `(ok, hold_grip, reason)` | Called by `safe_grasp` | Uses centralized guard in `current_sensing_grip_limiting.py` | reuse |
| `runtime_core.py` | `_lift_verify_and_tune_grasp` | Lift after close, verify current signal, tune hold grip if needed | In: arm/plan/hold grip; Out: `(ok, tuned_grip, reason)` | Called by `safe_grasp` | Stability recheck is also done later in carry via `verify_pick_stability_signal` | reuse |
| `runtime_core.py` | `safe_grasp` | End-to-end grasp from centered pixel to lifted object | In: arm/perception/centered UV; Out: `(ok, hold_grip, reason, grasp_info)` | Called by `run_grasp_and_carry_common` | No second full grasp implementation in inspected files | stay |
| `runtime_core.py` | `run_grasp_and_carry_common` | Shared post-center grasp + carry-mid + stability verification for all pick-like actions | In: state/arm/per/centered UV; Out: status, hold grip, carry supervisor | Called by normal pick, misplaced pick, return-placed correction | Strong overlap avoidance point; should remain shared | reuse |
| `runtime_core.py` | `GRIP_*`, `GRASP_*`, `POST_LIFT_*`, `PICK_STABILITY_RECHECK_*` | Grasp/current thresholds and trajectory parameters | Env/defaults -> constants | Used by grasp helpers above | Related current thresholds also defined in `current_sensing_grip_limiting.py` dataclasses | stay |

---

## 8) pick_misplaced flow

| File | Function/Class/Constant | Responsibility | Inputs / Outputs | Where Called | Similar Logic Elsewhere? | Recommendation |
|---|---|---|---|---|---|---|
| `runtime_core.py` | `can_pick_misplaced_cube_now`, `can_pick_misplaced_on_side_now` | Gate whether side-specific correction pick is currently legal | In: state (+ stack levels hint); Out: boolean | Used in command allow-list and correction command routing | Planner-step command builder has parallel policy gating, but these are hard runtime guards | reuse |
| `runtime_core.py` | `execute_pick_misplaced_cube_action` | Full side-specific misplaced-cube correction flow (lock/top-side enforcement, grasp, drop, authoritative pop) | In: state/arm/det/per/preferred side; Out: `(ok, reason, context)` | Called by `runtime_loop` for `pick_placed_*` and `pick_misplaced_*` actions | Reuses generic `run_track_handoff_session`; no other full misplaced implementation found | stay |
| `runtime_core.py` | `MISPLACED_PICK_*` constants | Correction lock quality gates, attempt limits, timeout behavior, side gating | Env/defaults -> constants | Used heavily inside `execute_pick_misplaced_cube_action` | Similar reject/timeout controls exist in pick_other and return handoff constants | stay |
| `runtime_core.py` | `MISPLACED_RETURN_DROP_*` constants | Defines correction-drop target mini-arc progression for repeated drops | Env/defaults -> constants | Used in tail of `execute_pick_misplaced_cube_action` | Return-to-origin logic uses different targets (`last_pick_return_xyz`) | left alone |
| `runtime_core.py` | `pop_authoritative_stack_top` (via correction success) | Decrements authoritative side stack after correction removal | In: state/section/removed xyz; Out: update summary | Called inside misplaced and return-placed correction flows | Append counterpart exists (`append_authoritative_stack_cube`) | reuse |

---

## 9) return/drop-zone logic

| File | Function/Class/Constant | Responsibility | Inputs / Outputs | Where Called | Similar Logic Elsewhere? | Recommendation |
|---|---|---|---|---|---|---|
| `runtime_core.py` | `execute_return_cube_action` | Returns currently held cube to recorded pick origin (`state.last_pick_return_xyz`) with occupancy reconcile | In: state + hold grip + carry supervisor; Out: `(ok, reason, context)` | Called by `runtime_loop` `return_cube` action and tune path | Uses same `safe_place` primitive as placement/correction drop | reuse |
| `runtime_core.py` | `run_return_verify_stage` | Verify the just-returned cube identity/location and lock its track for blocking | In: return target xyz + track session params; Out: verify-stage dict | Called by `run_return_verify_and_handoff_session`; also directly in tune flow | Structure parallels verify-v2 handoff patterns | stay |
| `runtime_core.py` | `run_return_handoff_stage` | After return verification, lock next valid target (excluding returned cube) and auto-classify it | In: blocked-return identifiers + session params; Out: handoff dict | Called by `run_return_verify_and_handoff_session` | Similar target handoff behavior in pick_other/misplaced sessions, but this one is return-specific | stay |
| `runtime_core.py` | `run_return_verify_and_handoff_session` | Composes verify + handoff into one policy-facing flow | In: state/arm/per/det/return target; Out: combined flow dict | Called by `runtime_loop` return action | Wrapper composition pattern; reuses two stage functions | reuse |
| `runtime_core.py` | `execute_return_placed_cube_correction` | Correction flow for returning a previously placed stack cube back to pick origin | In: state/stack context; Out: `(ok, reason, context)` | Called by `runtime_loop` `return_placed_cube` action | Reuses return verify + grasp/carry + safe_place + authoritative pop, similar to misplaced correction | stay |
| `runtime_loop.py` | return action orchestration (`return_cube` branch) | Applies returned-cube blocklist state and handoff result into loop state | In: flow results; Out: state mutations, planner feedback | Inside `main_prompted` | Some overlap with tune flow, but prompted loop is primary | left alone |

---

## 10) place verification

| File | Function/Class/Constant | Responsibility | Inputs / Outputs | Where Called | Similar Logic Elsewhere? | Recommendation |
|---|---|---|---|---|---|---|
| `runtime_core.py` | `verify_last_place_reliability` | Main verify-v2: track lock, measurement, geometry score, color sampling, result commit to ledger/state | In: state/arm/per/det; Out: verify result dict | Called via `verify_v2.verify_last_place_reliability` from `runtime_loop`; called during startup bootstrap and explicit verify action | `verify_v2.py` is a thin export wrapper only | stay |
| `runtime_core.py` | `score_place_geometry` | Scores measured vs expected geometry with margins/overlap/delta | In: expected/measured xyz + hits/margins; Out: status dict (`confirmed`, errors) | Called by verify-v2 core and slot-observation workflows | No alternate scoring function found | reuse |
| `runtime_core.py` | `collect_slot_observations` | Collects multi-frame projected candidates around expected slot for pre/post observation | In: expected xyz + sample/tolerance config; Out: observation summary | Called in `execute_prompted_place_action` pre/post verify support | Verify-v2 track-locked measurement path is separate but conceptually related | left alone |
| `runtime_core.py` | `center_object_on_expected_slot` | Verify recenter helper around expected slot/section/color constraints | In: expected xyz + filters; Out: lock UV or `None` | Used by remeasure helpers and verify-related flows | Related to pick centering but specialized | reuse |
| `runtime_core.py` | `PLACE_VERIFY_V2_*`, `PLACE_VERIFY_*` constants | Verification thresholds, recenter policy, color/track gating, margins | Env/defaults -> constants | Used throughout verify helpers and place action flow | N/A | stay |
| `verify_v2.py` | `verify_last_place_reliability` export (and `collect_slot_observations`, `score_place_geometry`) | Compatibility wrapper/re-export module | Inputs/outputs unchanged from core | Used by `runtime_loop` import path | Direct logic lives in `runtime_core.py` | reuse |

---

## 11) kinematics/camera transforms

| File | Function/Class/Constant | Responsibility | Inputs / Outputs | Where Called | Similar Logic Elsewhere? | Recommendation |
|---|---|---|---|---|---|---|
| `runtime_core.py` | `_dh` | Builds DH homogeneous transform matrix | In: `theta,d,a,alpha`; Out: `4x4 np.ndarray` | Called by `base_to_camera_T` | No duplicate in inspected files | stay |
| `runtime_core.py` | `base_to_camera_T` + `SHOULDER_HEIGHT_M`, `LINK2_LENGTH_M`, `LINK3_LENGTH_M`, `CAM_OFF_*` | Computes base->camera transform using measured joints and calibrated camera offsets | In: yaw/shoulder/elbow; Out: `4x4 T_base_to_cam` | Used by `estimate_base_xyz_from_uv_fast` and `measure_base_point_from_uv` | Transform math is also manually reused in measurement path by design | reuse |
| `runtime_core.py` | `uvz_to_xyz_cam` | Converts image coordinates/depth to camera XYZ using intrinsics | In: pixel/depth/intr; Out: camera XYZ | Used by projection and measurement paths | N/A | reuse |
| `runtime_core.py` | `get_cam_offsets`, `set_cam_offsets` | Exposes camera-offset runtime tuning interface | In: optional tuned values; Out: dict/updated globals | Used by tuning/profile load utilities | No second camera-offset setter found | left alone |
| `runtime_core.py` | `Arm.ik`, `Arm.goto_*` methods | IK conversion and motion-space execution (joint/task/waypoint) | In: task/joint targets; Out: movement success booleans and motion diagnostics | Used across all pick/place/return flows | Current-supervision hooks are integrated in same class | stay |
| `runtime_core.py` | `Perception` class (`intr`, `depth_scale`, aligned frames) | RealSense acquisition, alignment, intrinsics cache | In: stream config; Out: color/depth frames + camera intrinsics | Used by all detection/projection/verification loops | No alternate perception wrapper found | stay |

---

## 12) current sensing/grip safety

| File | Function/Class/Constant | Responsibility | Inputs / Outputs | Where Called | Similar Logic Elsewhere? | Recommendation |
|---|---|---|---|---|---|---|
| `current_sensing_grip_limiting.py` | `class GripCurrentLimits` | Dataclass of close-time grip-current thresholds and timing behavior | In: config values; Out: limits object | Instantiated as `GRIP_CURRENT_LIMITS` in `runtime_core.py` | Runtime constants mirror these fields via env mapping | stay |
| `current_sensing_grip_limiting.py` | `class MotionSupervisionLimits` | Dataclass of move-time grip/total-current thresholds and recovery behavior | In: config values; Out: limits object | Instantiated as `MOTION_SUPERVISION_LIMITS` in `runtime_core.py` | Runtime constants mirror these fields via env mapping | stay |
| `current_sensing_grip_limiting.py` | `class MotionGripSupervisor` | Stateful move-time current supervisor (`ok`, `freeze_recovering`, `unrecoverable`) | In: current readings + commanded grip; Out: action/event/state updates | Built by `runtime_core.make_motion_supervisor`; consumed in `Arm._apply_motion_supervisor` | No duplicate supervisor class in inspected files | reuse |
| `current_sensing_grip_limiting.py` | `read_joint_currents`, `read_gripper_current`, `read_total_arm_current` | Hardware-agnostic measured-current extraction helpers | In: qarm hardware object; Out: current measurements | Used by `close_gripper_with_current_guard` and `runtime_core.read_total_arm_current_abs` | `runtime_core.read_gripper_current_abs` does direct read of joint index 4; minor overlap | left alone |
| `current_sensing_grip_limiting.py` | `close_gripper_with_current_guard` | Controlled close loop with warn/hard/emergency logic and grip relax behavior | In: arm wrapper + target pose/grip/limits; Out: close result dict (`status`, currents, grip) | Called by `runtime_core._close_for_grasp` (first pass + soft retry pass) | No second close-loop engine found | reuse |
| `runtime_core.py` | `GRIP_CURRENT_LIMITS`, `MOTION_SUPERVISION_LIMITS`, `make_motion_supervisor`, `verify_grasp_in_air_current`, `tune_hold_grip_to_current_target`, `verify_pick_stability_signal` | Runtime-level integration of safety thresholds with grasp/carry phases | In: runtime state + currents; Out: guard decisions and verification status | Used across grasp, carry, push, and motion segments | Some low-level reading overlaps with imported helpers; integration point is centralized | stay |

---

## 13) planner/LLM policy calls

| File | Function/Class/Constant | Responsibility | Inputs / Outputs | Where Called | Similar Logic Elsewhere? | Recommendation |
|---|---|---|---|---|---|---|
| `llm_commander/planner/live_policy_brain.py` | `LivePolicyConfig`, `LivePolicyDecision` | Policy backend config and normalized decision payload structures | In: config/state values; Out: dataclass objects | Constructed in `runtime_core.build_live_policy_brain`; consumed in `runtime_loop` | None in inspected files | stay |
| `llm_commander/planner/live_policy_brain.py` | `LivePolicyBrain._extract_json`, `_validate_payload`, `_call_ollama`, `decide`, `_ALLOWED_COMMANDS`, `FINAL_JSON_*` | End-to-end policy call, response extraction/validation, side-specific command enforcement | In: planner state + allowed commands; Out: `LivePolicyDecision` | `runtime_loop.main_prompted` calls `policy_brain.decide(...)` | Runtime also does action-level defensive guards for generic pick commands | stay |
| `runtime_core.py` | `LLM_POLICY_*` constants | Runtime policy backend/model/endpoint/timeout/prompt settings | Env/defaults -> constants | Used by `build_live_policy_brain` and trace logging | `LivePolicyConfig` has its own defaults, but runtime explicitly overrides at construction | reuse |
| `runtime_core.py` | `build_live_policy_brain` | Adapter that instantiates planner brain or fails loud with import/init diagnostics | In: runtime constants; Out: `LivePolicyBrain` or `None` | Called by `runtime_loop.main_prompted()` via `planner_io.build_live_policy_brain` | `planner_io.py` wrapper simply re-exports this | reuse |
| `runtime_core.py` | `build_prompted_allowed_commands`, `build_prompted_step_allowed_commands` | Computes policy command allow-list from runtime state and safety gates | In: state + stack/scene context; Out: allowed command list | Called by `runtime_loop.main_prompted` each step (via wrappers) | Additional command validation also exists in `LivePolicyBrain._validate_payload` | stay |
| `runtime_core.py` | `build_prompted_planner_state` | Shapes compact planner input state payload | In: runtime state snapshots; Out: planner input dict | Called in `runtime_loop` before each policy decision | No alternate planner-state builder found | stay |
| `runtime_core.py` | `maybe_append_policy_raw_row` | Appends policy I/O trace row for observability/debugging | In: cycle/step/LLM I/O payloads; Out: `state.raw_policy_rows` append | Called in `runtime_loop` after each `decide` call | No duplicate policy trace writer in inspected files | left alone |
| `planner_io.py` | wrapper exports (`build_live_policy_brain`, `build_prompted_*`, `maybe_append_policy_raw_row`) | Compatibility/re-export layer for planner-related runtime-core functions | Inputs/outputs unchanged | Imported and used by `runtime_loop` | Direct logic resides in `runtime_core.py` | reuse |
| `runtime_loop.py` | policy call site in `main_prompted` | Binds allowed commands + planner_state -> `policy_brain.decide`, then dispatches action handlers | In: decision + runtime state; Out: action command execution and state mutation | Core runtime loop | Some tune-mode logic bypasses LLM and uses fixed sequence | stay |

---

## Quick Path Mismatch Findings (for this inventory run)

| Requested Path | Actual Path Used | Impact |
|---|---|---|
| `current_sensing.py` | `current_sensing_grip_limiting.py` | Current/grip safety logic is active and imported from this file in `runtime_core.py`. |
| `planner/llm_policy.py` | `llm_commander/planner/live_policy_brain.py` | LLM policy implementation lives here; `planner_io.py` provides wrapper exports used by `runtime_loop.py`. |
