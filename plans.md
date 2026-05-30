# Project Progress Plan Snapshot

## 1) CURRENT FEATURES

Below is what is currently implemented in code and active in the runtime path:

- **Prompted runtime orchestration + startup bootstrap**
  - `llm_commander_refactored/runtime_loop.py`
  - `main_prompted()`
  - `run_startup_stack_bootstrap_verify(mode="full"|"refresh")`
  - Startup flow moves to `PLACE_LOOKING`, runs place-space truth, then startup hydrate identity pass, then syncs stack levels.

- **Startup hydrate track-checklist pass (track-first)**
  - `llm_commander_refactored/runtime_core.py`
  - `run_startup_stack_identity_pass(...)`
  - Discovery groups place-space candidates, builds checklist by `track_id`, attempts per-track lock/measure/classify, writes per-side hydrated levels/sequences/tracks.
  - Updated behavior: startup no longer fails early from per-track no-detection drift; it now resolves known tracks then uses an all-accounted countdown before exit.

- **Authoritative stack hydration + single planner truth source**
  - `llm_commander_refactored/runtime_core.py`
  - `apply_startup_stack_hydration(...)`
  - `_planner_section_row_unified(...)`
  - `build_prompted_planner_state(...)`
  - Planner payload is intentionally lean and driven by authoritative `section_status` rows.

- **Strict planner output parsing + command validation**
  - `llm_commander_refactored/llm_commander/planner/live_policy_brain.py`
  - `_extract_json(...)` enforces `FINAL_JSON_START/FINAL_JSON_END` tagged output.
  - `_validate_payload(...)` rejects invalid/generic correction commands such as `pick_misplaced_cube` and `pick_placed_cube`.

- **Side-specific placed-cube correction path**
  - `llm_commander_refactored/runtime_core.py`
  - `execute_pick_misplaced_cube_action(...)` (used for `pick_placed_left/right` and side-specific misplaced variants)
  - Uses `run_track_handoff_session(...)` and includes top-reference height validation for safer top-cube correction picks.

- **Track-handoff centering/lock framework**
  - `llm_commander_refactored/runtime_core.py`
  - `run_track_handoff_session(...)`
  - `center_object_on_expected_slot(...)`
  - Supports track-aware lock flow and verify-style recentering hooks.

- **Push command (pick-space only)**
  - `llm_commander_refactored/runtime_core.py`
  - `execute_push_cube_action(...)`
  - `llm_commander_refactored/runtime_loop.py` dispatch path for `push_cube`.

- **Phase-aware allowed-command gating and policy safety**
  - `llm_commander_refactored/runtime_loop.py`
  - Allowed-command filtering by phase/holding/state sanity.
  - `pick_placed_left/right` correction commands are now permitted in both `observe` and `grasp` phases.
  - Stop/run safety gates for repeated observe misses and confirmed empty-scene conditions.

- **Prompt + mission analytics tooling**
  - Prompt: `llm_commander_refactored/llm_commander/prompts/live_sort_operator_v16.txt`
  - Aggregation: `llm_commander_refactored/tools/aggregate_results_metrics.py`
  - Review workflow: `llm_commander_refactored/tools/review_trial_failures.py`
  - Supports scored runs, adjudication (`planner_error`), and chart generation (when matplotlib is available).

- **Code health checks currently passing**
  - `python3 -m py_compile` passes for:
    - `llm_commander_refactored/runtime_core.py`
    - `llm_commander_refactored/runtime_loop.py`
    - `llm_commander_refactored/llm_commander/planner/live_policy_brain.py`

## 2) DESIRED FEATURES (NEXT IMMEDIATE GOALS)

- **Startup hydrate reliability hardening (primary next goal)**
  - Make startup processing complete all visible placed cubes more consistently (left/right both captured when present).
  - Keep lock/centering target intent persistent per active `track_id` until that target is resolved.

- **Hydrate centering rewrite toward verify_v2-style stability**
  - Replace brittle startup-specific recenter behavior with a stricter, track-consistent lock flow borrowed from the most reliable verify path.
  - Avoid target drift between detections during startup hydrate.

- **Optional occupancy-first fallback (if lock misses persist)**
  - If startup lock/classify fails for a clearly discovered cube, preserve occupancy with `unknown` color rather than collapsing side level to 0.
  - Keep this explicit and visible in logs (no silent fallback behavior).

- **Prompt/state simplification continuation (v17 direction)**
  - Keep planner input minimal and unambiguous.
  - Tighten prompt wording to reduce premature correction loops and premature `stop_run`.

## 3) CURRENT CHALLENGES

- **Startup hydrate under-count issue improved, still monitor**
  - The earlier timeout-style premature exit was addressed by track-driven completion + all-accounted countdown.
  - Continue watching for edge scenes where one side can still be missed under unstable detections.

- **Hydrate lock visualization/intent can appear unstable**
  - Bounding boxes remain visible/tracked, but centering line/lock intent can blink or drop, suggesting target continuity problems in startup sequence.

- **Planner can over-correct or stop early in edge states**
  - Current observed behavior: planner may choose `pick_other` repeatedly (“orange not legal for left”, then “blue not legal for right”) instead of corrective retrieval flow.

- **State mismatch loops after correction/place failures**
  - When correction pick fails or placement goes wrong physically, authoritative state recovery is sometimes delayed or incomplete, which can trigger repeated non-productive actions.

- **Right-side correction reliability is still a risk area**
  - Intermittent `reacquire_failed`-style outcomes still appear in some `pick_placed_right` scenarios despite side-specific command/path improvements.

## 4) TECH STACK & ARCHITECTURE

- **Language/runtime**
  - Python 3 (`llm_commander_refactored` runtime).

- **Core robotics/perception dependencies**
  - Quanser QArm APIs: `pal.products.qarm`, `hal.products.qarm`
  - Intel RealSense: `pyrealsense2`
  - Computer vision: `opencv-python (cv2)`, `numpy`
  - Detection/tracking: `ultralytics YOLO`

- **LLM policy layer**
  - `llm_commander_refactored/llm_commander/planner/live_policy_brain.py`
  - Backend: local Ollama HTTP endpoint
  - Strict tagged JSON output contract (`FINAL_JSON_START/FINAL_JSON_END`)

- **Runtime architecture**
  - Entrypoint/loop: `runtime_loop.py`
  - Behavioral core: `runtime_core.py`
  - Planner prompt files: `llm_commander/prompts/*.txt`
  - Raw run traces: `policy_raw_*.jsonl`
  - Analytics + adjudication scripts under `llm_commander_refactored/tools`

- **Project operating rules (from `agents.md`)**
  - Prefer explicit failure + clear logs over hidden fallback behavior.
  - Keep planner state lean and avoid competing truth sources.
  - Use track-id continuity where possible; keep top-cube intent explicit for correction picks.
  - In robotic failure handling, explicit stop/freeze is preferred to opaque automatic recovery.
