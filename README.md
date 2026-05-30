# `llm_commander_refactored`

## Summary
This folder is a **guarded, standalone runtime** for refactoring `LLM_Commander` without touching the original control script.

It is currently in **M0+M1**:
1. M0: frozen baseline copy + manifest/version guard.
2. M1: thin main script + extracted runtime loop/module wiring with behavior-preserving intent.

Practical meaning:
1. You can run experiments here without modifying the legacy runtime.
2. Model, prompt, and results default locally to this folder.
3. Core behavior still comes from `runtime_core.py` while extraction proceeds.

## Goals
1. Keep original root runtime untouched.
2. Preserve behavior while making control flow easier to read/explain.
3. Keep assets and outputs local for clean experimental provenance.

## Run
From project root:

```bash
python3 LLM_Commander_refactored_launcher.py
```

Tune mode:

```bash
python3 LLM_Commander_refactored_launcher.py --mode tune
```

Tune behavior highlights:
1. If failures streak to `QARM_TUNE_MAX_FAILS`, tune restores **best-so-far** camera offsets and grasp-z fraction, then continues.
2. This recovery is bounded by `QARM_TUNE_MAX_FAIL_RECOVERIES`.
3. Tune can also adjust `GRASP_Z_PICK_FRACTION` (`QARM_TUNE_ENABLE_GRASP_Z_TUNE=1` by default).
4. Grasp Z now uses a top-cube model: `z_pick = z_measured - (1-frac) * cube_edge` (see `QARM_GRASP_CUBE_EDGE_M`).

Or run directly:

```bash
cd llm_commander_refactored
python3 LLM_Commander.py
```

## Local Defaults
When run via launcher or from this folder:
1. Model defaults to `llm_commander_refactored/best.pt`.
2. Prompt defaults to `llm_commander_refactored/llm_commander/prompts/live_sort_operator_v10.txt`.
3. Results default under `llm_commander_refactored/Test Results/...`.
4. Policy raw traces default under `llm_commander_refactored/Test Results/live_policy_runs/...`.
5. Tune trials default under `llm_commander_refactored/Test Results/tune_runs/...`.

## Calibration Profiles
1. Tune mode writes a baseline snapshot and tuned profiles under `llm_commander_refactored/tune_profiles/`.
2. Tune mode updates `tune_profiles/latest.json` with the best tuned offsets from that run.
3. Tune profiles include camera offsets and `grasp_z_pick_fraction`.
4. Prompted mode now auto-loads `tune_profiles/latest.json` when present (can be disabled):

```bash
QARM_CALIB_PROFILE_AUTO=0 \
python3 LLM_Commander_refactored_launcher.py
```

5. Explicit path still overrides auto-load:

```bash
QARM_CALIB_PROFILE_PATH=/abs/path/to/llm_commander_refactored/tune_profiles/latest.json \
python3 LLM_Commander_refactored_launcher.py
```

When loaded, the profile now applies both:
1. `cam_off_x_m / cam_off_y_m / cam_off_z_m`
2. `grasp_z_pick_fraction` (if present in the profile)

## Default Stack Locations (Current Runtime)
These are the default place-stack target locations used by `runtime_core.py` before env overrides:
1. Grid center: `x=0.390`, `y=0.180` (`QARM_PLACE_GRID_CENTER_Y_M` default derives from `PLACE_LOOKING.y + 0.06`).
2. Grid spacing: `dx=0.070`, `dy=0.090`.
3. Base slot coordinates generated from that grid:
   - slot 0: `(0.460, 0.090, 0.037)`
   - slot 1: `(0.460, 0.270, 0.037)`
4. Logical section mapping (after runtime mirror swap):
   - `left` stack uses the higher-`y` base slot (`~y=0.270`).
   - `right` stack uses the lower-`y` base slot (`~y=0.090`).

Useful related tolerances:
1. Verify XY margin default: `QARM_PLACE_VERIFY_V2_XY_MARGIN_M=0.046`.
2. Section assignment XY distance default: `QARM_SCENE_RECON_SECTION_MAX_DIST_M=0.075`.

## How It Is Wired
End-to-end execution path:
1. Root launcher `LLM_Commander_refactored_launcher.py` starts Python in this folder.
2. `LLM_Commander.py` is intentionally thin and calls `runtime_loop.main_prompted()`.
3. `runtime_loop.py` holds prompted orchestration and action dispatch.
4. `planner_io.py` and `verify_v2.py` are called from `runtime_loop`.
5. In M1, those modules mostly forward to `runtime_core.py` to preserve behavior while structure is improved.

## Module Roles
1. `LLM_Commander.py`: thin entrypoint.
2. `runtime_loop.py`: high-level prompted loop and command dispatch.
3. `runtime_core.py`: source-of-truth runtime behavior during M0+M1.
4. `planner_io.py`: planner setup, allowed command shaping, planner-state shaping, raw trace append.
5. `verify_v2.py`: verification interface export.
6. `centering.py`: centering API export.
7. `grasp_place.py`: grasp/place API export.
8. `hardware_io.py`: Arm/Perception/Detector exports.
9. `geometry_perception.py`: projection/depth/transform helper exports.
10. `tracking_overlay.py`: track memory/selection/overlay exports.
11. `state_types.py`: shared dataclass exports.
12. `mission_guard.py`: M2 placeholder (currently pass-through).

## What Is Intentionally Different From Legacy Runtime
This workspace is not byte-identical by design. Intentional differences:
1. File/path defaults are folder-local for model/prompt/results.
2. Import priority is forced local in `runtime_core.py` (`sys.path` prepended with runtime base dir).
3. Planner default prompt path in `live_policy_brain.py` is file-relative in this folder.
4. Startup entry flow is modularized (`LLM_Commander.py` -> `runtime_loop.py`).

These changes are for isolation and reproducibility, not task-policy behavior changes.

## Recent Behavior Updates (And Why)
1. Added pre-grasp `classify -> pick_other` support by seeding pick-other block context from the currently classified cube (`track_id`, `uv`, `xyz` median).
Why: this allows the LLM to deliberately switch target cube before committing to `grasp_cube`.
2. Added `pick_other_block_source` (`none|classify|return`) and exposed planner-state `pick_options.pick_other_source`.
Why: this keeps LLM decision context explicit and reduces ambiguous pick-other usage.
3. `pick_other` stays one-shot: block context clears on successful alternate lock, but persists on `observe_retry`.
Why: this prevents stale exclusions while still allowing repeated retries in hard scenes.
4. `return_cube -> pick_other` path remains intact.
Why: preserves the original reroute behavior while expanding autonomy earlier in the flow.

## Behavior Parity Expectation
Target for M0+M1 is behavior-preserving execution with structural cleanup.

Recommended parity validation on hardware:
1. One static stack mission.
2. One alternating stack mission.
3. Compare command stream, verify statuses, stop reason, completion reliability, and timing.

## Version Guard / Baseline Manifest
Critical frozen files and hashes are tracked in `BASELINE_MANIFEST.json`.

Regenerate after intentional critical-file updates:

```bash
python3 llm_commander_refactored/tools/regenerate_baseline_manifest.py
```

Verify:

```bash
python3 llm_commander_refactored/tools/verify_baseline_manifest.py
```

## Key Files To Know
1. `LLM_Commander_original_snapshot.py`: frozen baseline snapshot for comparison.
2. `runtime_core.py`: main behavior file during this phase.
3. `llm_commander/planner/live_policy_brain.py`: live policy backend adapter.
4. `llm_commander/prompts/live_sort_operator_v10.txt`: active planner system prompt.
5. `llm_commander/prompts/live_sort_operator_v8.txt` and `live_sort_operator_v9.txt`: preserved prior prompt versions.
6. `BASELINE_MANIFEST.json`: cryptographic manifest of critical assets.

## Current Phase Status
1. M0 complete: isolated folder, copied assets, manifest tooling.
2. M1 in progress: loop extraction and module wiring in place; deeper logic migration still ongoing.
3. M2 not active yet: mission guard logic is scaffolded but pass-through.

## Notes for Thesis/Demo Use
1. Use this folder as experimental runtime so baseline remains preserved.
2. Keep mission prompt/model/think-mode fixed per run for fair comparisons.
3. Archive raw policy trace and verification logs per trial for post-hoc analysis.
