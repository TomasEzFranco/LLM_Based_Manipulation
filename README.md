# LLM_Based_Manipulation

This repository contains the software used for a thesis project on LLM-based robotic manipulation with a Quanser QArm. The system uses computer vision, cube tracking, structured robot actions, and an LLM policy to sort and stack colored cubes.

The project is organized around a real hardware loop:

1. Detect cubes with YOLO and RealSense depth.
2. Center the camera/arm on a target cube.
3. Ask an LLM policy to choose the next legal command.
4. Execute grasp, place, correction, or stop actions.
5. Verify stack state with geometry, color, and depth measurements.
6. Record trial logs and CSV summaries for analysis.

## Hardware And Dependencies

The runtime is intended for a Windows workstation connected to:

- Quanser QArm hardware.
- Quanser PAL/HAL Python libraries.
- Intel RealSense camera and `pyrealsense2`.
- A local YOLO cube detector model such as `best.pt`.
- A local LLM backend used by `llm_commander/planner/live_policy_brain.py`.

Large model files and run outputs are ignored by git. Keep model weights and trial results local unless they are archived separately.

## Running The Robot Runtime

From the repository root:

```powershell
python LLM_Commander.py
```

`LLM_Commander.py` is a thin entrypoint that starts the prompted runtime loop in `runtime_loop.py`.

The default policy prompt is:

```text
llm_commander/prompts/live_sort_operator_v22.txt
```

Older prompt versions are kept in `llm_commander/prompts/` for comparison and rollback.

## How The Runtime Works

The runtime is a constrained action loop. The LLM does not directly control motors. Instead, each cycle builds a JSON state summary and gives the LLM a list of legal commands.

Typical commands include:

- `observe_scene`
- `classify_cube`
- `grasp_cube`
- `pick_other`
- `place_left`
- `place_right`
- `place_left_stack`
- `place_right_stack`
- `pick_placed_left`
- `pick_placed_right`
- `pick_misplaced_left`
- `pick_misplaced_right`
- `return_cube`
- `stop_run`

The LLM returns one command and a reason. The runtime validates that command against the current phase before executing it.

## Policy State

The policy receives a compact `INPUT_JSON` payload with:

- The mission prompt.
- The current phase: observe, classification, grasp, or place.
- Whether the robot is holding a cube.
- The centered pick target or held cube color.
- Left and right stack status.
- The commands that are currently allowed.

The policy does not receive raw camera frames or unrestricted robot state.

## Perception And Tracking

The perception stack uses:

- YOLO detections for cube candidates.
- RealSense depth for 3D projection.
- Track memory for continuity while centering.
- Color classification from image patches.
- Stack-aware geometry checks for placed cubes.

Track IDs are helpful for continuity, but they are not treated as physical cube identity. For stacked cubes, the runtime uses side, layer, XYZ position, color, and confidence.

## Grasping And Placing

`pick_actions.py` handles pick-space grasps and `pick_other`.

`place_actions.py` handles base placements and stack placements. Stack placements use a stable base anchor so the stack column does not drift as higher layers are verified.

Recent stack placement behavior:

- Stack X can be adjusted from the measured pick position.
- Third-layer stack X has a small command-only correction.
- Third-layer release Z uses the same stack height step as the second layer.
- Verification targets remain nominal, while command-only offsets are logged separately.

## Verification

`verify_v2.py` checks whether a placement succeeded. It combines:

- Expected slot geometry.
- Measured XYZ from depth.
- XY and Z margins.
- Color checks when trusted.
- Target-side top candidate checks.
- Startup hydrate recovery when needed.

The verify ladder is designed to fail with clear logs instead of silently accepting uncertain stack state.

## Correction Actions

Correction commands remove cubes from existing stack areas:

- `pick_placed_left`
- `pick_placed_right`
- `pick_misplaced_left`
- `pick_misplaced_right`

These actions explicitly target a side and prefer the top cube. They verify section and height after centering so a correction pick does not accidentally grab a lower cube.

## Experiment Runner

The experiment runner is:

```text
tools/run_prompt_trials.py
```

It repeatedly launches `LLM_Commander.py` with prompts from:

```text
experiments/prompts.json
```

Default run:

```powershell
python tools/run_prompt_trials.py
```

Useful commands:

```powershell
python tools/run_prompt_trials.py --repeats 10
python tools/run_prompt_trials.py --session FinalVideoTrials
python tools/run_prompt_trials.py --prompt-id prompt_2_orange_right --repeats 2
python tools/run_prompt_trials.py --resume results/FinalVideoTrials
python tools/run_prompt_trials.py --resume results/FinalVideoTrials --delete-trials 2,4,8-10
python tools/run_prompt_trials.py --dry-run --repeats 1
```

The runner is interactive. Before each trial it prints:

- Trial ID.
- Prompt ID.
- Repeat index.
- Prompt text.
- Expected initial condition.
- Trial output folder.

The operator sets the physical scene, starts video capture if needed, and presses Enter to launch.

## Initial Condition Notation

The runner uses compact setup strings:

```text
L=BOB;R=;P=B3O3
L=OOB;R=;P=O1
L=BOB;R=B;P=2O
```

Meaning:

- `L` is the left stack from bottom to top.
- `R` is the right stack from bottom to top.
- `P` is the pick-space inventory.
- `B` means blue.
- `O` means orange.

## Trial Outputs

Each trial creates:

```text
results/<session>/trial_0001/
results/<session>/trial_0001/console_log.txt
results/<session>/trial_0001/trial_meta.json
```

Each session creates:

```text
results/<session>/trials.csv
results/<session>/failures.csv
results/<session>/summary.csv
results/<session>/progress.json
```

The CSV files include:

- Final result: success, partial, fail, or aborted.
- Completion ratio.
- Runtime action counts.
- Recovery counts.
- Verify statistics.
- Final left/right stack state.
- Parsed failure stage and failure source.
- Operator notes and video filename.

See `docs/prompt_trial_runner.md` for the detailed runner reference.

## Prompt Set

The current experiment prompt file contains four prompt families:

- `prompt_1_color_split`: blue cubes left, orange cubes right.
- `prompt_2_orange_right`: all orange cubes on the right stack.
- `prompt_3_alternating_two_stacks`: both stacks should be blue-orange-blue.
- `prompt_4_alternating_correction`: correction-heavy alternating stack task.

Prompt IDs should stay stable when resuming experiment sessions.

## Main Files

```text
LLM_Commander.py                         Runtime entrypoint.
runtime_loop.py                          Main prompted control loop.
runtime_loop_cycle.py                    One policy/action cycle.
runtime_loop_actions_*.py                Action-specific loop handlers.
runtime_core.py                          Constants, hardware wrapper, shared helpers.
vision_runtime.py                        YOLO and RealSense runtime.
projection_geometry.py                   Depth and base-frame projection.
centering.py                             Visual centering.
pick_actions.py                          Grasp and pick_other actions.
place_actions.py                         Place and stack actions.
misplaced_actions.py                     Correction actions.
verify_v2.py                             Placement verification.
stack_scene.py                           Stack state and anchors.
planner_io.py                            Policy input/output helpers.
tools/run_prompt_trials.py               Experiment runner.
tests/offline_smoke_tests.py             Offline import and helper tests.
```

## Validation

Offline checks do not touch the real robot:

```powershell
python -m py_compile runtime_core.py runtime_loop.py runtime_loop_observe.py place_actions.py verify_v2.py tools\run_prompt_trials.py tests\offline_smoke_tests.py
python -m unittest tests.offline_smoke_tests
python tools\run_prompt_trials.py --dry-run --repeats 1
```

## Safety Notes

This repository controls physical robot hardware. Use caution during live runs.

- Keep hands clear of the QArm workspace.
- Use small, reviewable parameter changes.
- Prefer explicit failures and clear logs over hidden fallback behavior.
- Do not tune current limits casually.
- If repeated Quanser HIL timeout messages appear, treat the run as a hardware communication failure and reset the hardware path before retrying.

## Git-Ignored Outputs

The repository intentionally ignores:

- `results/`
- `Test Results/`
- Python caches.
- YOLO/model artifacts such as `*.pt`, `*.onnx`, and `*.engine`.
- Generated videos, images, and logs.

This keeps the public repository focused on source code, prompts, tests, and experiment tooling.
