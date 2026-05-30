# Prompt Trial Runner

`tools/run_prompt_trials.py` runs `LLM_Commander.py` repeatedly with different
mission prompts and writes one CSV row per trial. It is external to the robot
runtime and only sets environment variables before launching the existing entry
point.

## Usage

From the project root:

```powershell
python tools/run_prompt_trials.py
python tools/run_prompt_trials.py --repeats 10
python tools/run_prompt_trials.py --session thesis_trials_01
python tools/run_prompt_trials.py --resume results/thesis_trials_01
python tools/run_prompt_trials.py --resume results/thesis_trials_01 --delete-trials 11,12,13-15
python tools/run_prompt_trials.py --prompts experiments/prompts.json
python tools/run_prompt_trials.py --prompt-id P1 --repeats 2
python tools/run_prompt_trials.py --python .\.venv312\Scripts\python.exe
python tools/run_prompt_trials.py --dry-run
```

Default collection is 30 trials: 3 prompts x 10 repeats. Trial order is
interleaved by repeat so prompt comparisons are less sensitive to drift:
`P1 r1, P2 r1, P3 r1, P1 r2, ...`.

## Operator Flow

Before every run the script prints the session name, trial id, prompt id, repeat
index, exact prompt text, previous initial condition for that prompt, and the
planned trial folder. Enter `initial_condition`, reset the scene, start video,
then press Enter to launch.

Use compact initial-condition notation such as:

```text
L=BOB;R=;P=B3O3
```

`L` and `R` are bottom-to-top stack sequences. `P` is pick-space inventory.
Each prompt in `experiments/prompts.json` can define `initial_condition`. The
runner shows that setup before launch and uses it as the default when there is
no previous value for that prompt. Press Enter on the initial-condition prompt
to reuse the previous value, or the prompt default if no previous value exists.

Pre-run commands:

- `Enter`: launch `python LLM_Commander.py`.
- `pause` or `quit`: save progress and exit without writing a trial row.
- `abort`: write an aborted row for that trial and continue.
- `dset 11,12,13-15`: delete those `trial_NNNN` rows/folders from the
  current session, rebuild CSV/progress files, and continue from the earliest
  incomplete planned trial.

After a normal zero-exit run, the runner prints the parsed auto fields and asks
only `final_result`, `notes`, and optional `video_file`. If the runtime exits
nonzero, cannot launch, or a previous run is found as still running on resume,
the runner records `final_result=aborted` and
`failure_source_auto=runtime_crash`.

## Outputs

Each trial writes:

```text
results/<session>/trial_0001/
results/<session>/trial_0001/console_log.txt
results/<session>/trial_0001/trial_meta.json
```

Session CSV files:

- `trials.csv`: one row per attempted or aborted trial.
- `failures.csv`: only `partial`, `fail`, and `aborted` rows.
- `progress.json`: completed trial ids and completed prompt/repeat keys for
  resume.

Resume skips completed prompt/repeat pairs, not just folder names. If a planned
folder such as `trial_0004` already exists, the runner leaves it alone and uses
the next available `trial_NNNN` directory for the next attempted run. Existing
trial folders are not overwritten.

To remove bad trials from a resumed session, use `--delete-trials 11,12,13-15`
at startup or type `dset 11,12,13-15` at the pre-run prompt. This removes
matching rows from `trials.csv`, rebuilds `failures.csv`, `summary.csv`, and
`progress.json`, deletes matching `trial_NNNN` folders, and then resumes.

At startup/resume the runner prints completed counts per prompt, next planned
trial, last initial condition per prompt, CSV paths, and the session folder.

`steps_required_auto` uses this precedence: prompt JSON `steps_required` or
`steps_required_default`, then a previous successful run for the same
`prompt_id + initial_condition`, then `manual_required`. For correction-heavy
tasks, `steps_completed_auto` counts place actions plus placed-cube correction
picks, capped at `steps_required_auto`.

The subprocess environment is:

- `QARM_MISSION_PROMPT`: selected prompt text.
- `QARM_POLICY_TRACE_DIR`: absolute trial directory.
- `PYTHONUNBUFFERED=1`: improves live tee logging.

## Manual Fields

`final_result`:
`success`, `partial`, `fail`, `aborted`

`notes` and `video_file` are free text. Step counts and failure-stage fields are
parsed automatically. If the parser cannot infer a value, it writes `unknown` or
`manual_required` so the CSV can be edited later.

## Plot-Friendly Columns

Use these columns for common thesis plots:

- Stacked bar by prompt: `prompt_id` x `final_result`.
- Completion by prompt: `steps_completed_auto`, `steps_required_auto`,
  `completion_ratio_auto`.
- Recovery by prompt: `num_recovery`, `recovery_attempted`.
- Failure source by prompt: `failure_source_auto`, `failure_stage_auto`,
  `failure_confidence`.
- Runtime action counts: `num_observe`, `num_classify`, `num_grasp`,
  `num_place`, `num_pick_other`, `num_pick_misplaced`, `num_verify`.
- Place verify XY accuracy: per trial `mean_place_verify_xy_error_m` and
  `place_verify_xy_error_count` parsed from `[PlaceVerifyV2] ... err_xy=...`
  (hypot distance in meters between measured and expected XY at place verify).
  `summary.csv` column `mean_place_verify_xy_error_m` averages those per-trial
  means (trials with no verify lines are omitted from that average).

## Prompt Editing

Edit `experiments/prompts.json` to change trial mission text (mirrored in
`runtime_core._MISSION_PROMPT_P*` constants). Keep each mission's final
`stop_run` line unchanged when tuning clarity. Keep exactly three prompt objects when comparing the
planned prompt set, and keep unique `id` values so resumed sessions can skip
already recorded trial ids cleanly.

## Validation

```powershell
python -m py_compile tools/run_prompt_trials.py
python tools/run_prompt_trials.py --dry-run --repeats 1
python tools/run_prompt_trials.py --dry-run --prompt-id P1 --repeats 2
```

Dry run prints the planned trial layout and does not launch
`LLM_Commander.py`.
