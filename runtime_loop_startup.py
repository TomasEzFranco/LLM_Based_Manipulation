"""Startup/bootstrap helpers extracted from runtime_loop (behavior-preserving)."""

from __future__ import annotations

import numpy as np

import runtime_core as core
import stack_scene
import verify_v2


def _preserved_hydrated_stacks(*, state) -> dict:
    left_row = stack_scene.get_startup_hydrated_section_row(state, core.SECTION_LEFT_NAME)
    right_row = stack_scene.get_startup_hydrated_section_row(state, core.SECTION_RIGHT_NAME)
    return {
        "status": "preserved_authoritative",
        "sections": {
            core.SECTION_LEFT_NAME: dict(left_row),
            core.SECTION_RIGHT_NAME: dict(right_row),
        },
        "observed_stack_levels": {
            core.SECTION_LEFT_NAME: int(left_row.get("stack_level", 0) or 0),
            core.SECTION_RIGHT_NAME: int(right_row.get("stack_level", 0) or 0),
        },
        "observed_sequences": {
            core.SECTION_LEFT_NAME: list(left_row.get("color_sequence_bottom_to_top", [])),
            core.SECTION_RIGHT_NAME: list(right_row.get("color_sequence_bottom_to_top", [])),
        },
    }


def run_startup_stack_bootstrap_verify(
    *,
    state,
    arm,
    per,
    det,
    mode: str = "full",
) -> dict:
    print("[StartupPhase] bootstrap_begin -> move PLACE_LOOKING, scan truth, hydrate stacks")
    startup_row = {
        "status": "startup_bootstrap_not_run",
        "fresh": False,
        "mismatch_sides": [],
        "collision_risk": False,
        "scene_revision": int(state.scene_revision),
        "place_space_empty": False,
        "ready_for_stacking": False,
        "post_place_verify_status": "pending",
        "hydration_status": "unknown",
        "hydration_missing_sides": [],
    }
    try:
        mode_norm = str(mode).strip().lower()
        is_refresh_mode = bool(mode_norm == "refresh")
        arm.goto_task_space(
            core.PLACE_LOOKING,
            duration=1.0,
            label="startup_bootstrap_place_look",
        )
        truth = stack_scene.run_place_space_truth_pass(
            state=state,
            arm=arm,
            per=per,
            det=det,
            centered_pos=None,
            active_track_id=None,
            mode="startup_preflight_place_space",
            detector_draw=bool(not is_refresh_mode),
            show_window=bool((not is_refresh_mode) and core.STARTUP_VERIFY_SHOW_WINDOW),
            status_line="startup_verify",
        )
        truth_status = str((truth or {}).get("status", "unknown")).strip().lower()
        observed = (
            truth.get("section_status", {}).get("observed", {})
            if isinstance(truth, dict)
            else {}
        )
        try:
            left_level = int(observed.get(core.SECTION_LEFT_NAME, {}).get("stack_level", 0) or 0)
        except Exception:
            left_level = 0
        try:
            right_level = int(observed.get(core.SECTION_RIGHT_NAME, {}).get("stack_level", 0) or 0)
        except Exception:
            right_level = 0
        place_space_empty = bool(left_level == 0 and right_level == 0)
        if bool(is_refresh_mode and place_space_empty):
            # Preserve existing authoritative stack state on empty refresh scans.
            # Do not collapse to zero based on a transient empty observation.
            hydrated_stacks = _preserved_hydrated_stacks(state=state)
            startup_post_place_status = "skipped_refresh"
        else:
            hydrated_stacks = stack_scene.run_startup_stack_identity_pass(
                state=state,
                arm=arm,
                per=per,
                det=det,
                samples=max(1, int(core.STARTUP_STACK_IDENTITY_SAMPLES)),
                show_window=bool((not is_refresh_mode) and core.STARTUP_VERIFY_SHOW_WINDOW),
                detector_draw=bool(not is_refresh_mode),
            )
            if is_refresh_mode:
                startup_post_place_status = "skipped_refresh"
            else:
                startup_verify_result = verify_v2.verify_last_place_reliability(
                    state=state,
                    arm=arm,
                    per=per,
                    det=det,
                    count_in_stats=False,
                )
                startup_post_place_status = str(
                    (startup_verify_result or {}).get("status", "unknown")
                ).strip().lower()
        hydration_status = str(
            (hydrated_stacks or {}).get("status", "unknown")
        ).strip().lower()
        unresolved_visible_track_ids = [
            int(tid)
            for tid in list((hydrated_stacks or {}).get("unresolved_visible_track_ids", []))
            if isinstance(tid, (int, np.integer))
            or (isinstance(tid, str) and str(tid).strip().lstrip("-").isdigit())
        ]
        unresolved_visible_track_ids = sorted(
            int(tid) for tid in unresolved_visible_track_ids
        )
        hydrated_levels = (
            (hydrated_stacks or {}).get("observed_stack_levels", {})
            if isinstance(hydrated_stacks, dict)
            else {}
        )
        try:
            hydrated_left_level = int(
                hydrated_levels.get(core.SECTION_LEFT_NAME, 0) or 0
            )
        except Exception:
            hydrated_left_level = 0
        try:
            hydrated_right_level = int(
                hydrated_levels.get(core.SECTION_RIGHT_NAME, 0) or 0
            )
        except Exception:
            hydrated_right_level = 0
        hydration_missing_sides: list[str] = []
        if int(left_level) > 0 and int(hydrated_left_level) <= 0:
            hydration_missing_sides.append(core.SECTION_LEFT_NAME)
        if int(right_level) > 0 and int(hydrated_right_level) <= 0:
            hydration_missing_sides.append(core.SECTION_RIGHT_NAME)
        expected_levels = (
            (hydrated_stacks or {}).get("expected_stack_levels", {})
            if isinstance(hydrated_stacks, dict)
            else {}
        )
        hydration_expected_shortfall_sides: list[str] = []
        if bool(core.STARTUP_STACK_REQUIRE_EXPECTED_LAYERS):
            for side_name, hydrated_level in [
                (core.SECTION_LEFT_NAME, int(hydrated_left_level)),
                (core.SECTION_RIGHT_NAME, int(hydrated_right_level)),
            ]:
                try:
                    expected_level = int(expected_levels.get(side_name, hydrated_level) or 0)
                except Exception:
                    expected_level = int(hydrated_level)
                if int(expected_level) > int(hydrated_level):
                    hydration_expected_shortfall_sides.append(str(side_name))
                    if side_name not in hydration_missing_sides:
                        hydration_missing_sides.append(str(side_name))
        hydrate_lock_incomplete = bool(hydration_missing_sides)
        if bool(core.STARTUP_STACK_ASSIGN_DEBUG):
            print(
                f"[StartupHydrateDiag] hydrated_levels="
                f"{{'{core.SECTION_LEFT_NAME}': {int(hydrated_left_level)}, '{core.SECTION_RIGHT_NAME}': {int(hydrated_right_level)}}} "
                f"expected_levels={dict(expected_levels) if isinstance(expected_levels, dict) else {}} "
                f"hydration_status={hydration_status} missing={list(hydration_missing_sides)} "
                f"expected_shortfall={list(hydration_expected_shortfall_sides)}"
            )
        mismatch_sides = [
            str(s).strip().lower()
            for s in list((truth or {}).get("mismatch_sides", []))
            if str(s).strip().lower() in {core.SECTION_LEFT_NAME, core.SECTION_RIGHT_NAME}
        ]
        has_confirmed_baseline = False
        for entry in list(state.placed_ledger):
            if not isinstance(entry, dict):
                continue
            if bool(entry.get("removed_by_return", False)):
                continue
            verify_row = entry.get("verify_result", None)
            if isinstance(verify_row, dict) and bool(verify_row.get("confirmed", False)):
                has_confirmed_baseline = True
                break
        # At startup, when no confirmed baseline exists yet, mismatch against ledger
        # is not actionable and can mislead the planner/logs.
        if not has_confirmed_baseline:
            mismatch_sides = []
        if hydration_missing_sides:
            for side_name in hydration_missing_sides:
                if side_name not in mismatch_sides:
                    mismatch_sides.append(str(side_name))
        collision_risk = bool((truth or {}).get("collision_risk", False))
        scene_revision = int((truth or {}).get("scene_revision", state.scene_revision))
        ready_for_stacking = bool(
            (truth_status == "ok")
            and (not collision_risk)
            and ((not has_confirmed_baseline) or (len(mismatch_sides) == 0))
        )
        if hydrate_lock_incomplete:
            ready_for_stacking = False
        if unresolved_visible_track_ids:
            ready_for_stacking = False
        startup_status = str(f"startup_place_space_{truth_status}")
        if hydrate_lock_incomplete:
            startup_status = str(f"{startup_status}_hydrate_incomplete")
        if unresolved_visible_track_ids:
            startup_status = str(f"{startup_status}_unresolved_visible_tracks")
        startup_row = {
            "status": str(startup_status),
            "fresh": True,
            "mismatch_sides": list(mismatch_sides),
            "collision_risk": bool(collision_risk),
            "scene_revision": int(scene_revision),
            "place_space_empty": bool(place_space_empty),
            "ready_for_stacking": bool(ready_for_stacking),
            "has_confirmed_baseline": bool(has_confirmed_baseline),
            "post_place_verify_status": str(startup_post_place_status),
            "hydration_status": str(hydration_status),
            "hydration_missing_sides": list(hydration_missing_sides),
            "hydration_expected_shortfall_sides": list(hydration_expected_shortfall_sides),
            "hydration_unresolved_visible_track_ids": list(unresolved_visible_track_ids),
            "hydrated_stacks": dict(hydrated_stacks if isinstance(hydrated_stacks, dict) else {}),
        }
    except Exception as exc:
        startup_row = {
            "status": "startup_bootstrap_error",
            "fresh": False,
            "mismatch_sides": [],
            "collision_risk": False,
            "scene_revision": int(state.scene_revision),
            "place_space_empty": False,
            "ready_for_stacking": False,
            "post_place_verify_status": f"error:{exc}",
            "hydration_status": "error",
            "hydration_missing_sides": [],
            "hydration_expected_shortfall_sides": [],
            "hydration_unresolved_visible_track_ids": [],
            # Preserve current authoritative stacks on bootstrap error rather than
            # falling back to an empty state.
            "hydrated_stacks": _preserved_hydrated_stacks(state=state),
        }
        print(f"[StartupVerify] bootstrap error: {exc}")
    finally:
        state.last_begin_stack_verify = dict(startup_row)
        print(
            f"[StartupVerify] post_place_verify={state.last_begin_stack_verify.get('post_place_verify_status')} "
            f"hydrate_status={state.last_begin_stack_verify.get('hydration_status', 'unknown')} "
            f"hydrate_missing={state.last_begin_stack_verify.get('hydration_missing_sides', [])} "
            f"hydrate_expected_shortfall={state.last_begin_stack_verify.get('hydration_expected_shortfall_sides', [])} "
            f"hydrate_unresolved={state.last_begin_stack_verify.get('hydration_unresolved_visible_track_ids', [])} "
            f"place_space_empty={bool(state.last_begin_stack_verify.get('place_space_empty', False))} "
            f"ready_for_stacking={bool(state.last_begin_stack_verify.get('ready_for_stacking', False))} "
            f"collision_risk={bool(state.last_begin_stack_verify.get('collision_risk', False))} "
            f"rev={int(state.last_begin_stack_verify.get('scene_revision', state.scene_revision))}"
        )
        arm.goto_task_space(
            core.PICK_LOOKING,
            duration=1.0,
            label="startup_bootstrap_return_pick_look",
        )
        print("[StartupPhase] bootstrap_end -> returned PICK_LOOKING")
    return dict(startup_row)


def sync_stack_levels_from_startup_bootstrap(
    *,
    state,
    stack_levels: dict[str, int],
    startup_boot_row: dict | None,
) -> None:
    if not isinstance(startup_boot_row, dict):
        return
    levels = stack_scene.apply_startup_stack_hydration(state, startup_boot_row)
    if not isinstance(levels, dict):
        return
    try:
        left_level = int(levels.get(core.SECTION_LEFT_NAME, stack_levels.get(core.SECTION_LEFT_NAME, 0)) or 0)
    except Exception:
        left_level = int(stack_levels.get(core.SECTION_LEFT_NAME, 0))
    try:
        right_level = int(levels.get(core.SECTION_RIGHT_NAME, stack_levels.get(core.SECTION_RIGHT_NAME, 0)) or 0)
    except Exception:
        right_level = int(stack_levels.get(core.SECTION_RIGHT_NAME, 0))
    stack_levels[core.SECTION_LEFT_NAME] = max(0, int(left_level))
    stack_levels[core.SECTION_RIGHT_NAME] = max(0, int(right_level))


def sync_stack_levels_from_authoritative_state(
    *,
    state,
    stack_levels: dict[str, int],
) -> None:
    levels = stack_scene.get_authoritative_stack_levels(state)
    try:
        stack_levels[core.SECTION_LEFT_NAME] = max(
            0, int(levels.get(core.SECTION_LEFT_NAME, stack_levels.get(core.SECTION_LEFT_NAME, 0)) or 0)
        )
    except Exception:
        stack_levels[core.SECTION_LEFT_NAME] = max(0, int(stack_levels.get(core.SECTION_LEFT_NAME, 0)))
    try:
        stack_levels[core.SECTION_RIGHT_NAME] = max(
            0, int(levels.get(core.SECTION_RIGHT_NAME, stack_levels.get(core.SECTION_RIGHT_NAME, 0)) or 0)
        )
    except Exception:
        stack_levels[core.SECTION_RIGHT_NAME] = max(0, int(stack_levels.get(core.SECTION_RIGHT_NAME, 0)))
