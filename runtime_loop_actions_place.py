"""Place-action handlers extracted from runtime_loop (behavior-preserving)."""

from __future__ import annotations

import math
import time

import place_actions
import runtime_loop_dispatch
import runtime_loop_observe
import stack_scene
import verify_v2


def _build_place_fail_diag(
    *,
    reason: str,
    grip_cmd: float,
    arm,
    state,
    clamp_grip_cmd_fn,
) -> dict:
    diag_src = getattr(arm, "last_motion_diag", {})
    diag_row = dict(diag_src) if isinstance(diag_src, dict) else {}
    err_deg = float(diag_row.get("max_err_deg", float("nan")))
    tol_deg = float(diag_row.get("tol_deg_used", float("nan")))
    settle_s = float(diag_row.get("settle_time_s", float("nan")))
    label = str(diag_row.get("label", "")).strip()
    motion_state = str(diag_row.get("motion_state", "")).strip().lower()
    motion_reason = str(diag_row.get("last_motion_reason", getattr(arm, "last_motion_reason", ""))).strip().lower()
    print(
        f"[PlaceFailDiag] reason={str(reason)} label={label or 'unknown'} "
        f"err_deg={err_deg:.3f} tol_deg={tol_deg:.3f} settle_s={settle_s:.2f} "
        f"motion_state={motion_state or 'unknown'} motion_reason={motion_reason or 'unknown'} "
        f"holding={bool(state.holding_object)} grip={float(clamp_grip_cmd_fn(grip_cmd)):.3f}"
    )
    return {
        "place_fail_diag": {
            "reason": str(reason),
            "label": str(label),
            "err_deg": err_deg,
            "tol_deg": tol_deg,
            "settle_s": settle_s,
        }
    }


def _fmt_xyz_for_log(xyz) -> str:
    if xyz is None:
        return "None"
    try:
        vals = list(xyz)
        if len(vals) < 3:
            return "None"
        return f"[{float(vals[0]):.3f},{float(vals[1]):.3f},{float(vals[2]):.3f}]"
    except Exception:
        return "None"


def _format_place_verify_hold_diag(*, section: str, place_verify: dict, remeasure_meta: dict) -> str:
    return (
        f"[PlaceVerifyHoldDiag] section={section} status={place_verify.get('status')} "
        f"expected_xyz={_fmt_xyz_for_log(place_verify.get('expected_xyz'))} "
        f"expected_eval_xyz={_fmt_xyz_for_log(place_verify.get('expected_xyz_eval'))} "
        f"measured_xyz={_fmt_xyz_for_log(place_verify.get('measured_xyz'))} "
        f"err_xy={float(place_verify.get('xy_error_m', float('inf'))):.3f} "
        f"err_z={float(place_verify.get('z_error_m', float('inf'))):.3f} "
        f"xy_margin={float(place_verify.get('effective_xy_margin_m', float('nan'))):.3f} "
        f"z_margin={float(place_verify.get('effective_z_margin_m', float('nan'))):.3f} "
        f"overlap={float(place_verify.get('overlap_ratio', 0.0)):.2f} "
        f"remeasure={str(remeasure_meta.get('status', 'n/a'))} "
        f"remeasure_valid={int(remeasure_meta.get('valid', 0))}"
    )


def _coerce_xyz_for_place_verify(value) -> list[float] | None:
    try:
        vals = list(value)
        if len(vals) < 3:
            return None
        x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
    except Exception:
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    return [float(x), float(y), float(z)]


def _evaluate_place_verify_hydrate_fallback(
    *,
    startup_boot_row: dict | None,
    section: str,
    pending_stack_level: int | None,
    expected_color: str | None,
    place_verify: dict,
) -> dict:
    section_norm = str(section or "").strip().lower()
    if section_norm not in {"left", "right"}:
        return {"accepted": False, "reason": "invalid_section", "section": section_norm}
    if pending_stack_level is None:
        return {"accepted": False, "reason": "missing_pending_level", "section": section_norm}
    try:
        pending_i = int(pending_stack_level)
    except Exception:
        return {"accepted": False, "reason": "invalid_pending_level", "section": section_norm}
    if pending_i <= 0:
        return {"accepted": False, "reason": "invalid_pending_level", "section": section_norm}
    if not isinstance(startup_boot_row, dict):
        return {"accepted": False, "reason": "missing_startup_row", "section": section_norm}

    hydration_status = str(startup_boot_row.get("hydration_status", "")).strip().lower()
    if hydration_status not in {"ok", "partial", "preserved_authoritative"}:
        return {
            "accepted": False,
            "reason": f"hydration_status:{hydration_status or 'unknown'}",
            "section": section_norm,
        }
    unresolved = list(startup_boot_row.get("hydration_unresolved_visible_track_ids", []) or [])
    if unresolved:
        return {
            "accepted": False,
            "reason": "unresolved_visible_tracks",
            "section": section_norm,
            "unresolved": list(unresolved),
        }

    hydrated = startup_boot_row.get("hydrated_stacks", {})
    sections = hydrated.get("sections", {}) if isinstance(hydrated, dict) else {}
    row_raw = sections.get(section_norm, {}) if isinstance(sections, dict) else {}
    row = stack_scene._normalize_hydrated_section_row(row_raw)
    level_i = int(row.get("stack_level", 0) or 0)
    if level_i < pending_i:
        return {
            "accepted": False,
            "reason": "level_below_pending",
            "section": section_norm,
            "hydrated_level": int(level_i),
            "pending_level": int(pending_i),
        }

    seq = [
        str(c).strip().lower()
        for c in list(row.get("color_sequence_bottom_to_top", []))
        if str(c).strip().lower() in {"orange", "blue", "unknown"}
    ]
    layer_idx = int(pending_i - 1)
    layer_color = str(seq[layer_idx]).strip().lower() if layer_idx < len(seq) else "unknown"
    expected_color_norm = str(expected_color or "").strip().lower()
    if expected_color_norm in {"orange", "blue"} and layer_color in {"orange", "blue"}:
        if layer_color != expected_color_norm:
            return {
                "accepted": False,
                "reason": "color_mismatch",
                "section": section_norm,
                "expected_color": expected_color_norm,
                "hydrated_color": layer_color,
            }

    entries = [dict(e) for e in list(row.get("entries", [])) if isinstance(e, dict)]
    entry = entries[layer_idx] if layer_idx < len(entries) else None
    entry_xyz = _coerce_xyz_for_place_verify((entry or {}).get("xyz", None))
    if entry_xyz is None:
        return {
            "accepted": False,
            "reason": "missing_layer_xyz",
            "section": section_norm,
            "hydrated_level": int(level_i),
            "pending_level": int(pending_i),
        }
    expected_eval = _coerce_xyz_for_place_verify(place_verify.get("expected_xyz_eval", None))
    if expected_eval is None:
        expected_eval = _coerce_xyz_for_place_verify(place_verify.get("expected_xyz", None))
    if expected_eval is None:
        return {"accepted": False, "reason": "missing_expected_xyz", "section": section_norm}

    try:
        xy_margin_m = float(place_verify.get("effective_xy_margin_m", float("nan")))
    except Exception:
        xy_margin_m = float("nan")
    if not math.isfinite(xy_margin_m):
        xy_margin_m = 0.0
    try:
        z_margin_m = float(place_verify.get("effective_z_margin_m", float("nan")))
    except Exception:
        z_margin_m = float("nan")
    if not math.isfinite(z_margin_m):
        z_margin_m = 0.0
    xy_margin_m = max(xy_margin_m, float(getattr(stack_scene, "STARTUP_STACK_LAYER_MATCH_XY_M", 0.030)))
    z_margin_m = max(z_margin_m, float(getattr(stack_scene, "STARTUP_STACK_LAYER_MATCH_Z_M", 0.025)))
    dx_m = float(entry_xyz[0]) - float(expected_eval[0])
    dy_m = float(entry_xyz[1]) - float(expected_eval[1])
    d_xy_m = float(math.hypot(dx_m, dy_m))
    d_z_m = float(abs(float(entry_xyz[2]) - float(expected_eval[2])))
    if d_xy_m > xy_margin_m:
        return {
            "accepted": False,
            "reason": "xy_out_of_margin",
            "section": section_norm,
            "measured_xyz": list(entry_xyz),
            "expected_xyz_eval": list(expected_eval),
            "xy_error_m": float(d_xy_m),
            "z_error_m": float(d_z_m),
            "xy_margin_m": float(xy_margin_m),
            "z_margin_m": float(z_margin_m),
        }
    if d_z_m > z_margin_m:
        return {
            "accepted": False,
            "reason": "z_out_of_margin",
            "section": section_norm,
            "measured_xyz": list(entry_xyz),
            "expected_xyz_eval": list(expected_eval),
            "xy_error_m": float(d_xy_m),
            "z_error_m": float(d_z_m),
            "xy_margin_m": float(xy_margin_m),
            "z_margin_m": float(z_margin_m),
        }
    return {
        "accepted": True,
        "reason": "ok",
        "section": section_norm,
        "hydrated_level": int(level_i),
        "pending_level": int(pending_i),
        "layer_index": int(layer_idx),
        "measured_xyz": list(entry_xyz),
        "expected_xyz_eval": list(expected_eval),
        "measured_color": str(layer_color),
        "xy_error_m": float(d_xy_m),
        "z_error_m": float(d_z_m),
        "dx_m": float(dx_m),
        "dy_m": float(dy_m),
        "xy_margin_m": float(xy_margin_m),
        "z_margin_m": float(z_margin_m),
    }


def _try_place_verify_hydrate_fallback(
    *,
    place_verify: dict,
    state,
    run_startup_stack_bootstrap_verify,
    section: str,
    pending_stack_level: int | None,
    expected_color: str | None,
) -> dict:
    if bool(place_verify.get("confirmed", False)):
        return dict(place_verify)
    if not bool(getattr(verify_v2, "PLACE_VERIFY_V2_HYDRATE_FALLBACK_ENABLED", True)):
        return dict(place_verify)
    print(
        f"[PlaceVerifyHydrateFallback] begin section={section} "
        f"pending_level={pending_stack_level} status={place_verify.get('status')}"
    )
    try:
        startup_boot = run_startup_stack_bootstrap_verify(mode="refresh")
    except Exception as exc:
        out = dict(place_verify)
        out["hydrate_fallback"] = {
            "accepted": False,
            "reason": f"exception:{type(exc).__name__}",
            "error": str(exc),
        }
        print(
            f"[PlaceVerifyHydrateFallback] rejected reason=exception:{type(exc).__name__} "
            f"error={exc}"
        )
        return out

    evaluation = _evaluate_place_verify_hydrate_fallback(
        startup_boot_row=startup_boot,
        section=section,
        pending_stack_level=pending_stack_level,
        expected_color=expected_color,
        place_verify=place_verify,
    )
    if not bool(evaluation.get("accepted", False)):
        out = dict(place_verify)
        out["hydrate_fallback"] = dict(evaluation)
        print(
            f"[PlaceVerifyHydrateFallback] rejected section={section} "
            f"pending_level={pending_stack_level} reason={evaluation.get('reason')}"
        )
        return out

    out = dict(place_verify)
    out["status"] = "placed_confirmed_startup_hydrate"
    out["confirmed"] = True
    out["measured_xyz"] = list(evaluation.get("measured_xyz", []))
    measured_color = str(evaluation.get("measured_color", "unknown")).strip().lower()
    if measured_color in {"orange", "blue"}:
        out["measured_color"] = str(measured_color)
        out["measured_color_conf"] = max(float(out.get("measured_color_conf", 0.0) or 0.0), 1.0)
        out["measured_color_hits"] = max(int(out.get("measured_color_hits", 0) or 0), 1)
        out["measured_color_source"] = "startup_hydrate"
    out["xy_error_m"] = float(evaluation.get("xy_error_m", out.get("xy_error_m", float("inf"))))
    out["z_error_m"] = float(evaluation.get("z_error_m", out.get("z_error_m", float("inf"))))
    out["dx_m"] = float(evaluation.get("dx_m", out.get("dx_m", float("inf"))))
    out["dy_m"] = float(evaluation.get("dy_m", out.get("dy_m", float("inf"))))
    out["effective_xy_margin_m"] = float(evaluation.get("xy_margin_m", out.get("effective_xy_margin_m", float("nan"))))
    out["effective_z_margin_m"] = float(evaluation.get("z_margin_m", out.get("effective_z_margin_m", float("nan"))))
    out["verify_exit_reason"] = "startup_hydrate_confirmed"
    out["verify_measurement_fallback_source"] = "startup_hydrate_refresh"
    out["hydrate_fallback"] = dict(evaluation)
    state.last_place_verification = dict(out)
    state.last_place_verification_v2 = dict(out)
    if state.placed_ledger:
        placement = state.placed_ledger[-1]
        placement["verify_result"] = dict(out)
        if bool(placement.get("verify_counted", False)):
            try:
                state.place_verify_uncertain_count = max(0, int(state.place_verify_uncertain_count) - 1)
            except Exception:
                pass
            try:
                state.place_verify_confirmed_count = int(state.place_verify_confirmed_count) + 1
            except Exception:
                pass
    print(
        f"[PlaceVerifyHydrateFallback] accepted section={section} "
        f"pending_level={pending_stack_level} measured={_fmt_xyz_for_log(out.get('measured_xyz'))} "
        f"err_xy={float(out.get('xy_error_m', float('inf'))):.3f} "
        f"err_z={float(out.get('z_error_m', float('inf'))):.3f}"
    )
    return out


def _open_gripper_in_place_after_current_collision(
    *,
    arm,
    release_grip: float,
    hold_s: float,
    clamp_grip_cmd_fn,
) -> float:
    release_cmd = float(clamp_grip_cmd_fn(release_grip))
    hold_s = max(0.0, float(hold_s))
    sample_time = max(0.005, float(getattr(arm, "sample_time", 0.005)))
    print(
        f"[PlaceCollisionRecover] release_in_place grip={release_cmd:.3f} "
        f"hold_s={hold_s:.2f}"
    )
    if not hasattr(arm, "tick_hold"):
        raise RuntimeError("arm_missing_tick_hold_for_place_collision_recovery")
    end_t = time.time() + hold_s
    arm.tick_hold(grip=release_cmd)
    while time.time() < end_t:
        time.sleep(min(sample_time, max(0.0, end_t - time.time())))
        arm.tick_hold(grip=release_cmd)
    return release_cmd


def handle_place_action(
    *,
    action_cmd: str,
    state,
    arm,
    det,
    per,
    hold_grip: float,
    carry_supervisor,
    centered_pos: tuple[int, int] | None,
    cube_color: str,
    color_conf: float,
    section_groups: dict[str, list[int]],
    stack_levels: dict[str, int],
    section_left_name: str,
    section_right_name: str,
    home_pose,
    place_release_open_grip: float,
    place_fail_continue_reasons: tuple[str, ...] | list[str] | set[str],
    stack_verify_correction_enabled: bool,
    stack_verify_require_confirmed_for_advance: bool,
    stack_verify_allow_downward_correction: bool,
    stack_verify_downward_require_stable_remeasure: bool,
    finite_xyz_or_none_fn,
    clamp_grip_cmd_fn,
    sync_stack_levels_from_authoritative_state,
    run_startup_stack_bootstrap_verify,
    sync_stack_levels_from_startup_bootstrap,
    log_ledger_stack_snapshot,
    run_post_lift_place_space_refresh,
    record_policy_step,
    run_observe_action,
) -> dict:
    action_cmd = str(action_cmd)
    if not action_cmd.startswith("place_"):
        return {"handled": False, "break_loop": False}

    if not state.holding_object:
        state.cycles_without_place_progress += 1
        state.invalid_precondition_recoveries += 1
        print(f"[Policy] place command '{action_cmd}' requested without object; auto-observing.")
        record_policy_step(action_cmd, "place_without_object", progress=False)
        runtime_loop_dispatch.run_auto_recovery_observe(run_observe_action=run_observe_action)
        return {"handled": True, "break_loop": False}
    place_ok, place_reason, slot_used, place_context = place_actions.execute_prompted_place_action(
        action_cmd=action_cmd,
        state=state,
        arm=arm,
        det=det,
        per=per,
        hold_grip=hold_grip,
        section_groups=section_groups,
        placed_targets=state.placed_targets,
        blocked_slots=state.blocked_slots,
        stack_levels=stack_levels,
        carry_supervisor=carry_supervisor,
    )
    if place_ok:
        ledger_pick_origin_xyz = finite_xyz_or_none_fn(state.last_pick_return_xyz)
        state.holding_object = False
        state.current_hold_grip = 0.0
        state.last_pick_return_xyz = None
        state.last_pick_measured_xyz = None
        state.active_target_track_id = None
        runtime_loop_observe.clear_pick_lock_snapshot(state=state, source="post_place_complete")
        ledger_expected_xyz = None
        ledger_stack_level = 0
        ledger_pre_obs = {}
        ledger_section = None
        pending_stack_level = None
        if place_context is not None:
            ledger_expected_xyz = finite_xyz_or_none_fn(place_context.get("expected_xyz"))
            ledger_stack_level = int(place_context.get("stack_level", 0))
            ledger_pre_obs = dict(place_context.get("pre_observation", {}))
            ledger_section = str(place_context.get("section", "")).strip().lower()
            pending_raw = place_context.get("pending_stack_level", None)
            if pending_raw is not None:
                pending_stack_level = int(pending_raw)
        if ledger_expected_xyz is None and slot_used is not None:
            ledger_expected_xyz = finite_xyz_or_none_fn(place_actions.slot_target_xyz(int(slot_used)))
        if ledger_expected_xyz is None:
            ledger_expected_xyz = [float("nan"), float("nan"), float("nan")]
        object_id = int(state.next_object_id)
        state.next_object_id += 1
        state.placed_ledger.append(
            {
                "object_id": object_id,
                "cycle": int(state.cycle_count),
                "command": str(action_cmd),
                "section": ledger_section,
                "cube_color": str(cube_color),
                "slot_index": (None if slot_used is None else int(slot_used)),
                "expected_xyz": list(ledger_expected_xyz),
                "pick_origin_xyz": (None if ledger_pick_origin_xyz is None else list(ledger_pick_origin_xyz)),
                "stack_level": int(ledger_stack_level),
                "pending_stack_level": pending_stack_level,
                "timestamp_ms": int(time.time() * 1000),
                "pre_observation": dict(ledger_pre_obs),
                "verify_result": None,
                "verify_counted": False,
                "removed_by_return": False,
                "removed_timestamp_ms": None,
                "removed_reason": "",
                "removed_command": "",
            }
        )
        if ledger_section in {section_left_name, section_right_name} and int(ledger_stack_level) == 0:
            stack_scene.commit_commanded_stack_base_anchor_from_place(
                state=state,
                section_name=str(ledger_section),
                expected_xyz=finite_xyz_or_none_fn(ledger_expected_xyz),
                placed_stack_level=int(ledger_stack_level),
            )
        place_verify = verify_v2.verify_last_place_reliability(
            state=state,
            arm=arm,
            per=per,
            det=det,
            count_in_stats=True,
        )
        if (
            pending_stack_level is not None
            and ledger_section in {section_left_name, section_right_name}
            and not bool(place_verify.get("confirmed", False))
        ):
            expected_color_for_hydrate = str(cube_color).strip().lower()
            if expected_color_for_hydrate not in {"orange", "blue"}:
                expected_color_for_hydrate = str(place_verify.get("expected_color", "")).strip().lower()
            place_verify = _try_place_verify_hydrate_fallback(
                place_verify=place_verify,
                state=state,
                run_startup_stack_bootstrap_verify=run_startup_stack_bootstrap_verify,
                section=str(ledger_section),
                pending_stack_level=pending_stack_level,
                expected_color=expected_color_for_hydrate,
            )
        if (
            pending_stack_level is not None
            and ledger_section in {section_left_name, section_right_name}
        ):
            allow_stack_advance = True
            hold_reason = "ok"
            measured_xyz = place_verify.get("measured_xyz", None)
            measured_missing = measured_xyz is None
            # If verify cannot recover a measured object, treat the layer as "lost":
            # keep stack level unchanged so the next stack command reuses this Z target.
            if measured_missing:
                allow_stack_advance = False
                hold_reason = "measured_missing_assume_lost"
            if stack_verify_correction_enabled and stack_verify_require_confirmed_for_advance:
                if allow_stack_advance:
                    allow_stack_advance = bool(place_verify.get("confirmed", False))
                    if not allow_stack_advance:
                        hold_reason = f"unconfirmed:{place_verify.get('status')}"
            if allow_stack_advance:
                stack_levels[ledger_section] = max(
                    int(stack_levels.get(ledger_section, 0)),
                    int(pending_stack_level),
                )
                print(
                    f"[StackLevel] advanced {ledger_section} -> {int(stack_levels.get(ledger_section, 0))} "
                    f"(verify_confirmed={bool(place_verify.get('confirmed', False))})"
                )
            else:
                remeasure_meta: dict = {"status": "skipped"}
                remeasured_xyz, remeasure_meta = stack_scene.remeasure_stack_xyz_until_stable(
                    arm=arm,
                    per=per,
                    det=det,
                    expected_xyz=place_verify.get("expected_xyz", None),
                    pending_stack_level=pending_stack_level,
                    section_name=ledger_section,
                    expected_color=(
                        None
                        if str(place_verify.get("expected_color", "")).strip().lower() in {"", "none", "unknown"}
                        else str(place_verify.get("expected_color", "")).strip().lower()
                    ),
                    state=state,
                )
                if remeasured_xyz is not None:
                    measured_xyz = remeasured_xyz
                current_layers = int(stack_levels.get(ledger_section, 0))
                if measured_xyz is None:
                    # No measurement means weak evidence; keep current stack level to avoid
                    # dropping target Z to table height and colliding through an existing stack.
                    inferred_layers = int(current_layers)
                else:
                    inferred_layers = stack_scene.infer_stack_layers_from_measurement(
                        measured_xyz=measured_xyz,
                        expected_xyz=place_verify.get("expected_xyz", None),
                        slot_used=slot_used,
                        current_layers=current_layers,
                    )
                if int(inferred_layers) < int(current_layers):
                    verify_status = str(place_verify.get("status", "")).strip().lower()
                    remeasure_status = str(remeasure_meta.get("status", "")).strip().lower()
                    if not stack_verify_allow_downward_correction:
                        inferred_layers = int(current_layers)
                        hold_reason = f"{hold_reason}|downward_disabled"
                    elif (
                        stack_verify_downward_require_stable_remeasure
                        and remeasure_status != "stable"
                    ):
                        inferred_layers = int(current_layers)
                        hold_reason = f"{hold_reason}|downward_requires_stable"
                    elif verify_status == "placed_mismatch_out_of_margin":
                        inferred_layers = int(current_layers)
                        hold_reason = f"{hold_reason}|mismatch_no_drop"
                stack_levels[ledger_section] = min(
                    int(stack_levels.get(ledger_section, 0)),
                    int(inferred_layers),
                )
                print(
                    _format_place_verify_hold_diag(
                        section=str(ledger_section),
                        place_verify=place_verify,
                        remeasure_meta=remeasure_meta,
                    )
                )
                print(
                    f"[StackLevel] HOLD {ledger_section} pending verify "
                    f"(target={int(pending_stack_level)} status={place_verify.get('status')} "
                    f"reason={hold_reason}"
                    f", inferred_layers={int(inferred_layers)}"
                    f", remeasure={str(remeasure_meta.get('status', 'n/a'))}"
                    f", remeasure_valid={int(remeasure_meta.get('valid', 0))})"
                )
        placement_confirmed = bool(place_verify.get("confirmed", False))
        auth_place_update = None
        if placement_confirmed:
            placed_color = str(cube_color).strip().lower()
            if placed_color not in {"orange", "blue"}:
                verify_color = str(place_verify.get("measured_color", "unknown")).strip().lower()
                if verify_color in {"orange", "blue"}:
                    placed_color = str(verify_color)
            if ledger_section in {section_left_name, section_right_name}:
                auth_place_update = stack_scene.append_authoritative_stack_cube(
                    state=state,
                    section_name=ledger_section,
                    cube_color=str(placed_color),
                )
                sync_stack_levels_from_authoritative_state()
            state.placed_count += 1
            state.cycles_without_place_progress = 0
            state.pick_other_block_track_id = None
            state.pick_other_block_uv = None
            state.pick_other_block_xyz = None
            state.pick_other_block_track_ids = []
            state.pick_other_block_xyzs = []
            state.pick_other_block_uvs = []
            state.pick_other_block_source = "none"
            if ledger_section in {section_left_name, section_right_name}:
                state.placed_counts_by_section[ledger_section] = (
                    int(state.placed_counts_by_section.get(ledger_section, 0)) + 1
                )
        else:
            sync_stack_levels_from_authoritative_state()
            state.cycles_without_place_progress += 1
            if (
                ledger_section in {section_left_name, section_right_name}
                and pending_stack_level is not None
            ):
                place_verify["requires_recovery"] = True
                place_verify["recovery_reason"] = "unconfirmed_pending_stack"
                state.last_place_verification = dict(place_verify)
                if state.placed_ledger:
                    try:
                        state.placed_ledger[-1]["verify_result"] = dict(place_verify)
                    except Exception:
                        pass
                print(
                    f"[PlaceGuard] unresolved verify section={ledger_section} "
                    f"status={place_verify.get('status')} pending_stack_level={int(pending_stack_level)} "
                    "-> blocking further place commands on this side until verify/correction resolves."
                )
            print(
                f"[Place] unverified placement; counts unchanged "
                f"(status={place_verify.get('status')}, confirmed={placement_confirmed})"
            )
        if auth_place_update is not None:
            print(f"[StackState] authoritative_place_update={auth_place_update}")
        print(f"[Place] command={action_cmd} slot={slot_used} complete. Placed={state.placed_count}")
        print(f"[PlaceVerify] status={place_verify.get('status')} confirmed={place_verify.get('confirmed')}")
        log_ledger_stack_snapshot("post_place_verify")
        post_place_side = (
            ledger_section
            if ledger_section in {section_left_name, section_right_name}
            else "all"
        )
        post_place_reconcile = stack_scene.reconcile_scene(
            state=state,
            arm=arm,
            per=per,
            det=det,
            side=str(post_place_side),
            mode="post_place_release",
            include_pick_rows=False,
        )
        print(
            f"[SceneReconcile] mode=post_place_release status={post_place_reconcile.get('status')} "
            f"drift={bool(post_place_reconcile.get('drift_detected', False))} "
            f"collision_risk={bool(post_place_reconcile.get('collision_risk', False))} "
            f"rev={int(post_place_reconcile.get('scene_revision', state.scene_revision))}"
        )
        arm.goto_task_space(home_pose, duration=1.2, label=f"prompted_cycle_{state.cycle_count}_home_end")
        place_result_tag = "place_success" if placement_confirmed else "place_unverified"
        record_policy_step(
            action_cmd,
            f"{place_result_tag}_slot_{slot_used}",
            progress=placement_confirmed,
        )
        return {
            "handled": True,
            "break_loop": True,
            "hold_grip": 0.0,
            "carry_supervisor": None,
            "centered_pos": None,
            "cube_color": "unknown",
            "color_conf": 0.0,
        }
    state.cycles_without_place_progress += 1
    print(f"[Place] failed for command '{action_cmd}' reason={place_reason}")
    fail_diag_ctx = _build_place_fail_diag(
        reason=str(place_reason),
        grip_cmd=float(hold_grip),
        arm=arm,
        state=state,
        clamp_grip_cmd_fn=clamp_grip_cmd_fn,
    )
    if place_reason == "place_collision_risk":
        fail_home = home_pose.copy()
        fail_home[3] = clamp_grip_cmd_fn(hold_grip)
        arm.goto_task_space(
            fail_home,
            duration=1.0,
            label=f"prompted_cycle_{state.cycle_count}_home_after_place_collision_risk",
        )
        record_policy_step(
            action_cmd,
            f"place_fail:{place_reason}",
            progress=False,
            feedback_context=fail_diag_ctx,
        )
        return {"handled": True, "break_loop": False}
    if place_reason == "move_overcurrent_unrecoverable":
        if bool(state.holding_object):
            try:
                release_cmd = _open_gripper_in_place_after_current_collision(
                    arm=arm,
                    release_grip=float(place_release_open_grip),
                    hold_s=float(getattr(place_actions, "PLACE_OPEN_HOLD_S", 0.22)),
                    clamp_grip_cmd_fn=clamp_grip_cmd_fn,
                )
                state.holding_object = False
                state.current_hold_grip = 0.0
                state.last_pick_return_xyz = None
                state.last_pick_measured_xyz = None
                state.active_target_track_id = None
                hold_grip = 0.0
                carry_supervisor = None
                centered_pos = None
                cube_color = "unknown"
                color_conf = 0.0
                runtime_loop_observe.clear_pick_lock_snapshot(
                    state=state,
                    source="place_current_recovery",
                )
                print("[PlaceCollisionRecover] hydrate_begin mode=refresh")
                startup_boot = run_startup_stack_bootstrap_verify(mode="refresh")
                sync_stack_levels_from_startup_bootstrap(startup_boot)
                hydrate_status = str((startup_boot or {}).get("hydrate_status", "unknown"))
                print(f"[PlaceCollisionRecover] hydrate_end status={hydrate_status}")
                recovery_ctx = dict(fail_diag_ctx)
                recovery_ctx["place_collision_recovery"] = {
                    "released_in_place": True,
                    "release_grip": float(release_cmd),
                    "hydrate_status": hydrate_status,
                }
                record_policy_step(
                    action_cmd,
                    f"place_collision_recovered:{place_reason}",
                    progress=False,
                    feedback_context=recovery_ctx,
                )
                return {
                    "handled": True,
                    "break_loop": True,
                    "hold_grip": 0.0,
                    "carry_supervisor": None,
                    "centered_pos": None,
                    "cube_color": "unknown",
                    "color_conf": 0.0,
                }
            except Exception as exc:
                print(f"[PlaceCollisionRecover] failed reason={type(exc).__name__}: {exc}")
                state.stop_reason = f"place_collision_recovery_failed:{type(exc).__name__}"
                state.skip_final_motion = True
                recovery_ctx = dict(fail_diag_ctx)
                recovery_ctx["place_collision_recovery_failed"] = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                record_policy_step(
                    action_cmd,
                    f"place_fail:{place_reason}:recovery_failed",
                    progress=False,
                    feedback_context=recovery_ctx,
                )
                return {"handled": True, "break_loop": True}
        state.stop_reason = place_reason
        state.skip_final_motion = True
        record_policy_step(
            action_cmd,
            f"place_fail:{place_reason}",
            progress=False,
            feedback_context=fail_diag_ctx,
        )
        return {"handled": True, "break_loop": True}
    if str(place_reason).strip().lower() in set(place_fail_continue_reasons):
        fail_home = home_pose.copy()
        fail_home[3] = clamp_grip_cmd_fn(place_release_open_grip)
        arm.goto_task_space(
            fail_home,
            duration=1.2,
            label=f"prompted_cycle_{state.cycle_count}_home_after_place_soft_fail",
        )
        state.holding_object = False
        state.current_hold_grip = 0.0
        state.last_pick_measured_xyz = None
        hold_grip = 0.0
        carry_supervisor = None
        centered_pos = None
        cube_color = "unknown"
        color_conf = 0.0
        state.active_target_track_id = None
        runtime_loop_observe.clear_pick_lock_snapshot(state=state, source="post_place_soft_fail")
        run_post_lift_place_space_refresh("post_place_fail_refresh")
        record_policy_step(
            action_cmd,
            f"place_fail_continue:{place_reason}",
            progress=False,
            feedback_context=fail_diag_ctx,
        )
        return {
            "handled": True,
            "break_loop": False,
            "hold_grip": float(hold_grip),
            "carry_supervisor": carry_supervisor,
            "centered_pos": centered_pos,
            "cube_color": str(cube_color),
            "color_conf": float(color_conf),
        }
    fail_home = home_pose.copy()
    fail_home[3] = clamp_grip_cmd_fn(hold_grip)
    arm.goto_task_space(
        fail_home,
        duration=1.2,
        label=f"prompted_cycle_{state.cycle_count}_home_after_place_fail",
    )
    state.stop_reason = f"place_failed:{place_reason}"
    record_policy_step(
        action_cmd,
        f"place_fail:{place_reason}",
        progress=False,
        feedback_context=fail_diag_ctx,
    )
    return {"handled": True, "break_loop": True}
