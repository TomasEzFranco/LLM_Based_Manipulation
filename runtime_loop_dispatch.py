"""Dispatch helpers extracted from runtime_loop (behavior-preserving)."""

from __future__ import annotations


def run_auto_recovery_observe(*, run_observe_action) -> None:
    run_observe_action(
        command_for_history="observe_scene",
        clear_first=False,
        source="auto_recovery",
    )


def handle_pre_action_dispatch(
    *,
    action_cmd: str,
    state,
    pick_other_streak: int,
    log_stop_reason,
    record_policy_step,
    run_observe_action,
    run_pick_other_action,
) -> dict:
    action_cmd = str(action_cmd)
    if action_cmd == "stop_run":
        if str(getattr(state, "stop_reason", "") or "").strip() != "recurrent_grasp_failures_check_pick_space_orientation":
            state.stop_reason = "policy requested stop_run"
        log_stop_reason(state.stop_reason)
        record_policy_step(action_cmd, "policy_stop_run", progress=False)
        return {
            "handled": True,
            "break_loop": True,
            "pick_other_streak": int(pick_other_streak),
        }
    if action_cmd == "observe_scene":
        pick_other_streak = 0
        run_observe_action(command_for_history=action_cmd, clear_first=False, source="policy_observe")
        return {
            "handled": True,
            "break_loop": False,
            "pick_other_streak": int(pick_other_streak),
        }
    if action_cmd == "pick_other":
        pick_other_streak += 1
        if pick_other_streak > 2:
            state.cycles_without_place_progress += 1
            print(
                "[PolicyLoopGuard] reason_code=pick_other_streak_limit "
                f"streak={pick_other_streak} action_blocked=pick_other"
            )
            record_policy_step(action_cmd, "pick_other_streak_limited", progress=False)
            return {
                "handled": True,
                "break_loop": False,
                "pick_other_streak": int(pick_other_streak),
            }
        run_pick_other_action(command_for_history=action_cmd)
        return {
            "handled": True,
            "break_loop": False,
            "pick_other_streak": int(pick_other_streak),
        }
    pick_other_streak = 0
    if action_cmd == "pick_misplaced_cube":
        # Defensive guard: planner validation should reprompt this command away.
        print(
            "[PolicyValidate] reason_code=invalid_generic_pick_misplaced "
            "command=pick_misplaced_cube require_side_specific=true"
        )
        state.cycles_without_place_progress += 1
        record_policy_step(action_cmd, "invalid_generic_pick_misplaced", progress=False)
        return {
            "handled": True,
            "break_loop": False,
            "pick_other_streak": int(pick_other_streak),
        }
    if action_cmd == "pick_placed_cube":
        # Defensive guard: planner validation should reprompt this command away.
        print(
            "[PolicyValidate] reason_code=invalid_generic_pick_placed "
            "command=pick_placed_cube require_side_specific=true"
        )
        state.cycles_without_place_progress += 1
        record_policy_step(action_cmd, "invalid_generic_pick_placed", progress=False)
        return {
            "handled": True,
            "break_loop": False,
            "pick_other_streak": int(pick_other_streak),
        }
    return {
        "handled": False,
        "break_loop": False,
        "pick_other_streak": int(pick_other_streak),
    }
