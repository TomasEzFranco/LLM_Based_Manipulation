"""Policy/log helpers extracted from runtime_loop (behavior-preserving)."""

from __future__ import annotations

import stack_scene


def summarize_policy_feedback(
    command: str,
    result: str,
    progress: bool,
    feedback_context: dict | None = None,
) -> dict:
    result_text = str(result).strip().lower()
    if bool(progress):
        outcome = "success"
    elif result_text in {"policy_stop_run", "observe_empty_scene_confirmed"}:
        outcome = "success"
    else:
        outcome = "failure"
    row = {
        "command": str(command),
        "outcome": outcome,
        "reason_code": str(result),
    }
    if isinstance(feedback_context, dict) and feedback_context:
        row.update(dict(feedback_context))
    return row


def record_policy_step(
    *,
    state,
    reobserve_streak: int,
    last_feedback: dict | None,
    command: str,
    result: str,
    progress: bool,
    feedback_context: dict | None = None,
) -> tuple[int, dict | None]:
    _ = (last_feedback,)
    state.policy_step_count += 1
    if command == "reobserve":
        reobserve_streak += 1
        state.reobserve_max_streak = max(int(state.reobserve_max_streak), int(reobserve_streak))
    else:
        reobserve_streak = 0
    last_feedback_out = summarize_policy_feedback(
        command,
        result,
        progress,
        feedback_context=feedback_context,
    )
    return int(reobserve_streak), last_feedback_out


def position_colors_from_seq(seq: list[str], level_override: int | None = None) -> dict[str, str]:
    seq_norm = [
        str(c).strip().lower()
        for c in list(seq)
        if str(c).strip().lower() in {"orange", "blue", "unknown"}
    ]
    if level_override is None:
        level = int(max(0, min(3, len(seq_norm))))
    else:
        try:
            level = int(max(0, min(3, int(level_override))))
        except Exception:
            level = int(max(0, min(3, len(seq_norm))))
    if len(seq_norm) < level:
        seq_norm = list(seq_norm) + (["unknown"] * int(level - len(seq_norm)))
    return {
        "bottom": (seq_norm[0] if level >= 1 and len(seq_norm) >= 1 else "empty"),
        "middle": (seq_norm[1] if level >= 2 and len(seq_norm) >= 2 else "empty"),
        "top": (seq_norm[2] if level >= 3 and len(seq_norm) >= 3 else "empty"),
    }


def section_seq_row_for_log(section_row: dict | None) -> tuple[int, dict[str, str]]:
    src = section_row if isinstance(section_row, dict) else {}
    try:
        level = int(src.get("stack_level", 0) or 0)
    except Exception:
        level = 0
    seq = [
        str(c).strip().lower()
        for c in list(src.get("color_sequence_bottom_to_top", []))
        if str(c).strip().lower() in {"orange", "blue", "unknown"}
    ]
    slots = src.get("slots", {}) if isinstance(src.get("slots", {}), dict) else {}
    if slots:
        base = str(slots.get("base", "empty")).strip().lower()
        middle = str(slots.get("middle", "empty")).strip().lower()
        top = str(slots.get("top", "empty")).strip().lower()
        return int(max(0, level)), {"bottom": base, "middle": middle, "top": top}
    return int(max(0, level)), position_colors_from_seq(seq, level_override=level)


def log_ledger_stack_snapshot(
    *,
    state,
    source: str,
    section_left_name: str,
    section_right_name: str,
) -> None:
    left_row = stack_scene.get_startup_hydrated_section_row(state, section_left_name)
    right_row = stack_scene.get_startup_hydrated_section_row(state, section_right_name)
    left_level, left_pos = section_seq_row_for_log(left_row)
    right_level, right_pos = section_seq_row_for_log(right_row)
    print(
        f"[StackState] source={source} "
        f"left(level={int(left_level)},base={left_pos.get('bottom')},middle={left_pos.get('middle')},top={left_pos.get('top')}) "
        f"right(level={int(right_level)},base={right_pos.get('bottom')},middle={right_pos.get('middle')},top={right_pos.get('top')}) "
        f"planner_source=authoritative_stack_state"
    )


def log_planner_stack_view(
    *,
    source: str,
    planner_state_row: dict | None,
    section_left_name: str,
    section_right_name: str,
) -> None:
    row = planner_state_row if isinstance(planner_state_row, dict) else {}
    sec = row.get("section_status", {}) if isinstance(row.get("section_status", {}), dict) else {}
    left_row = sec.get(section_left_name, {}) if isinstance(sec.get(section_left_name, {}), dict) else {}
    right_row = sec.get(section_right_name, {}) if isinstance(sec.get(section_right_name, {}), dict) else {}

    def _fmt(side_row: dict) -> str:
        try:
            level = int(side_row.get("stack_level", 0) or 0)
        except Exception:
            level = 0
        slots = side_row.get("slots", {}) if isinstance(side_row.get("slots", {}), dict) else {}
        pos = side_row.get("position_colors", {}) if isinstance(side_row.get("position_colors", {}), dict) else {}
        base = str(slots.get("base", pos.get("bottom", "empty"))).strip().lower()
        middle = str(slots.get("middle", pos.get("middle", "empty"))).strip().lower()
        top = str(slots.get("top", pos.get("top", "empty"))).strip().lower()
        return f"level={level},base={base},middle={middle},top={top}"

    print(
        f"[PlannerStack] source={source} "
        f"left({_fmt(left_row)}) right({_fmt(right_row)})"
    )
