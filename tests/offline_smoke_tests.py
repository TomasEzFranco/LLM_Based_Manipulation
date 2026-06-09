#!/usr/bin/env python3
"""Offline smoke tests for post-extraction module layout.

This suite must never touch real robot hardware, cameras, or YOLO runtime.
"""

from __future__ import annotations

import ast
import importlib
import math
import sys
import traceback
import types
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TARGET_MODULES = [
    "runtime_core",
    "runtime_loop",  # included: static inspection shows no import-time main-loop start
    "runtime_loop_startup",
    "runtime_loop_policy",
    "runtime_loop_observe",
    "runtime_loop_dispatch",
    "runtime_loop_cycle",
    "runtime_loop_actions_correction",
    "runtime_loop_actions_object",
    "runtime_loop_actions_return",
    "runtime_loop_actions_place",
    "runtime_loop_actions_router",
    "centering",
    "centering_controller",
    "verify_v2",
    "planner_io",
    "stack_scene",
    "pick_actions",
    "place_actions",
    "misplaced_actions",
    "cube_tracking",
    "projection_geometry",
    "vision_runtime",
    "current_sensing_grip_limiting",
    "llm_commander.planner.live_policy_brain",
]


HARDWARE_INIT_EVENTS: list[str] = []


def _record_hardware_init(name: str) -> None:
    HARDWARE_INIT_EVENTS.append(str(name))
    raise AssertionError(f"Hardware/YOLO constructor called during import: {name}")


def install_offline_stubs() -> None:
    """Install strict import-time stubs for hardware/runtime dependencies."""
    # Stub cv2 for environments without OpenCV. Keep all callables strict.
    if "cv2" not in sys.modules:
        cv2_mod = types.ModuleType("cv2")
        cv2_mod.WINDOW_NORMAL = 0
        cv2_mod.FONT_HERSHEY_SIMPLEX = 0
        cv2_mod.MARKER_CROSS = 0

        def _cv2_guard(*_args, **_kwargs):
            raise AssertionError("cv2 callable invoked during import/offline smoke test")

        for name in [
            "namedWindow",
            "resizeWindow",
            "imshow",
            "waitKey",
            "destroyAllWindows",
            "drawMarker",
            "circle",
            "putText",
            "line",
            "rectangle",
            "addWeighted",
            "cvtColor",
        ]:
            setattr(cv2_mod, name, _cv2_guard)
        sys.modules["cv2"] = cv2_mod

    # pal.products.qarm.QArm
    pal_mod = types.ModuleType("pal")
    pal_products_mod = types.ModuleType("pal.products")
    pal_qarm_mod = types.ModuleType("pal.products.qarm")

    class FakeQArm:
        def __init__(self, *_args, **_kwargs):
            _record_hardware_init("pal.products.qarm.QArm")

    pal_qarm_mod.QArm = FakeQArm
    sys.modules["pal"] = pal_mod
    sys.modules["pal.products"] = pal_products_mod
    sys.modules["pal.products.qarm"] = pal_qarm_mod

    # hal.products.qarm.QArmUtilities
    hal_mod = types.ModuleType("hal")
    hal_products_mod = types.ModuleType("hal.products")
    hal_qarm_mod = types.ModuleType("hal.products.qarm")

    class FakeQArmUtilities:
        def __init__(self, *_args, **_kwargs):
            _record_hardware_init("hal.products.qarm.QArmUtilities")

    hal_qarm_mod.QArmUtilities = FakeQArmUtilities
    sys.modules["hal"] = hal_mod
    sys.modules["hal.products"] = hal_products_mod
    sys.modules["hal.products.qarm"] = hal_qarm_mod

    # pyrealsense2
    rs_mod = types.ModuleType("pyrealsense2")

    class FakePipeline:
        def __init__(self, *_args, **_kwargs):
            _record_hardware_init("pyrealsense2.pipeline")

    class FakeConfig:
        def __init__(self, *_args, **_kwargs):
            _record_hardware_init("pyrealsense2.config")

    class FakeAlign:
        def __init__(self, *_args, **_kwargs):
            _record_hardware_init("pyrealsense2.align")

    class _Enum:
        depth = "depth"
        color = "color"
        z16 = "z16"
        bgr8 = "bgr8"

    rs_mod.pipeline = FakePipeline
    rs_mod.config = FakeConfig
    rs_mod.align = FakeAlign
    rs_mod.stream = _Enum
    rs_mod.format = _Enum

    def _video_profile_guard(*_args, **_kwargs):
        raise AssertionError("pyrealsense2.video_stream_profile used during import")

    rs_mod.video_stream_profile = _video_profile_guard
    sys.modules["pyrealsense2"] = rs_mod

    # ultralytics.YOLO
    ultra_mod = types.ModuleType("ultralytics")

    class FakeYOLO:
        def __init__(self, *_args, **_kwargs):
            _record_hardware_init("ultralytics.YOLO")

    ultra_mod.YOLO = FakeYOLO
    sys.modules["ultralytics"] = ultra_mod


def clear_project_modules() -> None:
    for mod_name in TARGET_MODULES:
        sys.modules.pop(mod_name, None)


def import_module_or_exc(name: str):
    try:
        module = importlib.import_module(name)
        return module, None
    except Exception as exc:  # noqa: BLE001 - explicit smoke-test capture
        return None, exc


def source_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def top_level_functions(path: Path) -> set[str]:
    module = source_tree(path)
    return {n.name for n in module.body if isinstance(n, ast.FunctionDef)}


def top_level_classes(path: Path) -> set[str]:
    module = source_tree(path)
    return {n.name for n in module.body if isinstance(n, ast.ClassDef)}


def top_level_assignments(path: Path) -> set[str]:
    module = source_tree(path)
    names: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def class_methods(path: Path, class_name: str) -> set[str]:
    module = source_tree(path)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
    return set()


def module_defined_names(path: Path) -> set[str]:
    module = source_tree(path)
    names: set[str] = set()
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname or alias.name)
    return names


def unresolved_uppercase_default_refs(path: Path) -> list[tuple[str, int, str]]:
    defined = module_defined_names(path)
    module = source_tree(path)
    issues: list[tuple[str, int, str]] = []
    for node in ast.walk(module):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        defaults = [d for d in [*node.args.defaults, *node.args.kw_defaults] if d is not None]
        for default in defaults:
            for ref in ast.walk(default):
                if not isinstance(ref, ast.Name):
                    continue
                if (not ref.id.isupper()) or (ref.id in defined):
                    continue
                issues.append((node.name, int(getattr(ref, "lineno", node.lineno)), ref.id))
    return issues


class OfflineImportSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HARDWARE_INIT_EVENTS.clear()
        install_offline_stubs()
        clear_project_modules()

    def test_import_targets_with_stubs(self):
        failures: list[tuple[str, Exception]] = []
        for mod_name in TARGET_MODULES:
            with self.subTest(module=mod_name):
                _mod, exc = import_module_or_exc(mod_name)
                if exc is not None:
                    failures.append((mod_name, exc))
                    self.fail(
                        f"Import failed for {mod_name}: {type(exc).__name__}: {exc}\n"
                        f"{traceback.format_exc()}"
                    )
        self.assertEqual([], failures)

    def test_no_hardware_constructor_on_import(self):
        self.assertEqual(
            HARDWARE_INIT_EVENTS,
            [],
            msg=f"Hardware/YOLO constructors were called: {HARDWARE_INIT_EVENTS}",
        )


class RuntimeCoreReExportTests(unittest.TestCase):
    TARGET_OWNER_MODULES = {
        "centering",
        "verify_v2",
        "planner_io",
        "stack_scene",
        "cube_tracking",
        "projection_geometry",
        "vision_runtime",
    }

    @classmethod
    def setUpClass(cls):
        install_offline_stubs()
        clear_project_modules()

    def _expected_runtime_core_import_names(self) -> list[str]:
        runtime_core_path = REPO_ROOT / "runtime_core.py"
        tree = source_tree(runtime_core_path)
        expected: list[str] = []
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module not in self.TARGET_OWNER_MODULES:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                expected.append(alias.asname or alias.name)
        return expected

    def test_runtime_core_reexports_expected_symbols(self):
        expected = self._expected_runtime_core_import_names()
        self.assertTrue(expected, "No owner-module imports discovered in runtime_core.py")
        runtime_core, exc = import_module_or_exc("runtime_core")
        if exc is not None:
            self.fail(f"runtime_core import failed: {type(exc).__name__}: {exc}")
        missing = [name for name in expected if not hasattr(runtime_core, name)]
        self.assertEqual([], missing, msg=f"Missing runtime_core re-export names: {missing}")


class PlannerContractSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_offline_stubs()
        clear_project_modules()
        cls.core, exc_core = import_module_or_exc("runtime_core")
        if exc_core is not None:
            raise AssertionError(f"runtime_core import failed for planner snapshots: {exc_core}")
        cls.planner_io, exc_planner = import_module_or_exc("planner_io")
        if exc_planner is not None:
            raise AssertionError(f"planner_io import failed for planner snapshots: {exc_planner}")

    def _empty_section_groups(self) -> dict[str, list[int]]:
        return {
            self.core.SECTION_LEFT_NAME: [],
            self.core.SECTION_RIGHT_NAME: [],
        }

    def _zero_stack_levels(self) -> dict[str, int]:
        return {
            self.core.SECTION_LEFT_NAME: 0,
            self.core.SECTION_RIGHT_NAME: 0,
        }

    def test_step_allowed_command_snapshots(self):
        state = self.core.CycleState()
        section_groups = self._empty_section_groups()
        stack_levels = self._zero_stack_levels()

        observe_allowed = self.planner_io.build_prompted_step_allowed_commands(
            state=state,
            section_groups=section_groups,
            stack_levels=stack_levels,
            centered_pos=None,
            cube_color="unknown",
            color_conf=0.0,
        )
        self.assertEqual(["stop_run", "observe_scene"], observe_allowed)

        classify_allowed = self.planner_io.build_prompted_step_allowed_commands(
            state=state,
            section_groups=section_groups,
            stack_levels=stack_levels,
            centered_pos=(320, 240),
            cube_color="unknown",
            color_conf=0.0,
        )
        self.assertEqual(["stop_run", "classify_cube", "push_cube"], classify_allowed)

        grasp_allowed = self.planner_io.build_prompted_step_allowed_commands(
            state=state,
            section_groups=section_groups,
            stack_levels=stack_levels,
            centered_pos=(320, 240),
            cube_color="blue",
            color_conf=0.99,
        )
        self.assertEqual(["stop_run", "grasp_cube", "push_cube"], grasp_allowed)

        state.holding_object = True
        state.last_pick_return_xyz = [0.10, 0.20, 0.30]
        return_allowed = self.planner_io.build_prompted_step_allowed_commands(
            state=state,
            section_groups=section_groups,
            stack_levels=stack_levels,
            centered_pos=(320, 240),
            cube_color="blue",
            color_conf=0.99,
        )
        self.assertEqual(["stop_run", "return_cube"], return_allowed)

    def test_prompted_planner_state_snapshot(self):
        state = self.core.CycleState()
        state.startup_hydrated_sections = {
            self.core.SECTION_LEFT_NAME: {
                "stack_level": 2,
                "top_color": "blue",
                "color_sequence_bottom_to_top": ["orange", "blue"],
                "tracks_bottom_to_top": [11, 12],
                "entries": [],
            },
            self.core.SECTION_RIGHT_NAME: {
                "stack_level": 1,
                "top_color": "blue",
                "color_sequence_bottom_to_top": ["blue"],
                "tracks_bottom_to_top": [21],
                "entries": [],
            },
        }
        planner_state = self.planner_io.build_prompted_planner_state(
            state=state,
            phase_name="grasp",
            holding_object=False,
            cube_color="blue",
            color_conf=1.0,
            centered_pos=(321, 242),
            stack_levels={
                self.core.SECTION_LEFT_NAME: 2,
                self.core.SECTION_RIGHT_NAME: 1,
            },
            picked_count=0,
            placed_count=0,
            scene_empty_confirmed=False,
            last_feedback=None,
        )
        left_expected = {
            "cube_count": 2,
            "slots": {"base": "orange", "middle": "blue", "top": "empty"},
        }
        right_expected = {
            "cube_count": 1,
            "slots": {"base": "blue", "middle": "empty", "top": "empty"},
        }
        if bool(getattr(self.core, "PLANNER_INCLUDE_COLOR_SEQUENCE", False)):
            left_expected["color_sequence_bottom_to_top"] = ["orange", "blue"]
            right_expected["color_sequence_bottom_to_top"] = ["blue"]

        expected = {
            "mission_prompt": self.core.MISSION_PROMPT,
            "phase": "grasp",
            "holding_object": False,
            "held_cube": None,
            "pick_target": {
                "color": "blue",
                "conf": 1.0,
                "is_centered": True,
            },
            "section_status": {
                self.core.SECTION_LEFT_NAME: left_expected,
                self.core.SECTION_RIGHT_NAME: right_expected,
            },
            "scene_empty_confirmed": False,
        }
        self.assertEqual(expected, planner_state)

    def test_prompted_planner_state_held_cube_when_carrying(self):
        state = self.core.CycleState()
        planner_state = self.planner_io.build_prompted_planner_state(
            state=state,
            phase_name="place",
            holding_object=True,
            cube_color="orange",
            color_conf=1.0,
            centered_pos=(316, 242),
            stack_levels={
                self.core.SECTION_LEFT_NAME: 1,
                self.core.SECTION_RIGHT_NAME: 1,
            },
            picked_count=1,
            placed_count=1,
            scene_empty_confirmed=False,
            last_feedback=None,
        )
        self.assertIsNone(planner_state.get("pick_target"))
        held = planner_state.get("held_cube")
        self.assertIsInstance(held, dict)
        self.assertEqual("orange", str(held.get("color")))
        self.assertAlmostEqual(1.0, float(held.get("conf")), places=4)

    def test_prompted_planner_state_normalizes_pick_target_color(self):
        state = self.core.CycleState()
        planner_state = self.planner_io.build_prompted_planner_state(
            state=state,
            phase_name="classification",
            holding_object=False,
            cube_color="Orange",
            color_conf=0.88,
            centered_pos=(320, 240),
            stack_levels={
                self.core.SECTION_LEFT_NAME: 0,
                self.core.SECTION_RIGHT_NAME: 0,
            },
            picked_count=0,
            placed_count=0,
            scene_empty_confirmed=False,
            last_feedback=None,
        )
        pick_target = planner_state.get("pick_target")
        self.assertIsInstance(pick_target, dict)
        self.assertEqual("orange", str(pick_target.get("color")))
        self.assertAlmostEqual(0.88, float(pick_target.get("conf")), places=4)

        unknown_state = self.planner_io.build_prompted_planner_state(
            state=state,
            phase_name="observe",
            holding_object=False,
            cube_color="",
            color_conf=0.0,
            centered_pos=None,
            stack_levels={
                self.core.SECTION_LEFT_NAME: 0,
                self.core.SECTION_RIGHT_NAME: 0,
            },
            picked_count=0,
            placed_count=0,
            scene_empty_confirmed=False,
            last_feedback=None,
        )
        unknown_pick = unknown_state.get("pick_target")
        self.assertEqual("unknown", str(unknown_pick.get("color")))
        self.assertEqual(0.0, float(unknown_pick.get("conf")))

    def test_pick_target_uses_pregrasp_lock_when_live_color_unknown(self):
        state = self.core.CycleState()
        state.pregrasp_pick_lock_color = "orange"
        state.pregrasp_pick_lock_color_conf = 1.0
        planner_state = self.planner_io.build_prompted_planner_state(
            state=state,
            phase_name="grasp",
            holding_object=False,
            cube_color="unknown",
            color_conf=0.0,
            centered_pos=(320, 240),
            stack_levels={
                self.core.SECTION_LEFT_NAME: 0,
                self.core.SECTION_RIGHT_NAME: 0,
            },
            picked_count=0,
            placed_count=0,
            scene_empty_confirmed=False,
            last_feedback=None,
        )
        pick_target = planner_state.get("pick_target")
        self.assertIsInstance(pick_target, dict)
        self.assertEqual("orange", str(pick_target.get("color")))
        self.assertAlmostEqual(1.0, float(pick_target.get("conf")), places=4)
        self.assertTrue(bool(pick_target.get("is_centered")))

    def test_cleared_pick_lock_requires_reclassification(self):
        observe_mod = importlib.import_module("runtime_loop_observe")
        state = self.core.CycleState()
        state.pregrasp_pick_lock_color = "orange"
        state.pregrasp_pick_lock_color_conf = 1.0

        observe_mod.clear_pick_lock_snapshot(state=state, source="unit_test")
        allowed = self.planner_io.build_prompted_step_allowed_commands(
            state=state,
            section_groups=self._empty_section_groups(),
            stack_levels=self._zero_stack_levels(),
            centered_pos=(320, 240),
            cube_color="unknown",
            color_conf=0.0,
        )

        self.assertEqual(["stop_run", "classify_cube", "push_cube"], allowed)
        planner_state = self.planner_io.build_prompted_planner_state(
            state=state,
            phase_name="classification",
            holding_object=False,
            cube_color="unknown",
            color_conf=0.0,
            centered_pos=(320, 240),
            stack_levels=self._zero_stack_levels(),
            picked_count=0,
            placed_count=0,
            scene_empty_confirmed=False,
            last_feedback=None,
        )
        pick_target = planner_state.get("pick_target")
        self.assertEqual("unknown", str(pick_target.get("color")))

    def test_visible_target_observe_retry_does_not_increment_observe_fail_streak(self):
        observe_mod = importlib.import_module("runtime_loop_observe")
        state = self.core.CycleState()
        records = []
        original_run_pick_center_cycle = observe_mod.pick_actions.run_pick_center_cycle

        def fake_run_pick_center_cycle(**_kwargs):
            state.last_center_failure = {
                "status": "active_detection_timeout",
                "candidate_count": 3,
                "filtered_count": 3,
                "selector_meta": {"eligible_count": 2},
            }
            return "retry", None

        observe_mod.pick_actions.run_pick_center_cycle = fake_run_pick_center_cycle
        try:
            centered_pos, cube_color, color_conf, observe_fail_streak = observe_mod.run_observe_action(
                command_for_history="observe_scene",
                clear_first=False,
                source="policy_observe",
                state=state,
                arm=None,
                per=None,
                det=None,
                section_groups=self._empty_section_groups(),
                cycle_count=1,
                centered_pos=None,
                cube_color="unknown",
                color_conf=0.0,
                observe_fail_streak=1,
                observe_fail_stop_after=2,
                record_policy_step=lambda *args, **kwargs: records.append((args, kwargs)),
            )
        finally:
            observe_mod.pick_actions.run_pick_center_cycle = original_run_pick_center_cycle

        self.assertIsNone(centered_pos)
        self.assertEqual("unknown", cube_color)
        self.assertEqual(0.0, float(color_conf))
        self.assertEqual(1, int(observe_fail_streak))
        self.assertEqual(0, int(state.no_pick_miss_count))
        self.assertEqual([(("observe_scene", "observe_retry"), {"progress": False})], records)

    def test_observe_clears_pick_other_blocks_before_centering(self):
        observe_mod = importlib.import_module("runtime_loop_observe")
        state = self.core.CycleState()
        state.pick_other_block_track_id = 19
        state.pick_other_block_xyz = [0.338, -0.160, 0.054]
        state.pick_other_block_uv = [317, 239]
        state.pick_other_block_track_ids = [23, 40]
        state.pick_other_block_xyzs = [[0.346, -0.277, 0.059]]
        state.pick_other_block_uvs = [[315, 244]]
        state.pick_other_block_source = "classify"
        captured: dict = {}
        original_run_pick_center_cycle = observe_mod.pick_actions.run_pick_center_cycle

        def fake_run_pick_center_cycle(**kwargs):
            captured["blocked_track_ids"] = set(kwargs.get("blocked_track_ids") or set())
            captured["blocked_xyzs"] = list(kwargs.get("blocked_xyzs") or [])
            captured["blocked_uvs"] = list(kwargs.get("blocked_uvs") or [])
            state.last_center_failure = {
                "status": "active_detection_timeout",
                "candidate_count": 3,
                "filtered_count": 3,
                "selector_meta": {"eligible_count": 0},
            }
            return "retry", None

        observe_mod.pick_actions.run_pick_center_cycle = fake_run_pick_center_cycle
        try:
            observe_mod.run_observe_action(
                command_for_history="observe_scene",
                clear_first=False,
                source="policy_observe",
                state=state,
                arm=None,
                per=None,
                det=None,
                section_groups=self._empty_section_groups(),
                cycle_count=1,
                centered_pos=None,
                cube_color="unknown",
                color_conf=0.0,
                observe_fail_streak=0,
                observe_fail_stop_after=2,
                record_policy_step=lambda *_args, **_kwargs: None,
            )
        finally:
            observe_mod.pick_actions.run_pick_center_cycle = original_run_pick_center_cycle

        self.assertEqual(set(), captured.get("blocked_track_ids"))
        self.assertEqual([], captured.get("blocked_xyzs"))
        self.assertEqual([], captured.get("blocked_uvs"))
        self.assertIsNone(state.pick_other_block_track_id)
        self.assertIsNone(state.pick_other_block_xyz)
        self.assertIsNone(state.pick_other_block_uv)
        self.assertEqual([], state.pick_other_block_track_ids)
        self.assertEqual([], state.pick_other_block_xyzs)
        self.assertEqual([], state.pick_other_block_uvs)
        self.assertEqual("none", str(state.pick_other_block_source))

    def test_failed_grasp_visible_target_miss_warns_and_retries(self):
        pick_actions = importlib.import_module("pick_actions")
        state = self.core.CycleState()
        state.recent_grasp_fail_count = 1
        state.recent_pick_active_miss_count = 0
        state.last_grasp_fail_reason = "approach_failed"
        state.last_picked_track_id = 20
        state.last_picked_uv = [320, 240]
        state.pick_other_block_track_id = 20
        state.pick_other_block_xyz = [0.36, -0.12, 0.058]

        class FakeArm:
            def __init__(self):
                self.goto_calls = []

            def goto_task_space(self, *args, **kwargs):
                self.goto_calls.append((args, kwargs))
                return True

        original_acquire = pick_actions.acquire_and_center_intended_cube

        def fake_acquire_and_center_intended_cube(**_kwargs):
            state.last_center_failure = {
                "status": "active_detection_timeout",
                "candidate_count": 4,
                "filtered_count": 4,
                "selector_meta": {
                    "eligible_count": 1,
                    "selected_track_id": 21,
                    "selected_uv": [488, 345],
                },
            }
            return "retry", None, 21

        pick_actions.acquire_and_center_intended_cube = fake_acquire_and_center_intended_cube
        try:
            status, centered_pos = pick_actions.run_pick_center_cycle(
                state=state,
                arm=FakeArm(),
                per=None,
                det=None,
                label_prefix="unit_test",
                section_groups=self._empty_section_groups(),
            )
        finally:
            pick_actions.acquire_and_center_intended_cube = original_acquire

        self.assertEqual("retry", status)
        self.assertIsNone(centered_pos)
        self.assertEqual("completed", str(state.stop_reason))
        self.assertEqual(1, int(state.recent_pick_active_miss_count))
        self.assertEqual(1, int(state.cycles_without_place_progress))

    def test_pick_other_handoff_no_candidate_timeout_default_is_five_seconds(self):
        self.assertAlmostEqual(5.0, float(self.core.TRACK_HANDOFF_NO_CANDIDATE_TIMEOUT_S), places=3)

    def test_policy_validate_accepts_json_without_confidence(self):
        brain_mod = importlib.import_module("llm_commander.planner.live_policy_brain")
        cmd, reason, conf, _, _ = brain_mod.LivePolicyBrain._validate_payload(
            {"command": "observe_scene", "reason": "Find next cube"},
            {"observe_scene"},
        )
        self.assertEqual("observe_scene", cmd)
        self.assertEqual("Find next cube", reason)
        self.assertIsNone(conf)

    def test_golden_phase_gate_trace(self):
        state = self.core.CycleState()
        section_groups = self._empty_section_groups()
        stack_levels = self._zero_stack_levels()

        trace: list[list[str]] = []
        trace.append(
            self.planner_io.build_prompted_step_allowed_commands(
                state=state,
                section_groups=section_groups,
                stack_levels=stack_levels,
                centered_pos=None,
                cube_color="unknown",
                color_conf=0.0,
            )
        )
        trace.append(
            self.planner_io.build_prompted_step_allowed_commands(
                state=state,
                section_groups=section_groups,
                stack_levels=stack_levels,
                centered_pos=(320, 240),
                cube_color="unknown",
                color_conf=0.0,
            )
        )
        trace.append(
            self.planner_io.build_prompted_step_allowed_commands(
                state=state,
                section_groups=section_groups,
                stack_levels=stack_levels,
                centered_pos=(320, 240),
                cube_color="blue",
                color_conf=0.95,
            )
        )
        state.holding_object = True
        state.last_pick_return_xyz = [0.1, 0.2, 0.3]
        trace.append(
            self.planner_io.build_prompted_step_allowed_commands(
                state=state,
                section_groups=section_groups,
                stack_levels=stack_levels,
                centered_pos=(320, 240),
                cube_color="blue",
                color_conf=0.95,
            )
        )

        expected_trace = [
            ["stop_run", "observe_scene"],
            ["stop_run", "classify_cube", "push_cube"],
            ["stop_run", "grasp_cube", "push_cube"],
            ["stop_run", "return_cube"],
        ]
        self.assertEqual(expected_trace, trace)

    def test_pick_placed_allowed_in_classification_phase(self):
        runtime_loop_cycle = importlib.import_module("runtime_loop_cycle")
        state = self.core.CycleState()
        state.startup_hydrated_sections = {
            self.core.SECTION_RIGHT_NAME: {
                "stack_level": 2,
                "top_color": "orange",
                "color_sequence_bottom_to_top": ["blue", "orange"],
                "tracks_bottom_to_top": [1, 2],
                "entries": [],
            },
            self.core.SECTION_LEFT_NAME: {
                "stack_level": 0,
                "top_color": "unknown",
                "color_sequence_bottom_to_top": [],
                "tracks_bottom_to_top": [],
                "entries": [],
            },
        }
        stack_levels = {
            self.core.SECTION_LEFT_NAME: 0,
            self.core.SECTION_RIGHT_NAME: 2,
        }
        row = runtime_loop_cycle.compute_phase_and_allowed_commands(
            state=state,
            step_index=2,
            section_groups=self._empty_section_groups(),
            stack_levels=stack_levels,
            centered_pos=(320, 240),
            cube_color="unknown",
            color_conf=0.0,
            observe_fail_streak=0,
            observe_fail_stop_after=999,
            empty_scene_confirm_passes=3,
            max_stack_levels_per_section=3,
            section_left_name=self.core.SECTION_LEFT_NAME,
            section_right_name=self.core.SECTION_RIGHT_NAME,
            planner_io_module=self.planner_io,
            policy_log_allowed_commands=False,
        )
        self.assertEqual("classification", str(row.get("phase_name")))
        allowed = list(row.get("allowed_commands", []))
        self.assertIn("pick_placed_right", allowed)
        removed = [str(x) for x in list(row.get("removed_by_sanity", []))]
        self.assertFalse(any("pick_placed_right" in x and "removed" in x for x in removed))


class ImportSafetyStaticChecks(unittest.TestCase):
    EXTRACTED_MODULES = [
        "runtime_loop_startup.py",
        "runtime_loop_policy.py",
        "runtime_loop_observe.py",
        "runtime_loop_dispatch.py",
        "runtime_loop_cycle.py",
        "runtime_loop_actions_correction.py",
        "runtime_loop_actions_object.py",
        "runtime_loop_actions_return.py",
        "runtime_loop_actions_place.py",
        "runtime_loop_actions_router.py",
        "centering.py",
        "centering_controller.py",
        "verify_v2.py",
        "planner_io.py",
        "stack_scene.py",
        "pick_actions.py",
        "place_actions.py",
        "misplaced_actions.py",
        "cube_tracking.py",
        "projection_geometry.py",
        "vision_runtime.py",
    ]

    def test_no_unresolved_uppercase_default_refs(self):
        failures: list[str] = []
        for rel in self.EXTRACTED_MODULES:
            path = REPO_ROOT / rel
            issues = unresolved_uppercase_default_refs(path)
            if not issues:
                continue
            failures.append(
                f"{rel}: "
                + ", ".join([f"{fn}@{line} -> {name}" for fn, line, name in issues])
            )
        self.assertEqual(
            [],
            failures,
            msg=(
                "Found import-order-coupled uppercase defaults in extracted modules. "
                "Use None defaults + in-body resolution after _bind_core_globals(). "
                f"Issues: {failures}"
            ),
        )


class DuplicateOwnerStaticChecks(unittest.TestCase):
    def test_centering_functions_not_duplicated_in_runtime_core(self):
        runtime_core_defs = top_level_functions(REPO_ROOT / "runtime_core.py")
        owner_defs = top_level_functions(REPO_ROOT / "centering.py")
        promoted = {
            "_draw_center_reference_overlay",
            "_draw_center_stability_overlay",
            "_draw_forbidden_uv_overlay",
            "_show_center_frame",
            "_handle_selected_center_candidate",
            "center_object_slowly",
            "center_object_on_expected_slot",
        }
        self.assertTrue(promoted.issubset(owner_defs))
        self.assertTrue(promoted.isdisjoint(runtime_core_defs))

    def test_centering_constants_owned_by_centering_controller(self):
        runtime_core_assign = top_level_assignments(REPO_ROOT / "runtime_core.py")
        owner_assign = top_level_assignments(REPO_ROOT / "centering_controller.py")
        constants = {
            "KYAW",
            "KSHOULDER",
            "KELBOW",
            "CENTER_EY_KI",
            "CENTER_EY_I_CLAMP",
            "CENTER_EY_I_DECAY",
            "CENTER_EY_I_ENABLE_ABS_PX",
            "CENTER_EY_I_DOWN_SCALE",
            "CENTER_EY_DOWN_DPHI_SCALE",
            "CENTER_EY_DOWN_SIGN",
            "MAX_JOINT_NUDGE",
        }
        self.assertTrue(constants.issubset(owner_assign))
        self.assertTrue(constants.isdisjoint(runtime_core_assign))

    def test_verify_v2_functions_not_duplicated_in_runtime_core(self):
        runtime_core_defs = top_level_functions(REPO_ROOT / "runtime_core.py")
        owner_defs = top_level_functions(REPO_ROOT / "verify_v2.py")
        promoted = {
            "_filter_verify_candidates",
            "compute_verify_stack_min_z",
            "compute_verify_z_margin",
            "_filter_projected_slot_candidates",
            "collect_slot_observations",
            "associate_newest_placement",
            "score_place_geometry",
            "verify_last_place_reliability",
        }
        self.assertTrue(promoted.issubset(owner_defs))
        self.assertTrue(promoted.isdisjoint(runtime_core_defs))

    def test_planner_io_functions_not_duplicated_in_runtime_core(self):
        runtime_core_defs = top_level_functions(REPO_ROOT / "runtime_core.py")
        owner_defs = top_level_functions(REPO_ROOT / "planner_io.py")
        promoted = {
            "build_live_policy_brain",
            "build_prompted_allowed_commands",
            "build_prompted_step_allowed_commands",
            "maybe_append_policy_raw_row",
            "build_prompted_planner_state",
        }
        self.assertTrue(promoted.issubset(owner_defs))
        self.assertTrue(promoted.isdisjoint(runtime_core_defs))

    def test_stack_scene_remeasure_functions_not_duplicated_in_runtime_core(self):
        runtime_core_defs = top_level_functions(REPO_ROOT / "runtime_core.py")
        owner_defs = top_level_functions(REPO_ROOT / "stack_scene.py")
        promoted = {
            "_extract_valid_z",
            "remeasure_stack_xyz_after_center",
            "remeasure_stack_xyz_until_stable",
            "infer_stack_layers_from_measurement",
        }
        self.assertTrue(promoted.issubset(owner_defs))
        self.assertTrue(promoted.isdisjoint(runtime_core_defs))

    def test_tracking_helpers_owned_by_cube_tracking(self):
        runtime_core_defs = top_level_functions(REPO_ROOT / "runtime_core.py")
        owner_defs = top_level_functions(REPO_ROOT / "cube_tracking.py")
        promoted = {
            "update_cube_tracks",
            "select_intended_track_for_pick",
            "nearest_visible_track_by_uv",
            "estimate_visible_section_counts_from_tracks",
        }
        self.assertTrue(promoted.issubset(owner_defs))
        self.assertTrue(promoted.isdisjoint(runtime_core_defs))

    def test_projection_helpers_owned_by_projection_geometry(self):
        runtime_core_defs = top_level_functions(REPO_ROOT / "runtime_core.py")
        owner_defs = top_level_functions(REPO_ROOT / "projection_geometry.py")
        promoted = {
            "base_to_camera_T",
            "uvz_to_xyz_cam",
            "robust_depth_m",
            "estimate_base_xyz_from_uv_fast",
            "project_candidates_to_base",
            "get_cam_offsets",
            "set_cam_offsets",
        }
        self.assertTrue(promoted.issubset(owner_defs))
        self.assertTrue(promoted.isdisjoint(runtime_core_defs))

    def test_vision_helpers_owned_by_vision_runtime(self):
        runtime_core_defs = top_level_functions(REPO_ROOT / "runtime_core.py")
        runtime_core_classes = top_level_classes(REPO_ROOT / "runtime_core.py")
        vision_classes = top_level_classes(REPO_ROOT / "vision_runtime.py")
        yolo_methods = class_methods(REPO_ROOT / "vision_runtime.py", "YOLODetector")

        self.assertIn("YOLODetector", vision_classes)
        self.assertNotIn("YOLODetector", runtime_core_classes)
        self.assertNotIn("detect_candidates_and_draw", runtime_core_defs)
        self.assertIn("detect_candidates_and_draw", yolo_methods)


class PureHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_offline_stubs()

    def test_uvz_to_xyz_cam(self):
        projection_geometry = importlib.import_module("projection_geometry")

        class FakeIntr:
            ppx = 320.0
            ppy = 240.0
            fx = 640.0
            fy = 640.0

        x, y, z = projection_geometry.uvz_to_xyz_cam(320, 240, 1.0, FakeIntr())
        self.assertAlmostEqual(0.0, x, places=7)
        self.assertAlmostEqual(0.0, y, places=7)
        self.assertAlmostEqual(1.0, z, places=7)

    def test_robust_depth_m(self):
        projection_geometry = importlib.import_module("projection_geometry")

        depth = np.array(
            [
                [0, 1000, 1000, 1000, 0],
                [1000, 1000, 2000, 1000, 1000],
                [1000, 2000, 3000, 2000, 1000],
                [1000, 1000, 2000, 1000, 1000],
                [0, 1000, 1000, 1000, 0],
            ],
            dtype=np.uint16,
        )

        class FakeDepthFrame:
            def get_width(self):
                return int(depth.shape[1])

            def get_height(self):
                return int(depth.shape[0])

            def get_data(self):
                return depth

        out = projection_geometry.robust_depth_m(
            depth_frame=FakeDepthFrame(),
            u=2,
            v=2,
            depth_scale=0.001,
            win=3,
            percentile=50,
        )
        self.assertTrue(np.isfinite(out))
        self.assertGreater(out, 0.0)
        self.assertLess(out, 3.1)

    def test_update_cube_tracks_with_fake_runtime_core(self):
        cube_tracking = importlib.import_module("cube_tracking")
        orig_runtime_core = sys.modules.get("runtime_core")
        fake_core = types.ModuleType("runtime_core")
        fake_core.TRACK_MAX_MISS_FRAMES = 2
        fake_core.TRACK_ENABLE = True
        fake_core.SECTION_LEFT_NAME = "left"
        fake_core.SECTION_RIGHT_NAME = "right"
        fake_core.TRACK_WARN_MISSING_IDS = False
        fake_core.TRACK_WARN_MISSING_IDS_INTERVAL_S = 0.0
        fake_core.PICK_MAX_BASE_Y_M = 0.10
        fake_core.get_place_slots = lambda: [np.array([0.0, 0.20, 0.0]), np.array([0.0, 0.40, 0.0])]
        fake_core.section_slot_groups = lambda _slots: {"left": [0], "right": [1]}

        def _finite_xyz_or_none(xyz):
            arr = np.array(xyz, dtype=float).reshape(-1)
            if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
                return None
            return [float(arr[0]), float(arr[1]), float(arr[2])]

        fake_core._finite_xyz_or_none = _finite_xyz_or_none

        sys.modules["runtime_core"] = fake_core
        try:
            class FakeState:
                def __init__(self):
                    self.track_memory = {}
                    self.active_target_track_id = None
                    self.last_track_snapshot = {}
                    self.track_untracked_frames = 0
                    self.track_untracked_detections_total = 0
                    self.track_last_warn_ms = 0

            state = FakeState()
            detections = [
                {
                    "track_id": 7,
                    "u": 321,
                    "v": 239,
                    "xyz": [0.25, 0.30, 0.06],
                    "conf": 0.93,
                    "name": "Cube",
                    "cls": 0,
                    "bbox_xyxy": [300, 200, 340, 260],
                    "is_tracked": True,
                }
            ]
            tracks = cube_tracking.update_cube_tracks(
                state=state,
                detections=detections,
                max_miss_frames=2,
                now_ms=123456789,
                image_center_uv=(320, 240),
            )
            self.assertIn(7, tracks)
            self.assertEqual(1, int(state.last_track_snapshot.get("visible_track_count", 0)))
            self.assertEqual(1, int(state.last_track_snapshot["visible_section_counts"]["left"]) + int(state.last_track_snapshot["visible_section_counts"]["right"]))
        finally:
            if orig_runtime_core is None:
                sys.modules.pop("runtime_core", None)
            else:
                sys.modules["runtime_core"] = orig_runtime_core

    def test_centering_nudge_math(self):
        centering_controller = importlib.import_module("centering_controller")
        centering_controller._reset_centering_integrator()
        dphi = centering_controller._compute_centering_nudge(ex=18, ey=12)
        self.assertEqual((4,), tuple(dphi.shape))
        self.assertTrue(np.all(np.isfinite(dphi)))
        self.assertLessEqual(float(np.max(np.abs(dphi))), float(centering_controller.MAX_JOINT_NUDGE) + 1e-12)

    def test_centered_frames_required_default_is_three(self):
        core = importlib.import_module("runtime_core")
        self.assertEqual(3, int(core.CENTERED_FRAMES_REQUIRED))

    def test_stack_anchor_last_popped_without_verify(self):
        stack_scene = importlib.import_module("stack_scene")
        core = importlib.import_module("runtime_core")
        state = core.CycleState()
        state.startup_hydrated_sections = {
            core.SECTION_LEFT_NAME: {
                "stack_level": 1,
                "top_color": "blue",
                "color_sequence_bottom_to_top": ["blue"],
                "tracks_bottom_to_top": [1],
                "entries": [],
            },
            core.SECTION_RIGHT_NAME: stack_scene._startup_default_hydrated_section_row(),
        }
        state.last_popped_left_xy = [0.42, -0.14]
        anchor_xyz, anchor_source = stack_scene.get_latest_side_stack_anchor_xyz(
            state,
            core.SECTION_LEFT_NAME,
        )
        self.assertIsNotNone(anchor_xyz)
        self.assertEqual("last_popped_xy", str(anchor_source))
        self.assertAlmostEqual(0.42, float(anchor_xyz[0]), places=4)
        self.assertAlmostEqual(-0.14, float(anchor_xyz[1]), places=4)

    def test_stack_anchor_x_comp_skips_commanded_base_only(self):
        place_actions = importlib.import_module("place_actions")
        self.assertFalse(
            place_actions._stack_anchor_x_comp_allowed("commanded_place_base_level0")
        )
        self.assertTrue(place_actions._stack_anchor_x_comp_allowed("startup_hydrate_top"))
        self.assertTrue(place_actions._stack_anchor_x_comp_allowed("last_popped_xy"))
        self.assertTrue(place_actions._stack_anchor_x_comp_allowed(None))

    def test_stack_anchor_x_comp_defaults_keep_top_from_verify_anchor_overpull(self):
        core = importlib.import_module("runtime_core")

        self.assertTrue(bool(core.STACK_ANCHOR_X_COMP_ENABLED))
        self.assertAlmostEqual(0.000, float(core.STACK_ANCHOR_X_COMP_LEVEL1_M), places=4)
        self.assertAlmostEqual(-0.005, float(core.STACK_ANCHOR_X_COMP_M), places=4)

    def test_stack_anchor_commanded_level0_ignores_upper_verify_measured(self):
        stack_scene = importlib.import_module("stack_scene")
        core = importlib.import_module("runtime_core")
        state = core.CycleState()
        state.placed_ledger = [
            {
                "object_id": 1,
                "cycle": 1,
                "command": "place_right",
                "section": core.SECTION_RIGHT_NAME,
                "cube_color": "orange",
                "slot_index": 0,
                "expected_xyz": [0.467, 0.078, 0.05],
                "stack_level": 0,
                "verify_result": {
                    "confirmed": True,
                    "active": True,
                    "measured_xyz": [0.467, 0.078, 0.05],
                },
                "removed_by_return": False,
            },
            {
                "object_id": 2,
                "cycle": 3,
                "command": "place_right_stack",
                "section": core.SECTION_RIGHT_NAME,
                "cube_color": "orange",
                "slot_index": 0,
                "expected_xyz": [0.467, 0.078, 0.11],
                "stack_level": 1,
                "verify_result": {
                    "confirmed": True,
                    "active": True,
                    "measured_xyz": [0.425, 0.067, 0.096],
                },
                "removed_by_return": False,
            },
        ]
        anchor_xyz, anchor_source = stack_scene.get_latest_side_stack_anchor_xyz(
            state,
            core.SECTION_RIGHT_NAME,
        )
        self.assertIsNotNone(anchor_xyz)
        self.assertEqual("commanded_place_base_level0", str(anchor_source))
        self.assertAlmostEqual(0.467, float(anchor_xyz[0]), places=4)
        self.assertAlmostEqual(0.078, float(anchor_xyz[1]), places=4)
        anchor_xyz2, anchor_source2 = stack_scene.get_latest_side_stack_anchor_xyz(
            state,
            core.SECTION_RIGHT_NAME,
        )
        self.assertAlmostEqual(float(anchor_xyz[0]), float(anchor_xyz2[0]), places=4)
        self.assertAlmostEqual(float(anchor_xyz[1]), float(anchor_xyz2[1]), places=4)
        self.assertEqual(str(anchor_source), str(anchor_source2))

    def test_stack_anchor_first_place_left_stack_ledger_level0(self):
        stack_scene = importlib.import_module("stack_scene")
        core = importlib.import_module("runtime_core")
        state = core.CycleState()
        state.placed_ledger = [
            {
                "object_id": 1,
                "cycle": 1,
                "command": "place_left_stack",
                "section": core.SECTION_LEFT_NAME,
                "cube_color": "blue",
                "slot_index": 1,
                "expected_xyz": [0.460, 0.276, 0.045],
                "stack_level": 0,
                "pending_stack_level": 1,
                "removed_by_return": False,
            },
        ]
        anchor_xyz, anchor_source = stack_scene.get_latest_side_stack_anchor_xyz(
            state,
            core.SECTION_LEFT_NAME,
        )
        self.assertIsNotNone(anchor_xyz)
        self.assertEqual("commanded_place_base_level0", str(anchor_source))
        self.assertAlmostEqual(0.460, float(anchor_xyz[0]), places=4)
        self.assertAlmostEqual(0.276, float(anchor_xyz[1]), places=4)

    def test_startup_vote_hits_for_layer_slot(self):
        stack_scene = importlib.import_module("stack_scene")
        core = importlib.import_module("runtime_core")
        dz = float(core.STACK_LEVEL_DZ_M)
        base_z = 0.06
        anchor = (0.46, 0.12)
        layer_xy = float(core.STARTUP_STACK_LAYER_MATCH_XY_M)
        layer_z = float(core.STARTUP_STACK_LAYER_MATCH_Z_M)
        burst = [
            {
                "prow": {"track_id": 10, "u": 300, "v": 200},
                "obs": None,
                "xyz": np.array([anchor[0], anchor[1], base_z + 2 * dz], dtype=float),
            },
            {
                "prow": {"track_id": 11, "u": 301, "v": 210},
                "obs": None,
                "xyz": np.array([anchor[0], anchor[1], base_z + dz], dtype=float),
            },
            {
                "prow": {"track_id": 12, "u": 302, "v": 220},
                "obs": None,
                "xyz": np.array([anchor[0], anchor[1], base_z], dtype=float),
            },
        ]
        pred_top = np.array([anchor[0], anchor[1], base_z + 2 * dz], dtype=float)
        pred_mid = np.array([anchor[0], anchor[1], base_z + dz], dtype=float)
        top_hits = stack_scene._startup_vote_hits_for_layer_slot(
            burst,
            predicted_xyz=pred_top,
            layer_xy_m=layer_xy,
            layer_z_m=layer_z,
        )
        mid_hits = stack_scene._startup_vote_hits_for_layer_slot(
            burst,
            predicted_xyz=pred_mid,
            layer_xy_m=layer_xy,
            layer_z_m=layer_z,
            excluded_track_ids={10},
        )
        self.assertEqual(1, len(top_hits))
        self.assertEqual(10, int(top_hits[0]["prow"]["track_id"]))
        self.assertEqual(1, len(mid_hits))
        self.assertEqual(11, int(mid_hits[0]["prow"]["track_id"]))

    def test_startup_side_full_rescan_defaults(self):
        core = importlib.import_module("runtime_core")
        self.assertFalse(bool(core.STARTUP_STACK_SIDE_FULL_RESCAN_ENABLED))
        self.assertGreaterEqual(int(core.STARTUP_STACK_SIDE_FULL_RESCAN_MIN_EXPECTED), 3)
        self.assertGreaterEqual(int(core.STARTUP_STACK_SIDE_FULL_RESCAN_FRAMES), 1)

    def test_startup_hydrate_sides_at_cap(self):
        stack_scene = importlib.import_module("stack_scene")
        core = importlib.import_module("runtime_core")
        cap = int(core.STARTUP_STACK_MAX_CUBES_PER_SIDE)
        full = {
            core.SECTION_LEFT_NAME: [{"track_id": i} for i in range(cap)],
            core.SECTION_RIGHT_NAME: [{"track_id": i} for i in range(cap)],
        }
        partial = {
            core.SECTION_LEFT_NAME: [{"track_id": 1}] * cap,
            core.SECTION_RIGHT_NAME: [{"track_id": 2}] * max(0, cap - 1),
        }
        self.assertTrue(stack_scene._startup_hydrate_sides_at_cap(full))
        self.assertFalse(stack_scene._startup_hydrate_sides_at_cap(partial))

    def test_pick_placed_handoff_defaults_and_pixel_rank(self):
        core = importlib.import_module("runtime_core")
        misplaced = importlib.import_module("misplaced_actions")
        self.assertFalse(bool(core.PICK_PLACED_HANDOFF_SECTION_HARD_FILTER))
        self.assertTrue(bool(core.MISPLACED_PICK_ENFORCE_PICK_SPACE_GATE))
        self.assertEqual(1, int(core.PICK_PLACED_VERIFY_STRIKES))
        misplaced_src = (REPO_ROOT / "misplaced_actions.py").read_text(encoding="utf-8")
        self.assertIn("correction_pick_min_y_m = 0.0", misplaced_src)
        self.assertIn("reject_below_base_y_m=(", misplaced_src)
        cx = 320
        cy = 240
        candidates = [
            {"track_id": 10, "u": 420, "v": 400, "conf": 0.95},
            {"track_id": 11, "u": 180, "v": 350, "conf": 0.90},
        ]
        projected = [
            {"u": 420, "v": 400, "xyz": [0.45, 0.12, 0.12]},
            {"u": 180, "v": 350, "xyz": [0.45, 0.29, 0.12]},
        ]
        picked = misplaced.rank_misplaced_handoff_track_candidate(
            candidates,
            projected,
            cx=int(cx),
            cy=int(cy),
            min_conf=0.5,
            preferred_section_norm=str(core.SECTION_LEFT_NAME),
            enforce_preferred_section_hard_filter=False,
            section_centers_xy={
                str(core.SECTION_LEFT_NAME): (0.45, 0.29),
                str(core.SECTION_RIGHT_NAME): (0.45, 0.13),
            },
        )
        self.assertIsNotNone(picked)
        self.assertEqual(11, int(picked["track_id"]))
        self.assertEqual("pixel", str(picked.get("side_pref_source", "")))

    def test_phase2_correction_and_carry_defaults(self):
        core = importlib.import_module("runtime_core")
        self.assertTrue(bool(core.CORRECTION_RETREAT_HOME_ENABLED))
        self.assertFalse(bool(core.CORRECTION_DROP_TRANSIT_ENABLED))
        self.assertTrue(bool(core.POST_LIFT_CARRY_BLEED_MOVE_TIME))
        self.assertEqual(2, int(core.PICK_PLACED_SOLE_TRACK_RETRIES))
        self.assertGreaterEqual(float(core.PICK_PLACED_LOCK_TIMEOUT_S), float(core.MISPLACED_PICK_HARD_TIMEOUT_S))
        self.assertGreaterEqual(
            float(core.PICK_PLACED_MAX_TOTAL_LOCK_TIME_S),
            float(core.MISPLACED_PICK_MAX_TOTAL_LOCK_TIME_S),
        )
        self.assertTrue(core.carry_bleed_should_apply_for_label("prompted_cycle_1_carry"))
        self.assertFalse(core.carry_bleed_should_apply_for_label("place_slot_0_align_high"))

    def test_grasp_pre_lift_bleed_defaults(self):
        core = importlib.import_module("runtime_core")
        self.assertTrue(bool(core.GRASP_PRE_LIFT_BLEED_ENABLED))
        self.assertGreaterEqual(float(core.GRASP_LIFT_SEGMENT_S), 1.5)
        self.assertGreaterEqual(int(core.GRASP_LIFT_STEPS), 60)

    def test_far_xy_grasp_z_lift_threshold(self):
        core = importlib.import_module("runtime_core")
        self.assertFalse(bool(core.GRASP_FAR_XY_Z_LIFT_ENABLED))
        near_lift, near_reach = core._compute_grasp_far_xy_z_lift(0.510, -0.180)
        far_lift, far_reach = core._compute_grasp_far_xy_z_lift(0.449, -0.322)
        self.assertAlmostEqual(0.0, float(near_lift), places=6)
        self.assertAlmostEqual(0.0, float(far_lift), places=6)
        self.assertTrue(math.isnan(float(near_reach)))
        self.assertTrue(math.isnan(float(far_reach)))

    def test_carry_bleed_move_time_relaxes_high_current(self):
        core = importlib.import_module("runtime_core")

        class _FakeArm:
            def __init__(self):
                self.tick_calls = 0

            def tick_hold(self, *, grip: float):
                self.tick_calls += 1

        arm = _FakeArm()
        orig_read = core.read_gripper_current_abs
        try:
            core.read_gripper_current_abs = lambda _arm: 0.35
            out = core.maybe_bleed_carry_grip_on_move(arm, float(core.MAX_GRIP_CMD), "prompted_cycle_1_carry")
        finally:
            core.read_gripper_current_abs = orig_read
        self.assertLess(float(out), float(core.MAX_GRIP_CMD))
        self.assertGreaterEqual(float(out), float(core.POST_LIFT_CARRY_MIN_GRIP_CMD) - 1e-6)

    def test_misplaced_pick_top_height_gate_rejects_wrong_layer(self):
        misplaced = importlib.import_module("misplaced_actions")
        step_m = 0.060
        tol_m = 0.008
        reject = misplaced._misplaced_pick_top_height_eval(
            0.125,
            expected_level=3,
            step_m=step_m,
            top_tol_m=tol_m,
            require_top_level_match=True,
        )
        self.assertFalse(bool(reject.get("valid", False)))
        self.assertEqual("height_not_top_level", str(reject.get("reason")))
        self.assertEqual(2, int(reject.get("selected_level")))
        self.assertEqual(3, int(reject.get("expected_state_level")))

        accept_top = misplaced._misplaced_pick_top_height_eval(
            0.178,
            expected_level=3,
            step_m=step_m,
            top_tol_m=tol_m,
            require_top_level_match=True,
        )
        self.assertTrue(bool(accept_top.get("valid", False)))
        self.assertEqual("ok", str(accept_top.get("reason")))
        self.assertEqual(3, int(accept_top.get("selected_level")))

        accept_two_high = misplaced._misplaced_pick_top_height_eval(
            0.125,
            expected_level=2,
            step_m=step_m,
            top_tol_m=tol_m,
            require_top_level_match=True,
        )
        self.assertTrue(bool(accept_two_high.get("valid", False)))
        self.assertEqual("ok", str(accept_two_high.get("reason")))
        self.assertEqual(2, int(accept_two_high.get("selected_level")))

    def test_misplaced_pick_top_height_tol_default_accepts_top_not_base(self):
        misplaced = importlib.import_module("misplaced_actions")
        core = importlib.import_module("runtime_core")
        step_m = 0.060
        top_tol = float(core.MISPLACED_PICK_TOP_HEIGHT_TOL_M)
        reject_base = misplaced._misplaced_pick_top_height_eval(
            0.073,
            expected_level=2,
            step_m=step_m,
            top_tol_m=top_tol,
            require_top_level_match=True,
        )
        self.assertFalse(bool(reject_base.get("valid", False)))
        self.assertEqual("height_not_top_level", str(reject_base.get("reason")))
        self.assertEqual(1, int(reject_base.get("selected_level")))
        accept_face = misplaced._misplaced_pick_top_height_eval(
            0.108,
            expected_level=2,
            step_m=step_m,
            top_tol_m=top_tol,
            require_top_level_match=True,
        )
        self.assertTrue(bool(accept_face.get("valid", False)))
        self.assertEqual("ok", str(accept_face.get("reason")))
        self.assertEqual(2, int(accept_face.get("selected_level")))

    def test_misplaced_pick_top_height_accepts_higher_than_expected(self):
        misplaced = importlib.import_module("misplaced_actions")
        core = importlib.import_module("runtime_core")
        top_tol = float(core.MISPLACED_PICK_TOP_HEIGHT_TOL_M)
        accept_higher = misplaced._misplaced_pick_top_height_eval(
            0.178,
            expected_level=2,
            step_m=0.060,
            top_tol_m=top_tol,
            require_top_level_match=True,
        )
        self.assertTrue(bool(accept_higher.get("valid", False)))
        self.assertEqual("ok_higher_than_expected", str(accept_higher.get("reason")))
        self.assertEqual(3, int(accept_higher.get("selected_level")))
        self.assertEqual(2, int(accept_higher.get("expected_state_level")))
        self.assertEqual("raise_state_to_measured", str(accept_higher.get("state_reconcile_action")))

    def test_pick_placed_handoff_prefers_max_z_over_pixel_top(self):
        core = importlib.import_module("runtime_core")
        misplaced = importlib.import_module("misplaced_actions")
        cx = 320
        cy = 240
        candidates = [
            {"track_id": 10, "u": 420, "v": 80, "conf": 0.95},
            {"track_id": 11, "u": 400, "v": 200, "conf": 0.90},
        ]
        projected = [
            {"u": 420, "v": 80, "xyz": [0.45, 0.13, 0.178]},
            {"u": 400, "v": 200, "xyz": [0.45, 0.13, 0.074]},
        ]
        picked = misplaced.rank_misplaced_handoff_track_candidate(
            candidates,
            projected,
            cx=int(cx),
            cy=int(cy),
            min_conf=0.5,
            preferred_section_norm=str(core.SECTION_RIGHT_NAME),
            enforce_preferred_section_hard_filter=False,
            section_centers_xy={
                str(core.SECTION_LEFT_NAME): (0.45, 0.29),
                str(core.SECTION_RIGHT_NAME): (0.45, 0.13),
            },
        )
        self.assertIsNotNone(picked)
        self.assertEqual(10, int(picked["track_id"]))
        self.assertEqual("max_z", str(picked.get("handoff_rank", "")))

    def test_clamp_grip_cmd_respects_max(self):
        core = importlib.import_module("runtime_core")
        self.assertAlmostEqual(0.58, float(core.MAX_GRIP_CMD), places=6)
        self.assertAlmostEqual(0.58, float(core.GRIP_DEFAULT), places=6)
        self.assertLessEqual(float(core.clamp_grip_cmd(0.61)), float(core.MAX_GRIP_CMD) + 1e-9)
        self.assertAlmostEqual(float(core.clamp_grip_cmd(0.61)), float(core.MAX_GRIP_CMD), places=6)

    def test_place_release_opens_fully_before_clearance(self):
        core = importlib.import_module("runtime_core")
        place_actions = importlib.import_module("place_actions")

        self.assertAlmostEqual(float(core.PLACE_RELEASE_OPEN_GRIP), float(core.PLACE_RELEASE_TOUCH_OPEN_GRIP), places=6)
        self.assertGreaterEqual(float(core.PLACE_RELEASE_DURATION_S), 1.5)
        plan, reason = place_actions._build_place_plan(
            np.array([0.48, 0.15, 0.03], dtype=float),
            hold_grip=float(core.MAX_GRIP_CMD),
            verified_max_stack_level=0,
        )
        self.assertEqual("ok", str(reason))
        self.assertIsNotNone(plan)
        self.assertAlmostEqual(float(plan.release_open_grip), float(plan.release_touch_grip), places=6)
        self.assertAlmostEqual(float(plan.place_open[3]), float(plan.release_open_grip), places=6)
        self.assertAlmostEqual(float(plan.place_release_clear[3]), float(plan.release_open_grip), places=6)

    def test_carry_bleed_constants_defaults(self):
        core = importlib.import_module("runtime_core")
        self.assertTrue(bool(core.POST_LIFT_CARRY_BLEED_ENABLED))
        self.assertGreater(float(core.POST_LIFT_CARRY_GRIP_TRIGGER_CMD), 0.50)
        self.assertLessEqual(float(core.POST_LIFT_CARRY_GRIP_TRIGGER_CMD), float(core.MAX_GRIP_CMD))
        self.assertLessEqual(int(core.POST_LIFT_CARRY_MAX_STEPS), int(core.POST_LIFT_TUNE_MAX_STEPS))
        self.assertLessEqual(float(core.POST_LIFT_CARRY_MIN_GRIP_CMD), float(core.MAX_GRIP_CMD))

    def test_xyz_in_pick_workspace_defaults(self):
        core = importlib.import_module("runtime_core")
        y_max = float(core.pick_workspace_y_max_m())
        self.assertTrue(core.xyz_in_pick_workspace([0.40, 0.01, 0.05]))
        self.assertFalse(core.xyz_in_pick_workspace([0.40, 0.12, 0.05]))
        self.assertLessEqual(float(core.PICK_MAX_BASE_Y_M), y_max)
        reason = core.pick_workspace_reject_reason([0.40, 0.12, 0.05])
        self.assertIsNotNone(reason)
        self.assertIn("candidate_pick_space_rejected", str(reason))
        self.assertIn(f"max={y_max:.3f}", str(reason))

    def test_stack_anchor_startup_hydrate_top_locked(self):
        stack_scene = importlib.import_module("stack_scene")
        core = importlib.import_module("runtime_core")
        state = core.CycleState()
        state.startup_hydrated_sections = {
            core.SECTION_RIGHT_NAME: {
                "stack_level": 2,
                "top_color": "blue",
                "color_sequence_bottom_to_top": ["orange", "blue"],
                "tracks_bottom_to_top": [1, 2],
                "entries": [
                    {"xyz": [0.42, 0.12, 0.05], "color": "orange"},
                    {"xyz": [0.43, 0.11, 0.11], "color": "blue"},
                ],
            },
            core.SECTION_LEFT_NAME: stack_scene._startup_default_hydrated_section_row(),
        }
        stack_scene._set_locked_stack_anchor_xyz(
            state,
            core.SECTION_RIGHT_NAME,
            [0.43, 0.11, 0.11],
            "startup_hydrate_top",
        )
        anchor_xyz, anchor_source = stack_scene.get_latest_side_stack_anchor_xyz(
            state,
            core.SECTION_RIGHT_NAME,
        )
        self.assertIsNotNone(anchor_xyz)
        self.assertEqual("startup_hydrate_top", str(anchor_source))
        self.assertAlmostEqual(0.43, float(anchor_xyz[0]), places=4)
        self.assertAlmostEqual(0.11, float(anchor_xyz[1]), places=4)

    def test_append_authoritative_stack_cube_updates_sequence(self):
        stack_scene = importlib.import_module("stack_scene")
        core = importlib.import_module("runtime_core")
        state = core.CycleState()

        result = stack_scene.append_authoritative_stack_cube(
            state,
            core.SECTION_RIGHT_NAME,
            "orange",
        )

        self.assertTrue(bool(result.get("changed")))
        self.assertEqual("ok", str(result.get("reason")))
        self.assertEqual(1, int(result.get("stack_level")))
        self.assertEqual(["orange"], list(result.get("sequence")))
        self.assertEqual("orange", str(result.get("top_color")))
        row = stack_scene.get_startup_hydrated_section_row(state, core.SECTION_RIGHT_NAME)
        self.assertEqual(1, int(row.get("stack_level")))
        self.assertEqual(["orange"], list(row.get("color_sequence_bottom_to_top", [])))
        self.assertEqual("orange", str(row.get("top_color")))


class TestPlaceMeasurementPolicy(unittest.TestCase):
    def test_scan_offset_corrects_raw_toward_command_frame(self):
        import projection_geometry as proj

        raw = np.array([0.463, 0.227, 0.174], dtype=float)
        corrected = proj.apply_scan_base_xy_offset(raw)
        self.assertAlmostEqual(0.470, float(corrected[0]), places=4)
        self.assertAlmostEqual(0.248, float(corrected[1]), places=4)
        self.assertAlmostEqual(0.174, float(corrected[2]), places=4)

    def test_place_command_offset_disabled_by_default(self):
        import runtime_core as core

        self.assertFalse(bool(core.PLACE_CMD_XY_OFFSET_ENABLED))
        anchor = np.array([0.464, 0.248, 0.165], dtype=float)
        unchanged = core.apply_place_command_xy_offset(anchor)
        self.assertAlmostEqual(0.464, float(unchanged[0]), places=4)
        self.assertAlmostEqual(0.248, float(unchanged[1]), places=4)

    def test_place_slots_match_current_stack_locations(self):
        import runtime_core as core
        import stack_scene

        slots = stack_scene.get_place_slots()
        groups = stack_scene.section_slot_groups(slots)
        right = slots[groups[core.SECTION_RIGHT_NAME][0]]
        left = slots[groups[core.SECTION_LEFT_NAME][0]]
        self.assertAlmostEqual(0.480, float(right[0]), places=4)
        self.assertAlmostEqual(0.100, float(right[1]), places=4)
        self.assertAlmostEqual(0.037, float(right[2]), places=4)
        self.assertAlmostEqual(0.480, float(left[0]), places=4)
        self.assertAlmostEqual(0.225, float(left[1]), places=4)
        self.assertAlmostEqual(0.037, float(left[2]), places=4)

    def test_third_stack_level_uses_same_dz_step_as_middle(self):
        import place_actions
        import runtime_core as core

        base_z = float(core.PLACE_RELEASE_Z_M)
        self.assertAlmostEqual(0.000, float(core.PLACE_STACK_LEVEL3_EXTRA_Z_M), places=4)
        self.assertAlmostEqual(base_z, place_actions._stack_release_z_for_level(base_z, 0), places=4)
        self.assertAlmostEqual(
            base_z + float(core.PLACE_STACK_LEVEL_DZ_M) + float(core.PLACE_STACK_UPPER_EXTRA_Z_M),
            place_actions._stack_release_z_for_level(base_z, 1),
            places=4,
        )
        self.assertAlmostEqual(
            base_z
            + (2.0 * float(core.PLACE_STACK_LEVEL_DZ_M))
            + float(core.PLACE_STACK_UPPER_EXTRA_Z_M),
            place_actions._stack_release_z_for_level(base_z, 2),
            places=4,
        )

    def test_verify_eval_offsets_optional(self):
        import runtime_core as core
        import verify_v2

        anchor = np.array([0.464, 0.248, 0.165], dtype=float)
        measured = np.array([0.466, 0.248, 0.174], dtype=float)
        score = verify_v2.score_place_geometry(
            expected_xyz=anchor,
            measured_xyz=measured,
            hits=8,
            min_hits=4,
            xy_margin_m=0.078,
            z_margin_m=0.023,
            delta_score=1.0,
            delta_min=-1.0,
        )
        self.assertTrue(bool(score["confirmed"]))

    def test_verify_expected_eval_uses_surface_z_for_placed_cube(self):
        import verify_v2

        expected_eval, x_off, y_off, z_off = verify_v2.build_verify_expected_for_score(
            np.array([0.480, 0.150, 0.030], dtype=float)
        )
        self.assertAlmostEqual(0.0, float(x_off), places=4)
        self.assertAlmostEqual(0.0, float(y_off), places=4)
        self.assertAlmostEqual(0.030, float(z_off), places=4)
        self.assertAlmostEqual(0.480, float(expected_eval[0]), places=4)
        self.assertAlmostEqual(0.150, float(expected_eval[1]), places=4)
        self.assertAlmostEqual(0.060, float(expected_eval[2]), places=4)
        score = verify_v2.score_place_geometry(
            expected_xyz=expected_eval,
            measured_xyz=np.array([0.486, 0.149, 0.058], dtype=float),
            hits=8,
            min_hits=4,
            xy_margin_m=0.068,
            z_margin_m=0.023,
            delta_score=1.0,
            delta_min=-1.0,
        )
        self.assertTrue(bool(score["confirmed"]))

    def test_verify_uses_pending_stack_level_for_top_targeting(self):
        import runtime_core as core
        import verify_v2

        placement = {"stack_level": 1, "pending_stack_level": 2}
        target_level = verify_v2._verify_stack_level_for_placement(placement)
        self.assertEqual(2, int(target_level))
        self.assertIsNotNone(verify_v2.compute_verify_stack_min_z(0.096, int(target_level)))
        self.assertTrue(bool(core.PLACE_VERIFY_V2_STACK_PREFER_TOP and int(target_level) >= 2))

    def test_place_verify_hard_timeout_covers_recenter_ladder(self):
        import runtime_core as core

        recenter_windows = (
            1
            + max(0, int(core.PLACE_VERIFY_V2_EXPECTED_SLOT_RETRIES))
            + max(0, int(core.PLACE_VERIFY_V2_TOP_CANDIDATE_CHECKS))
        )
        active_budget_s = recenter_windows * float(core.PLACE_VERIFY_V2_ACTIVE_CENTER_TIMEOUT_S)

        self.assertEqual(4, int(recenter_windows))
        self.assertAlmostEqual(20.0, float(core.PLACE_VERIFY_V2_HARD_TIMEOUT_S), places=4)
        self.assertGreaterEqual(float(core.PLACE_VERIFY_V2_HARD_TIMEOUT_S), active_budget_s + 1.0)

    def test_verify_candidate_filter_ignores_wrong_section_before_tracking(self):
        import runtime_core as core
        import stack_scene
        import verify_v2

        centers = stack_scene._verify_section_xy_centers()
        left_x, left_y = centers[core.SECTION_LEFT_NAME]
        right_x, right_y = centers[core.SECTION_RIGHT_NAME]
        raw_candidates = [
            {"track_id": 10, "u": 320, "v": 235, "conf": 0.95},
            {"track_id": 11, "u": 318, "v": 238, "conf": 0.94},
            {"track_id": 12, "u": 321, "v": 236, "conf": 0.93},
        ]
        projected_rows = [
            {"u": 320, "v": 235, "conf": 0.95, "xyz": [right_x, right_y, 0.118]},
            {"u": 318, "v": 238, "conf": 0.94, "xyz": [left_x, left_y, 0.064]},
            {"u": 321, "v": 236, "conf": 0.93, "xyz": [0.577, -0.125, 0.061]},
        ]
        filtered, meta = verify_v2._filter_verify_candidates(
            raw_candidates=raw_candidates,
            bgr_img=np.zeros((480, 640, 3), dtype=np.uint8),
            projected_rows=projected_rows,
            expected_section=core.SECTION_LEFT_NAME,
            expected_color=None,
            expected_xyz=np.array([left_x, left_y, 0.068], dtype=float),
        )
        self.assertEqual([11], [int(row["track_id"]) for row in filtered])
        self.assertEqual(1, int(meta.get("filtered_count", 0)))

    def test_verify_candidate_filter_can_disable_prelock_projection(self):
        import runtime_core as core
        import stack_scene
        import verify_v2

        centers = stack_scene._verify_section_xy_centers()
        left_x, left_y = centers[core.SECTION_LEFT_NAME]
        right_x, right_y = centers[core.SECTION_RIGHT_NAME]
        raw_candidates = [
            {"track_id": 10, "u": 320, "v": 235, "conf": 0.95},
            {"track_id": 11, "u": 318, "v": 238, "conf": 0.94},
        ]
        projected_rows = [
            {"u": 320, "v": 235, "conf": 0.95, "xyz": [left_x, left_y, 0.118]},
            {"u": 318, "v": 238, "conf": 0.94, "xyz": [right_x, right_y, 0.064]},
        ]
        filtered, meta = verify_v2._filter_verify_candidates(
            raw_candidates=raw_candidates,
            bgr_img=np.zeros((480, 640, 3), dtype=np.uint8),
            projected_rows=projected_rows,
            expected_section=core.SECTION_RIGHT_NAME,
            expected_color=None,
            expected_xyz=np.array([right_x, right_y, 0.068], dtype=float),
            use_projected_geometry=False,
        )
        self.assertEqual([10, 11], [int(row["track_id"]) for row in filtered])
        self.assertEqual(2, int(meta.get("filtered_count", 0)))
        self.assertEqual(0, int(meta.get("wrong_xy_hits", 0)))

    def test_verify_recenter_negative_y_prefilter_keeps_low_positive_y(self):
        import centering

        raw_candidates = [
            {"track_id": 10, "u": 320, "v": 235, "conf": 0.95},
            {"track_id": 11, "u": 318, "v": 238, "conf": 0.94},
            {"track_id": 12, "u": 321, "v": 236, "conf": 0.93},
        ]
        projected_rows = [
            {"u": 320, "v": 235, "conf": 0.95, "xyz": [0.48, -0.083, 0.060]},
            {"u": 318, "v": 238, "conf": 0.94, "xyz": [0.48, 0.010, 0.060]},
            {"u": 321, "v": 236, "conf": 0.93, "xyz": [0.48, 0.100, 0.060]},
        ]
        filtered, meta = centering._filter_verify_negative_y_candidates(raw_candidates, projected_rows)
        self.assertEqual([11, 12], [int(row["track_id"]) for row in filtered])
        self.assertEqual(1, int(meta.get("negative_y_skips", 0)))

    def test_verify_restores_track_handoff_measurement(self):
        verify_src = (REPO_ROOT / "verify_v2.py").read_text(encoding="utf-8")
        self.assertNotIn("slot_scan_association", verify_src)
        self.assertIn("run_track_handoff_session(", verify_src)
        self.assertIn("on_locked_candidate=_verify_on_locked_candidate", verify_src)
        self.assertIn("use_projected_xyz_for_filter=False", verify_src)
        self.assertIn("use_pixel_blacklist=False", verify_src)
        self.assertIn("reset_timeout_on_first_candidate=True", verify_src)
        self.assertIn("reject_below_base_y_m=", verify_src)

    def test_verify_color_commit_requires_matching_geometry(self):
        import verify_v2

        wrong_target = {
            "confirmed": False,
            "measured_xyz": [0.577, -0.125, 0.061],
            "xy_error_m": 0.365,
            "z_error_m": 0.007,
            "effective_xy_margin_m": 0.068,
            "effective_z_margin_m": 0.023,
        }
        close_target = {
            "confirmed": False,
            "measured_xyz": [0.476, 0.224, 0.066],
            "xy_error_m": 0.003,
            "z_error_m": 0.002,
            "effective_xy_margin_m": 0.068,
            "effective_z_margin_m": 0.023,
        }
        self.assertFalse(verify_v2._color_geometry_ok_for_commit(wrong_target))
        self.assertTrue(verify_v2._color_geometry_ok_for_commit(close_target))
        wrong_target["confirmed"] = True
        self.assertTrue(verify_v2._color_geometry_ok_for_commit(wrong_target))

    def test_place_pick_bias_compensate_disabled_by_default_and_inverts_when_enabled(self):
        import runtime_core as core

        self.assertFalse(bool(core.PLACE_PICK_BIAS_COMPENSATE_ENABLED))
        self.assertFalse(bool(core.PLACE_PICK_BIAS_COMPENSATE_STACK_ANCHOR_ENABLED))
        self.assertAlmostEqual(0.037, float(core.PLACE_RELEASE_Z_M), places=4)
        anchor = np.array([0.463, 0.277, 0.165], dtype=float)
        cmd, dx, dy, applied = core.apply_place_pick_bias_compensate(anchor)
        self.assertFalse(bool(applied))
        self.assertAlmostEqual(0.0, float(dx), places=4)
        self.assertAlmostEqual(0.0, float(dy), places=4)
        self.assertAlmostEqual(float(anchor[0]), float(cmd[0]), places=4)
        self.assertAlmostEqual(float(anchor[1]), float(cmd[1]), places=4)
        self.assertAlmostEqual(0.165, float(cmd[2]), places=4)

        old_enabled = core.PLACE_PICK_BIAS_COMPENSATE_ENABLED
        try:
            core.PLACE_PICK_BIAS_COMPENSATE_ENABLED = True
            cmd, dx, dy, applied = core.apply_place_pick_bias_compensate(anchor)
        finally:
            core.PLACE_PICK_BIAS_COMPENSATE_ENABLED = old_enabled
        self.assertTrue(bool(applied))
        self.assertAlmostEqual(
            -float(core.GRASP_PICK_X_BIAS_M) * float(core.PLACE_PICK_BIAS_COMPENSATE_SCALE),
            float(dx),
            places=4,
        )
        self.assertAlmostEqual(
            -float(core.GRASP_PICK_Y_BIAS_M) * float(core.PLACE_PICK_BIAS_COMPENSATE_SCALE),
            float(dy),
            places=4,
        )
        self.assertAlmostEqual(float(anchor[0]) + float(dx), float(cmd[0]), places=4)
        self.assertAlmostEqual(float(anchor[1]) + float(dy), float(cmd[1]), places=4)
        self.assertAlmostEqual(0.165, float(cmd[2]), places=4)

    def test_stack_pick_x_offset_scales_from_latched_pick_x(self):
        import runtime_core as core

        self.assertFalse(bool(core.STACK_X_LEVEL_OFFSET_ENABLED))
        self.assertTrue(bool(core.STACK_PICK_X_OFFSET_ENABLED))
        self.assertAlmostEqual(-0.002, float(core.STACK_PICK_X_OFFSET_FAR_M), places=4)
        self.assertAlmostEqual(-0.005, float(core.STACK_PICK_X_LEVEL2_EXTRA_M), places=4)
        self.assertAlmostEqual(0.003, float(core.STACK_PICK_X_NEAR_Z_EXTRA_M), places=4)
        self.assertAlmostEqual(0.002, float(core.STACK_PICK_X_FAR_Z_EXTRA_M), places=4)

        near_dx, near_meta = core.compute_stack_pick_x_offset([0.3015, -0.1705, 0.0567])
        self.assertEqual("ok", str(near_meta.get("reason")))
        self.assertAlmostEqual(float(core.STACK_PICK_X_OFFSET_NEAR_M), float(near_dx), places=4)
        self.assertAlmostEqual(0.0, float(near_meta.get("t")), places=4)

        far_dx, far_meta = core.compute_stack_pick_x_offset([0.5259, -0.1771, 0.0678])
        self.assertEqual("ok", str(far_meta.get("reason")))
        self.assertAlmostEqual(float(core.STACK_PICK_X_OFFSET_FAR_M), float(far_dx), places=3)
        self.assertAlmostEqual(1.0, float(far_meta.get("t")), places=3)

        mid_x = 0.5 * (float(core.STACK_PICK_X_NEAR_M) + float(core.STACK_PICK_X_FAR_M))
        mid_dx, mid_meta = core.compute_stack_pick_x_offset([mid_x, -0.100, 0.060])
        expected_mid = 0.5 * (
            float(core.STACK_PICK_X_OFFSET_NEAR_M) + float(core.STACK_PICK_X_OFFSET_FAR_M)
        )
        self.assertEqual("ok", str(mid_meta.get("reason")))
        self.assertAlmostEqual(expected_mid, float(mid_dx), places=4)
        self.assertAlmostEqual(0.5, float(mid_meta.get("t")), places=4)

        observed_mid_dx, observed_mid_meta = core.compute_stack_pick_x_offset([0.4186, -0.1299, 0.0622])
        self.assertEqual("ok", str(observed_mid_meta.get("reason")))
        self.assertGreater(float(observed_mid_dx), float(core.STACK_PICK_X_OFFSET_NEAR_M))
        self.assertLess(float(observed_mid_dx), float(core.STACK_PICK_X_OFFSET_FAR_M))

        log_pick_dx, log_pick_meta = core.compute_stack_pick_x_offset([0.465, -0.276, 0.065])
        self.assertEqual("ok", str(log_pick_meta.get("reason")))
        self.assertAlmostEqual(0.474, 0.480 + float(log_pick_dx), places=3)
        self.assertAlmostEqual(
            0.469,
            0.480 + float(log_pick_dx) + float(core.STACK_PICK_X_LEVEL2_EXTRA_M),
            places=3,
        )

    def test_stack_pick_x_offset_reports_missing_pick_xyz(self):
        import runtime_core as core

        dx, meta = core.compute_stack_pick_x_offset(None)
        self.assertAlmostEqual(0.0, float(dx), places=4)
        self.assertIn(str(meta.get("reason")), {"invalid_pick_xyz", "missing_pick_xyz"})
        self.assertFalse(bool(meta.get("applied")))

    def test_verify_mismatch_score_exposes_margin_metadata(self):
        import verify_v2

        score = verify_v2.score_place_geometry(
            expected_xyz=np.array([0.430, 0.188, 0.030], dtype=float),
            measured_xyz=np.array([0.520, -0.140, 0.058], dtype=float),
            hits=8,
            min_hits=4,
            xy_margin_m=0.068,
            z_margin_m=0.023,
            delta_score=1.0,
            delta_min=-1.0,
            min_overlap=0.65,
        )
        self.assertEqual("placed_mismatch_out_of_margin", str(score.get("status")))
        self.assertAlmostEqual(0.068, float(score.get("xy_margin_m")), places=4)
        self.assertAlmostEqual(0.023, float(score.get("z_margin_m")), places=4)
        self.assertAlmostEqual(0.65, float(score.get("min_overlap")), places=4)

    def test_hydrate_fallback_accepts_same_section_layer_geometry(self):
        import runtime_core as core
        import runtime_loop_actions_place as place_handler

        startup_row = {
            "hydration_status": "ok",
            "hydration_unresolved_visible_track_ids": [],
            "hydrated_stacks": {
                "sections": {
                    core.SECTION_LEFT_NAME: {
                        "stack_level": 1,
                        "top_color": "blue",
                        "color_sequence_bottom_to_top": ["blue"],
                        "tracks_bottom_to_top": [31],
                        "entries": [
                            {
                                "track_id": 31,
                                "xyz": [0.476, 0.224, 0.066],
                                "color": "blue",
                            }
                        ],
                    },
                    core.SECTION_RIGHT_NAME: {
                        "stack_level": 2,
                        "top_color": "orange",
                        "color_sequence_bottom_to_top": ["orange", "orange"],
                        "tracks_bottom_to_top": [20, 21],
                        "entries": [
                            {"track_id": 20, "xyz": [0.480, 0.100, 0.060], "color": "orange"},
                            {"track_id": 21, "xyz": [0.480, 0.100, 0.120], "color": "orange"},
                        ],
                    },
                }
            },
        }
        evaluation = place_handler._evaluate_place_verify_hydrate_fallback(
            startup_boot_row=startup_row,
            section=core.SECTION_LEFT_NAME,
            pending_stack_level=1,
            expected_color="blue",
            place_verify={
                "expected_xyz_eval": [0.475, 0.225, 0.068],
                "effective_xy_margin_m": 0.068,
                "effective_z_margin_m": 0.023,
            },
        )
        self.assertTrue(bool(evaluation.get("accepted")))
        self.assertEqual("ok", str(evaluation.get("reason")))
        self.assertAlmostEqual(0.002, float(evaluation.get("z_error_m")), places=4)

    def test_hydrate_fallback_rejects_wrong_color_layer(self):
        import runtime_core as core
        import runtime_loop_actions_place as place_handler

        startup_row = {
            "hydration_status": "ok",
            "hydration_unresolved_visible_track_ids": [],
            "hydrated_stacks": {
                "sections": {
                    core.SECTION_LEFT_NAME: {
                        "stack_level": 1,
                        "top_color": "orange",
                        "color_sequence_bottom_to_top": ["orange"],
                        "tracks_bottom_to_top": [31],
                        "entries": [
                            {
                                "track_id": 31,
                                "xyz": [0.476, 0.224, 0.066],
                                "color": "orange",
                            }
                        ],
                    }
                }
            },
        }
        evaluation = place_handler._evaluate_place_verify_hydrate_fallback(
            startup_boot_row=startup_row,
            section=core.SECTION_LEFT_NAME,
            pending_stack_level=1,
            expected_color="blue",
            place_verify={
                "expected_xyz_eval": [0.475, 0.225, 0.068],
                "effective_xy_margin_m": 0.068,
                "effective_z_margin_m": 0.023,
            },
        )
        self.assertFalse(bool(evaluation.get("accepted")))
        self.assertEqual("color_mismatch", str(evaluation.get("reason")))

    def test_higher_layer_classifier_detects_same_xy_high_z_unconfirmed_stack(self):
        import runtime_core as core
        import runtime_loop_actions_place as place_handler

        place_verify = {
            "status": "placed_mismatch_out_of_margin",
            "confirmed": False,
            "expected_xyz_eval": [0.480, 0.225, 0.068],
            "measured_xyz": [0.486, 0.224, 0.128],
        }

        result = place_handler._classify_place_verify_higher_layer_scan(
            place_verify=place_verify,
            section=core.SECTION_LEFT_NAME,
            pending_stack_level=1,
        )

        self.assertTrue(bool(result.get("detected")))
        self.assertEqual("expected_layer_scanned_higher_than_expected", str(result.get("reason")))
        self.assertAlmostEqual(0.060, float(result.get("z_delta_m")), places=4)

    def test_higher_layer_classifier_rejects_missing_wrong_xy_confirmed_and_absurd_z(self):
        import runtime_core as core
        import runtime_loop_actions_place as place_handler

        base = {
            "status": "placed_mismatch_out_of_margin",
            "confirmed": False,
            "expected_xyz_eval": [0.480, 0.225, 0.068],
            "measured_xyz": [0.486, 0.224, 0.128],
        }
        cases = [
            ("missing_measured_xyz", {**base, "measured_xyz": None}, core.SECTION_LEFT_NAME, 1),
            ("xy_out_of_gate", {**base, "measured_xyz": [0.550, 0.225, 0.128]}, core.SECTION_LEFT_NAME, 1),
            ("already_confirmed", {**base, "confirmed": True}, core.SECTION_LEFT_NAME, 1),
            ("invalid_section", base, "center", 1),
            ("z_delta_too_large", {**base, "measured_xyz": [0.486, 0.224, 0.250]}, core.SECTION_LEFT_NAME, 1),
        ]
        for reason, verify_row, section, pending in cases:
            with self.subTest(reason=reason):
                result = place_handler._classify_place_verify_higher_layer_scan(
                    place_verify=verify_row,
                    section=section,
                    pending_stack_level=pending,
                )
                self.assertFalse(bool(result.get("detected")))
                self.assertEqual(reason, str(result.get("reason")))

    def test_higher_layer_classifier_uses_verify_reject_even_after_middle_confirm(self):
        import runtime_core as core
        import runtime_loop_actions_place as place_handler

        place_verify = {
            "status": "placed_confirmed_geometry",
            "confirmed": True,
            "expected_xyz_eval": [0.480, 0.100, 0.132],
            "measured_xyz": [0.451, 0.090, 0.135],
            "verify_higher_layer_reject_seen": True,
            "verify_higher_layer_rejects": [
                {
                    "track_id": 40,
                    "selected_xyz": [0.471, 0.107, 0.179],
                    "expected_xyz_eval": [0.480, 0.100, 0.132],
                    "xy_error_m": 0.011,
                    "z_delta_m": 0.047,
                    "xy_gate_m": 0.030,
                    "min_dz_m": 0.030,
                    "max_dz_m": 0.140,
                    "status": "placed_mismatch_out_of_margin",
                }
            ],
        }

        result = place_handler._classify_place_verify_higher_layer_scan(
            place_verify=place_verify,
            section=core.SECTION_RIGHT_NAME,
            pending_stack_level=2,
        )

        self.assertTrue(bool(result.get("detected")))
        self.assertEqual("verify_higher_layer_reject", str(result.get("source")))
        self.assertTrue(bool(result.get("original_confirmed")))
        self.assertEqual(40, int(result.get("track_id")))

    def test_higher_layer_hydrate_accepts_pending_layer_and_reports_extra_layers(self):
        import runtime_core as core
        import runtime_loop_actions_place as place_handler

        startup_row = {
            "hydration_status": "ok",
            "hydration_unresolved_visible_track_ids": [99],
            "hydrated_stacks": {
                "sections": {
                    core.SECTION_LEFT_NAME: {
                        "stack_level": 2,
                        "top_color": "orange",
                        "color_sequence_bottom_to_top": ["blue", "orange"],
                        "tracks_bottom_to_top": [31, 32],
                        "entries": [
                            {"track_id": 31, "xyz": [0.481, 0.224, 0.067], "color": "blue"},
                            {"track_id": 32, "xyz": [0.482, 0.224, 0.128], "color": "orange"},
                        ],
                    }
                }
            },
        }

        evaluation = place_handler._evaluate_place_verify_higher_layer_hydrate(
            startup_boot_row=startup_row,
            section=core.SECTION_LEFT_NAME,
            pending_stack_level=1,
            expected_color="blue",
            place_verify={
                "expected_xyz_eval": [0.480, 0.225, 0.068],
                "effective_z_margin_m": 0.023,
            },
        )

        self.assertTrue(bool(evaluation.get("accepted")))
        self.assertEqual(2, int(evaluation.get("hydrated_level")))
        self.assertEqual(1, int(evaluation.get("observed_extra_layers")))
        self.assertEqual("blue", str(evaluation.get("measured_color")))
        self.assertAlmostEqual(0.001, float(evaluation.get("z_error_m")), places=4)
        self.assertEqual([99], list(evaluation.get("unresolved_visible_track_ids", [])))
        self.assertTrue(bool(evaluation.get("unresolved_visible_tracks_ignored_for_scoped_apply")))

    def test_higher_layer_complete_hydrate_accepts_hidden_pending_layer_xy_mismatch(self):
        import runtime_core as core
        import runtime_loop_actions_place as place_handler
        import stack_scene

        startup_row = {
            "hydration_status": "ok",
            "hydration_missing_sides": [],
            "hydration_expected_shortfall_sides": [],
            "hydration_unresolved_visible_track_ids": [],
            "hydrated_stacks": {
                "observed_stack_levels": {
                    core.SECTION_LEFT_NAME: 3,
                    core.SECTION_RIGHT_NAME: 3,
                },
                "expected_stack_levels": {
                    core.SECTION_LEFT_NAME: 3,
                    core.SECTION_RIGHT_NAME: 3,
                },
                "sections": {
                    core.SECTION_LEFT_NAME: {
                        "stack_level": 3,
                        "top_color": "blue",
                        "color_sequence_bottom_to_top": ["blue", "orange", "blue"],
                        "tracks_bottom_to_top": [75, 77, 76],
                        "entries": [
                            {"track_id": 75, "xyz": [0.433, 0.226, 0.092], "color": "blue"},
                            {"track_id": 77, "xyz": [0.428, 0.232, 0.135], "color": "orange"},
                            {"track_id": 76, "xyz": [0.453, 0.245, 0.180], "color": "blue"},
                        ],
                    },
                    core.SECTION_RIGHT_NAME: {
                        "stack_level": 3,
                        "top_color": "orange",
                        "color_sequence_bottom_to_top": ["orange", "blue", "orange"],
                        "tracks_bottom_to_top": [108, 79, 74],
                        "entries": [
                            {"track_id": 108, "xyz": [0.434, 0.091, 0.087], "color": "orange"},
                            {"track_id": 79, "xyz": [0.433, 0.092, 0.122], "color": "blue"},
                            {"track_id": 74, "xyz": [0.458, 0.096, 0.176], "color": "orange"},
                        ],
                    },
                },
            },
        }

        evaluation = place_handler._evaluate_place_verify_higher_layer_hydrate(
            startup_boot_row=startup_row,
            section=core.SECTION_RIGHT_NAME,
            pending_stack_level=2,
            expected_color="blue",
            place_verify={
                "expected_xyz_eval": [0.480, 0.100, 0.132],
                "effective_z_margin_m": 0.023,
            },
        )

        self.assertTrue(bool(evaluation.get("accepted")))
        self.assertEqual(
            "complete_hydrate_authoritative_after_higher_layer",
            str(evaluation.get("reason")),
        )
        self.assertTrue(bool(evaluation.get("geometry_gate_skipped")))
        self.assertGreater(float(evaluation.get("xy_error_m")), float(evaluation.get("xy_margin_m")))

        state = core.CycleState()
        state.startup_hydrated_sections = {
            core.SECTION_LEFT_NAME: {
                "stack_level": 1,
                "top_color": "blue",
                "color_sequence_bottom_to_top": ["blue"],
                "tracks_bottom_to_top": [10],
                "entries": [{"track_id": 10, "xyz": [0.480, 0.225, 0.067], "color": "blue"}],
            },
            core.SECTION_RIGHT_NAME: {
                "stack_level": 1,
                "top_color": "orange",
                "color_sequence_bottom_to_top": ["orange"],
                "tracks_bottom_to_top": [20],
                "entries": [{"track_id": 20, "xyz": [0.480, 0.100, 0.067], "color": "orange"}],
            },
        }
        apply_result = stack_scene.apply_startup_stack_hydration_for_section(
            state,
            startup_row,
            core.SECTION_RIGHT_NAME,
        )
        left_row = stack_scene.get_startup_hydrated_section_row(state, core.SECTION_LEFT_NAME)
        right_row = stack_scene.get_startup_hydrated_section_row(state, core.SECTION_RIGHT_NAME)

        self.assertTrue(bool(apply_result.get("changed")))
        self.assertEqual(1, int(left_row.get("stack_level")))
        self.assertEqual(3, int(right_row.get("stack_level")))
        self.assertEqual(["orange", "blue", "orange"], list(right_row.get("color_sequence_bottom_to_top", [])))

    def test_higher_layer_hydrate_rejects_unresolved_shortfall_color_missing_xy_and_z(self):
        import runtime_core as core
        import runtime_loop_actions_place as place_handler

        base_row = {
            "hydration_status": "ok",
            "hydration_unresolved_visible_track_ids": [],
            "hydrated_stacks": {
                "sections": {
                    core.SECTION_LEFT_NAME: {
                        "stack_level": 1,
                        "top_color": "blue",
                        "color_sequence_bottom_to_top": ["blue"],
                        "tracks_bottom_to_top": [31],
                        "entries": [
                            {"track_id": 31, "xyz": [0.481, 0.224, 0.067], "color": "blue"},
                        ],
                    }
                }
            },
        }
        verify_row = {
            "expected_xyz_eval": [0.480, 0.225, 0.068],
            "effective_z_margin_m": 0.023,
        }
        cases = [
            ("level_below_pending", base_row, 2, "blue"),
            ("color_mismatch", base_row, 1, "orange"),
            (
                "missing_layer_xyz",
                {
                    **base_row,
                    "hydrated_stacks": {
                        "sections": {
                            core.SECTION_LEFT_NAME: {
                                "stack_level": 1,
                                "color_sequence_bottom_to_top": ["blue"],
                                "entries": [{"track_id": 31, "color": "blue"}],
                            }
                        }
                    },
                },
                1,
                "blue",
            ),
            (
                "xy_out_of_margin",
                {
                    **base_row,
                    "hydrated_stacks": {
                        "sections": {
                            core.SECTION_LEFT_NAME: {
                                "stack_level": 1,
                                "color_sequence_bottom_to_top": ["blue"],
                                "entries": [{"track_id": 31, "xyz": [0.540, 0.225, 0.067], "color": "blue"}],
                            }
                        }
                    },
                },
                1,
                "blue",
            ),
            (
                "z_out_of_margin",
                {
                    **base_row,
                    "hydrated_stacks": {
                        "sections": {
                            core.SECTION_LEFT_NAME: {
                                "stack_level": 1,
                                "color_sequence_bottom_to_top": ["blue"],
                                "entries": [{"track_id": 31, "xyz": [0.481, 0.224, 0.120], "color": "blue"}],
                            }
                        }
                    },
                },
                1,
                "blue",
            ),
        ]
        for reason, startup_row, pending, expected_color in cases:
            with self.subTest(reason=reason):
                evaluation = place_handler._evaluate_place_verify_higher_layer_hydrate(
                    startup_boot_row=startup_row,
                    section=core.SECTION_LEFT_NAME,
                    pending_stack_level=pending,
                    expected_color=expected_color,
                    place_verify=verify_row,
                )
                self.assertFalse(bool(evaluation.get("accepted")))
                self.assertEqual(reason, str(evaluation.get("reason")))

    def test_side_only_hydration_applies_one_section_and_preserves_other_side(self):
        import runtime_core as core
        import stack_scene

        state = core.CycleState()
        state.startup_hydrated_sections = {
            core.SECTION_LEFT_NAME: {
                "stack_level": 1,
                "top_color": "blue",
                "color_sequence_bottom_to_top": ["blue"],
                "tracks_bottom_to_top": [10],
                "entries": [{"track_id": 10, "xyz": [0.480, 0.225, 0.067], "color": "blue"}],
            },
            core.SECTION_RIGHT_NAME: {
                "stack_level": 1,
                "top_color": "orange",
                "color_sequence_bottom_to_top": ["orange"],
                "tracks_bottom_to_top": [20],
                "entries": [{"track_id": 20, "xyz": [0.480, 0.100, 0.067], "color": "orange"}],
            },
        }
        startup_row = {
            "hydrated_stacks": {
                "sections": {
                    core.SECTION_LEFT_NAME: {
                        "stack_level": 2,
                        "top_color": "orange",
                        "color_sequence_bottom_to_top": ["blue", "orange"],
                        "tracks_bottom_to_top": [10, 11],
                        "entries": [
                            {"track_id": 10, "xyz": [0.480, 0.225, 0.067], "color": "blue"},
                            {"track_id": 11, "xyz": [0.481, 0.226, 0.128], "color": "orange"},
                        ],
                    },
                    core.SECTION_RIGHT_NAME: {
                        "stack_level": 2,
                        "top_color": "blue",
                        "color_sequence_bottom_to_top": ["orange", "blue"],
                        "tracks_bottom_to_top": [20, 21],
                        "entries": [
                            {"track_id": 20, "xyz": [0.480, 0.100, 0.067], "color": "orange"},
                            {"track_id": 21, "xyz": [0.480, 0.100, 0.128], "color": "blue"},
                        ],
                    },
                }
            }
        }

        result = stack_scene.apply_startup_stack_hydration_for_section(
            state,
            startup_row,
            core.SECTION_LEFT_NAME,
        )

        self.assertTrue(bool(result.get("changed")))
        left_row = stack_scene.get_startup_hydrated_section_row(state, core.SECTION_LEFT_NAME)
        right_row = stack_scene.get_startup_hydrated_section_row(state, core.SECTION_RIGHT_NAME)
        self.assertEqual(2, int(left_row.get("stack_level")))
        self.assertEqual(["blue", "orange"], list(left_row.get("color_sequence_bottom_to_top", [])))
        self.assertEqual(1, int(right_row.get("stack_level")))
        self.assertEqual(["orange"], list(right_row.get("color_sequence_bottom_to_top", [])))
        anchor_xyz, anchor_source = stack_scene.get_latest_side_stack_anchor_xyz(
            state,
            core.SECTION_LEFT_NAME,
        )
        self.assertEqual("startup_hydrate_top", str(anchor_source))
        self.assertAlmostEqual(0.128, float(anchor_xyz[2]), places=4)

    def test_higher_layer_authoritative_marker_skips_append(self):
        import runtime_loop_actions_place as place_handler

        self.assertTrue(
            bool(
                place_handler._place_verify_authoritative_state_already_applied(
                    {"authoritative_state_source": "higher_layer_scoped_hydrate"}
                )
            )
        )
        self.assertFalse(bool(place_handler._place_verify_authoritative_state_already_applied({})))

    def test_higher_layer_unresolved_hydrate_reject_skips_remeasure(self):
        import runtime_loop_actions_place as place_handler

        self.assertTrue(
            bool(
                place_handler._should_skip_remeasure_after_higher_layer_hydrate_reject(
                    {
                        "status": "expected_layer_scanned_higher_than_expected",
                        "higher_layer_hydrate": {"reason": "unresolved_visible_tracks"},
                    }
                )
            )
        )
        self.assertFalse(
            bool(
                place_handler._should_skip_remeasure_after_higher_layer_hydrate_reject(
                    {
                        "status": "expected_layer_scanned_higher_than_expected",
                        "higher_layer_hydrate": {"reason": "level_below_pending"},
                    }
                )
            )
        )
        self.assertFalse(
            bool(
                place_handler._should_skip_remeasure_after_higher_layer_hydrate_reject(
                    {"status": "placed_mismatch_out_of_margin"}
                )
            )
        )

    def test_place_verify_hold_diag_includes_expected_measured_and_remeasure(self):
        import runtime_loop_actions_place as place_handler

        line = place_handler._format_place_verify_hold_diag(
            section="right",
            place_verify={
                "status": "placed_mismatch_out_of_margin",
                "expected_xyz": [0.430, 0.188, 0.030],
                "expected_xyz_eval": [0.430, 0.188, 0.030],
                "measured_xyz": [0.520, -0.140, 0.058],
                "xy_error_m": 0.340,
                "z_error_m": 0.028,
                "effective_xy_margin_m": 0.068,
                "effective_z_margin_m": 0.023,
                "overlap_ratio": 0.0,
            },
            remeasure_meta={"status": "stable", "valid": 2},
        )
        self.assertIn("expected_xyz=[0.430,0.188,0.030]", line)
        self.assertIn("measured_xyz=[0.520,-0.140,0.058]", line)
        self.assertIn("err_xy=0.340", line)
        self.assertIn("xy_margin=0.068", line)
        self.assertIn("remeasure=stable", line)
        self.assertIn("remeasure_valid=2", line)

    def test_place_current_collision_releases_in_place_and_rehydrates(self):
        import place_actions
        import runtime_core as core
        import runtime_loop_actions_place as place_handler

        class FakeArm:
            sample_time = 0.001
            last_motion_reason = "move_overcurrent_unrecoverable"
            last_motion_diag = {
                "label": "place_slot_0_descend_near",
                "max_err_deg": 2.166,
                "tol_deg_used": 2.2,
                "settle_time_s": 2.0,
                "motion_state": "ok",
                "last_motion_reason": "move_overcurrent_unrecoverable",
            }

            def __init__(self):
                self.tick_grips = []
                self.goto_calls = []

            def tick_hold(self, *, grip):
                self.tick_grips.append(float(grip))

            def goto_task_space(self, *args, **kwargs):
                self.goto_calls.append((args, kwargs))

        original_execute = place_actions.execute_prompted_place_action
        original_hold_s = getattr(place_actions, "PLACE_OPEN_HOLD_S", None)

        def fake_execute_prompted_place_action(**_kwargs):
            return False, "move_overcurrent_unrecoverable", None, None

        state = core.CycleState()
        state.holding_object = True
        state.current_hold_grip = 0.582
        state.last_pick_return_xyz = [0.53, -0.24, 0.07]
        state.active_target_track_id = 11
        state.pregrasp_pick_lock_color = "orange"
        state.pregrasp_pick_lock_color_conf = 1.0
        state.pregrasp_pick_lock_track_id = 11
        state.pregrasp_pick_lock_uv = [321, 235]
        arm = FakeArm()
        records = []
        hydrate_modes = []
        sync_rows = []

        try:
            place_actions.execute_prompted_place_action = fake_execute_prompted_place_action
            place_actions.PLACE_OPEN_HOLD_S = 0.0
            row = place_handler.handle_place_action(
                action_cmd="place_right_stack",
                state=state,
                arm=arm,
                det=None,
                per=None,
                hold_grip=0.582,
                carry_supervisor=object(),
                centered_pos=(321, 235),
                cube_color="orange",
                color_conf=1.0,
                section_groups={},
                stack_levels={core.SECTION_LEFT_NAME: 0, core.SECTION_RIGHT_NAME: 0},
                section_left_name=core.SECTION_LEFT_NAME,
                section_right_name=core.SECTION_RIGHT_NAME,
                home_pose=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
                place_release_open_grip=0.24,
                place_fail_continue_reasons=tuple(),
                stack_verify_correction_enabled=True,
                stack_verify_require_confirmed_for_advance=True,
                stack_verify_allow_downward_correction=False,
                stack_verify_downward_require_stable_remeasure=True,
                finite_xyz_or_none_fn=core._finite_xyz_or_none,
                clamp_grip_cmd_fn=lambda value: float(value),
                sync_stack_levels_from_authoritative_state=lambda: None,
                run_startup_stack_bootstrap_verify=lambda *, mode="full": (
                    hydrate_modes.append(str(mode)) or {"hydrate_status": "ok"}
                ),
                sync_stack_levels_from_startup_bootstrap=lambda startup_row: sync_rows.append(dict(startup_row)),
                log_ledger_stack_snapshot=lambda _source: None,
                run_post_lift_place_space_refresh=lambda _label: None,
                record_policy_step=lambda *args, **kwargs: records.append((args, kwargs)),
                run_observe_action=lambda: None,
            )
        finally:
            place_actions.execute_prompted_place_action = original_execute
            if original_hold_s is None:
                try:
                    delattr(place_actions, "PLACE_OPEN_HOLD_S")
                except AttributeError:
                    pass
            else:
                place_actions.PLACE_OPEN_HOLD_S = original_hold_s

        self.assertTrue(bool(row.get("handled")))
        self.assertTrue(bool(row.get("break_loop")))
        self.assertEqual([0.24], arm.tick_grips)
        self.assertEqual([], arm.goto_calls)
        self.assertFalse(bool(state.holding_object))
        self.assertEqual(0.0, float(state.current_hold_grip))
        self.assertIsNone(state.active_target_track_id)
        self.assertIsNone(state.last_pick_return_xyz)
        self.assertEqual("unknown", str(state.pregrasp_pick_lock_color))
        self.assertEqual(0.0, float(state.pregrasp_pick_lock_color_conf))
        self.assertIsNone(state.pregrasp_pick_lock_track_id)
        self.assertFalse(bool(state.skip_final_motion))
        self.assertEqual("completed", str(state.stop_reason))
        self.assertEqual(["refresh"], hydrate_modes)
        self.assertEqual([{"hydrate_status": "ok"}], sync_rows)
        self.assertEqual("unknown", str(row.get("cube_color")))
        self.assertEqual(0.0, float(row.get("hold_grip")))
        self.assertIsNone(row.get("centered_pos"))
        self.assertTrue(any("place_collision_recovered" in str(args[1]) for args, _kwargs in records))

    def test_pick_other_block_context_combines_seed_and_persisted_rejects(self):
        import runtime_core as core
        import runtime_loop_observe as observe

        state = core.CycleState()
        state.pick_other_block_track_id = 42
        state.pick_other_block_xyz = [0.50, -0.13, 0.06]
        state.pick_other_block_uv = [322, 237]
        state.pick_other_block_track_ids = [46, 49]
        state.pick_other_block_xyzs = [[0.47, -0.16, 0.03], [float("nan"), 0.0, 0.0]]
        state.pick_other_block_uvs = [[318, 236]]

        track_ids, xyzs, uvs = observe._normalize_pick_other_block_context(state)

        self.assertEqual({42, 46, 49}, track_ids)
        self.assertEqual([[0.5, -0.13, 0.06], [0.47, -0.16, 0.03]], xyzs)
        self.assertEqual([[322, 237], [318, 236]], uvs)

    def test_pick_other_failed_session_persists_rejected_track_and_xyz_blocks(self):
        import runtime_core as core
        import runtime_loop_observe as observe

        state = core.CycleState()
        state.pick_other_block_track_id = 42
        state.pick_other_block_xyz = [0.50, -0.13, 0.06]
        state.pick_other_block_uv = [322, 237]

        observe._remember_pick_other_session_blocks(
            state,
            {
                "blocked_track_ids": [42, 46],
                "blocked_xyzs": [[0.50, -0.13, 0.06], [0.47, -0.16, 0.03]],
            },
        )

        self.assertEqual([42, 46], state.pick_other_block_track_ids)
        self.assertEqual([[0.5, -0.13, 0.06], [0.47, -0.16, 0.03]], state.pick_other_block_xyzs)
        self.assertEqual([[322, 237]], state.pick_other_block_uvs)

    def test_pick_misplaced_section_bound_matches_startup_hydrate_default(self):
        import runtime_core as core

        self.assertAlmostEqual(
            float(core.MISPLACED_PICK_SECTION_MAX_DIST_M),
            float(core.STARTUP_STACK_LOCK_ASSIGN_XY_MARGIN_M),
            places=6,
        )

    def test_pick_misplaced_grasp_offsets_are_correction_only_defaults(self):
        import runtime_core as core

        self.assertAlmostEqual(0.006, float(core.GRASP_STACK_FORWARD_PER_LEVEL_M), places=6)
        self.assertEqual(0.0, float(core.PICK_MISPLACED_GRASP_X_OFFSET_M))
        self.assertEqual(0.0, float(core.PICK_MISPLACED_GRASP_Y_OFFSET_M))
        self.assertGreater(float(core.PICK_MISPLACED_GRASP_Y_PER_LEVEL_M), 0.0)
        self.assertGreater(float(core.PICK_MISPLACED_GRASP_Z_OFFSET_M), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
