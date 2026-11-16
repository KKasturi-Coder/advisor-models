"""Offline tests of the E10 six-feature state (spec §3.1, gate G4).

The test-output parser, the exit-code window, the progress hash and the
no-tests case, on fixture student runs. No GPU, network, Docker or model.

Run: python -m unittest discover -s tests -p "test_e10_features.py"
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advisor_models.swe_smith.features import (
    FeatureTracker,
    SEEN_OUTPUT_LIMIT,
    diff_hash,
    is_test_command,
    parse_failing_count,
)

PYTEST_FAIL = """\
============================= test session starts ==============================
collected 123 items

tests/test_tenacity.py ....F..E....                                       [100%]

=========================== short test summary info ============================
FAILED tests/test_tenacity.py::TestRetry::test_stop - AssertionError
ERROR tests/test_tenacity.py::TestWait::test_fixed - TypeError
=================== 2 failed, 119 passed, 1 error in 1.92s ====================
"""
PYTEST_PASS = "========== 123 passed, 3 warnings in 1.80s ==========\n"
PYTEST_QUIET = "....F.\n1 failed, 5 passed in 0.12s\n"
PYTEST_NO_TESTS = "======== no tests ran in 0.01s ========\n"
PYTEST_COLLECTION = """\
ERROR collecting tests/test_tenacity.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 0.30s ===============================
"""
UNITTEST_FAIL = """\
..F.E
----------------------------------------------------------------------
Ran 5 tests in 0.004s

FAILED (failures=1, errors=1)
"""
UNITTEST_OK = "....\n----------------------------------------------------------------------\nRan 4 tests in 0.002s\n\nOK\n"
TOX_TWO_RUNS = PYTEST_FAIL + "py310: commands failed\n" + PYTEST_PASS + "  py311: OK\n"


class TestCommands(unittest.TestCase):
    def test_matches(self):
        for cmd in [
            "pytest",
            "pytest tests/test_tenacity.py -x -q",
            "cd /testbed && python -m pytest -k retry",
            "python3 -m unittest discover",
            "python3.10 -m pytest",
            "tox -e py310",
            "nox -s tests",
            "py.test tests",
            "(cd /testbed; pytest)",
        ]:
            with self.subTest(cmd=cmd):
                self.assertTrue(is_test_command(cmd))

    def test_non_matches(self):
        for cmd in [
            "cat tests/test_tenacity.py",
            "grep -rn pytest_plugins .",
            "pip install pytest-cov",
            "python reproduce.py",
            "ls toxics/",
            "sed -n '1,20p' noxfile.py",
        ]:
            with self.subTest(cmd=cmd):
                self.assertFalse(is_test_command(cmd))


class TestParser(unittest.TestCase):
    def test_counts(self):
        cases = {
            "pytest failed + error": (PYTEST_FAIL, 3),
            "pytest all passed": (PYTEST_PASS, 0),
            "pytest -q": (PYTEST_QUIET, 1),
            "pytest no tests": (PYTEST_NO_TESTS, 0),
            "pytest collection error": (PYTEST_COLLECTION, 1),
            "unittest failed": (UNITTEST_FAIL, 2),
            "unittest ok": (UNITTEST_OK, 0),
            "tox, last run wins": (TOX_TWO_RUNS, 0),
        }
        for name, (output, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(parse_failing_count(output), expected)

    def test_unparsable(self):
        for output in ["", "Traceback (most recent call last):\n  ...\nKeyboardInterrupt\n",
                       "bash: pytest: command not found\n"]:
            with self.subTest(output=output[:30]):
                self.assertIsNone(parse_failing_count(output))


def _inputs(snapshot):
    return dict(zip(snapshot["names"], snapshot["inputs"]))


class TestTracker(unittest.TestCase):
    def test_first_turn(self):
        snap = FeatureTracker().snapshot(advice_turn=0)
        self.assertEqual(snap["inputs"], [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.assertFalse(snap["tests_unparsed"])

    def test_no_tests_case(self):
        t = FeatureTracker()
        for _ in range(5):
            t.record_step("ls", 0, "a\nb\n", diff_hash(""))
        x = _inputs(t.snapshot(advice_turn=1))
        self.assertEqual(x["failing_tests"], 0.0)
        self.assertEqual(x["no_tests_yet"], 1.0)
        self.assertEqual(x["steps_used"], 5 / 40)
        self.assertEqual(x["steps_remaining"], 35 / 40)
        self.assertEqual(x["steps_without_progress"], 5 / 40)
        self.assertEqual(x["advice_turn"], 1 / 9)

    def test_failing_tests_capped_and_raw_kept(self):
        t = FeatureTracker()
        t.record_step("pytest", 1, "====== 25 failed, 98 passed in 2.0s ======\n", None)
        snap = t.snapshot(advice_turn=1)
        self.assertEqual(_inputs(snap)["failing_tests"], 1.0)
        self.assertEqual(snap["raw"]["failing_tests"], 25)
        self.assertEqual(_inputs(snap)["no_tests_yet"], 0.0)

    def test_exit_window(self):
        t = FeatureTracker()
        self.assertEqual(_inputs(t.snapshot(0))["command_failures"], 0.0)
        t.record_step("false", 1, "", None)
        t.record_step("true", 0, "", None)
        self.assertEqual(_inputs(t.snapshot(1))["command_failures"], 1 / 2)
        for _ in range(4):
            t.record_step("true", 0, "", None)
        # Last 5 commands: all exit 0; the early failure has left the window.
        self.assertEqual(_inputs(t.snapshot(2))["command_failures"], 0.0)
        t.record_step("sleep 999", None, "", None)  # timed out: a failure
        snap = t.snapshot(3)
        self.assertEqual(_inputs(snap)["command_failures"], 1 / 5)
        self.assertEqual(snap["raw"]["commands_in_window"], 5)

    def test_steps_without_a_command_count_as_steps_only(self):
        t = FeatureTracker()
        t.record_step(None, None, None, None)  # e.g. a format error
        snap = t.snapshot(1)
        self.assertEqual(snap["raw"]["steps_used"], 1)
        self.assertEqual(snap["raw"]["commands_in_window"], 0)

    def test_progress_from_diff(self):
        t = FeatureTracker()
        t.record_step("cat a.py", 0, "", diff_hash(""))
        t.record_step("sed -i s/a/b/ a.py", 0, "", diff_hash("diff --git a/a.py"))
        t.record_step("cat a.py", 0, "", diff_hash("diff --git a/a.py"))
        t.record_step("cat b.py", 0, "", None)  # diff unreadable: unchanged
        self.assertEqual(t.snapshot(1)["raw"]["steps_without_progress"], 2)

    def test_progress_from_failing_tests(self):
        t = FeatureTracker()
        t.record_step("pytest", 1, PYTEST_FAIL, None)  # 3a: 0 -> 3/20, progress at step 1
        t.record_step("ls", 0, "", None)
        t.record_step("pytest", 1, PYTEST_FAIL, None)  # same count: no progress
        self.assertEqual(t.snapshot(1)["raw"]["steps_without_progress"], 2)
        t.record_step("pytest", 0, PYTEST_PASS, None)  # 3/20 -> 0: progress at step 4
        self.assertEqual(t.snapshot(2)["raw"]["steps_without_progress"], 0)

    def test_passing_first_run_is_not_progress(self):
        # 3a stays 0 when the first test run passes; only 3b changes.
        t = FeatureTracker()
        t.record_step("pytest", 0, PYTEST_PASS, None)
        self.assertEqual(t.snapshot(1)["raw"]["steps_without_progress"], 1)

    def test_unparsed_keeps_previous_value(self):
        t = FeatureTracker()
        t.record_step("pytest", 1, PYTEST_FAIL, None)
        t.record_step("pytest", 2, "Traceback ...\nKeyboardInterrupt\n", None)
        snap = t.snapshot(1)
        self.assertEqual(_inputs(snap)["failing_tests"], 3 / 20)
        self.assertTrue(snap["tests_unparsed"])
        t.record_step("pytest", 0, PYTEST_PASS, None)
        self.assertFalse(t.snapshot(2)["tests_unparsed"])

    def test_output_the_student_did_not_see_is_unparsed(self):
        t = FeatureTracker()
        long_output = "x" * SEEN_OUTPUT_LIMIT + "\n" + PYTEST_PASS
        t.record_step("pytest", 0, long_output, None)
        snap = t.snapshot(1)
        self.assertTrue(snap["tests_unparsed"])
        self.assertEqual(_inputs(snap)["no_tests_yet"], 0.0)
        self.assertEqual(_inputs(snap)["failing_tests"], 0.0)

    def test_inputs_in_unit_interval(self):
        t = FeatureTracker()
        for i in range(45):  # past the 40-step cap
            t.record_step("pytest", 1, PYTEST_FAIL, diff_hash(str(i % 3)))
        snap = t.snapshot(advice_turn=9)
        self.assertEqual(len(snap["inputs"]), 7)
        for name, v in zip(snap["names"], snap["inputs"]):
            with self.subTest(name=name):
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)


if __name__ == "__main__":
    unittest.main()
