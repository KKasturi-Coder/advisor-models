"""E10 six-feature state (spec §3.1), computed at each advice turn.

Sagnik's six features, as seven inputs in [0, 1], from what the student has
done and seen in its own run: its commands, their exit codes and outputs, and
the repo's working tree. Never from the hidden tests or the eval server.

  1  steps used              s / 40
  2  steps remaining         (40 - s) / 40
  3a failing tests           failed + errored count of the most recent test
                             command, min(count, 20) / 20; 0 if none has run
  3b no tests run yet        1 if no test command has run, else 0
  4  command failures        share of the last 5 commands with a non-zero exit
                             (over the commands run, if fewer; 0 if none)
  5  steps without progress  steps since the last progress event / 40.
                             Progress = the working-tree diff hash changes, or
                             input 3a changes. Counted from step 0 if neither
                             has happened.
  6  advice turn index       k / 9, k = 0 for the first advice turn

If a test command's output cannot be parsed, input 3a keeps its previous value
and the snapshot flags ``tests_unparsed``. Output the student did not see (the
agent shows a warning instead of output of 10,000 characters or more) counts as
unparsed.

Pure Python, no SkyRL or Docker import, so it is unit-tested offline.
"""

import hashlib
import re
from typing import Dict, List, Optional

MAX_STEPS = 40
MAX_ADVICE_TURNS = 9
FAILING_CAP = 20
EXIT_WINDOW = 5
# mini-swe-agent's action_observation_template shows the output only below
# this length (config/default.yaml), and a warning otherwise.
SEEN_OUTPUT_LIMIT = 10000

INPUT_NAMES = (
    "steps_used",
    "steps_remaining",
    "failing_tests",
    "no_tests_yet",
    "command_failures",
    "steps_without_progress",
    "advice_turn",
)

_TEST_COMMAND = re.compile(
    r"(?:^|[\s;&|(`])(?:"
    r"pytest|py\.test|tox|nox"
    r"|python[0-9.]*\s+-m\s+(?:pytest|unittest)"
    r")(?=$|[\s;&|)`])"
)
# pytest's summary, with or without "=" rules: "2 failed, 5 passed, 1 error in 0.3s"
_PYTEST_SUMMARY = re.compile(r"\b\d+\s+(?:failed|passed|errors?|skipped|xfailed|xpassed|deselected)\b.*?\bin\s+[\d.]+\s*s")
_PYTEST_COUNT = re.compile(r"\b(\d+)\s+(failed|errors?)\b")
_PYTEST_NO_TESTS = re.compile(r"\bno tests ran\b")
# unittest: "Ran 4 tests in 0.01s" then "OK" or "FAILED (failures=1, errors=2)"
_UNITTEST_RAN = re.compile(r"^Ran \d+ tests? in [\d.]+s$", re.MULTILINE)
_UNITTEST_FAILED = re.compile(r"^FAILED \(([^)]*)\)", re.MULTILINE)
_UNITTEST_OK = re.compile(r"^OK\b", re.MULTILINE)
_UNITTEST_COUNT = re.compile(r"\b(failures|errors)=(\d+)")


def is_test_command(command: str) -> bool:
    """True if the command runs tests: pytest, python -m pytest/unittest, tox or nox."""
    return bool(_TEST_COMMAND.search(command))


def parse_failing_count(output: str) -> Optional[int]:
    """Failed plus errored tests from a test command's output; None if unparsable.

    Uses the last summary in the output (tox and nox can wrap several runs).
    """
    pytest_lines = [line for line in output.splitlines()
                    if _PYTEST_SUMMARY.search(line) or _PYTEST_NO_TESTS.search(line)]
    unittest_end = None
    if _UNITTEST_RAN.search(output):
        failed = list(_UNITTEST_FAILED.finditer(output))
        oks = list(_UNITTEST_OK.finditer(output))
        last_failed = failed[-1].start() if failed else -1
        last_ok = oks[-1].start() if oks else -1
        if last_failed > last_ok:
            unittest_end = (last_failed, sum(int(n) for _, n in _UNITTEST_COUNT.findall(failed[-1].group(1))))
        elif last_ok >= 0:
            unittest_end = (last_ok, 0)

    pytest_end = None
    if pytest_lines:
        line = pytest_lines[-1]
        count = sum(int(n) for n, _ in _PYTEST_COUNT.findall(line))
        pytest_end = (output.rfind(line), count)

    ends = [e for e in (pytest_end, unittest_end) if e is not None]
    if not ends:
        return None
    return max(ends)[1]


def diff_hash(diff: str) -> str:
    """Hash of a working-tree diff, for the progress feature."""
    return hashlib.sha256(diff.encode("utf-8", errors="replace")).hexdigest()


class FeatureTracker:
    """Follows one student run step by step and gives the features at each advice turn."""

    def __init__(self) -> None:
        self.steps = 0
        self.exit_codes: List[Optional[int]] = []
        self.failing_count: Optional[int] = None  # raw count of the last parsed test command
        self.tests_run = False
        self.last_test_unparsed = False
        # The session starts at the instance's commit, so the diff is empty.
        self.diff_hash = diff_hash("")
        self.last_progress_step = 0

    def _failing_input(self) -> float:
        if self.failing_count is None:
            return 0.0
        return min(self.failing_count, FAILING_CAP) / FAILING_CAP

    def record_step(
        self,
        command: Optional[str],
        returncode: Optional[int],
        output: Optional[str],
        diff_hash: Optional[str],
    ) -> None:
        """One student step.

        Args:
            command: the shell command the student ran, or None if the step ran
                none (a format error, a failed model call).
            returncode: its exit code; None if it timed out (counted as a failure).
            output: its output as the command returned it.
            diff_hash: hash of the working-tree diff after the step; None if it
                could not be read (then the diff is taken as unchanged).
        """
        self.steps += 1
        before = self._failing_input()

        if command is not None:
            self.exit_codes.append(returncode)
            if is_test_command(command):
                self.tests_run = True
                seen = output is not None and len(output) < SEEN_OUTPUT_LIMIT
                count = parse_failing_count(output) if seen else None
                if count is None:
                    self.last_test_unparsed = True
                else:
                    self.failing_count = count
                    self.last_test_unparsed = False

        progressed = self._failing_input() != before
        if diff_hash is not None and diff_hash != self.diff_hash:
            progressed = True
            self.diff_hash = diff_hash
        if progressed:
            self.last_progress_step = self.steps

    def snapshot(self, advice_turn: int) -> Dict[str, object]:
        """Features at an advice turn, before the advice is generated."""
        s = min(self.steps, MAX_STEPS)
        window = self.exit_codes[-EXIT_WINDOW:]
        n_failed = sum(code != 0 for code in window)
        raw = {
            "steps_used": s,
            "failing_tests": self.failing_count if self.failing_count is not None else 0,
            "tests_run": self.tests_run,
            "commands_in_window": len(window),
            "failed_commands_in_window": n_failed,
            "steps_without_progress": s - min(self.last_progress_step, s),
            "advice_turn": advice_turn,
        }
        inputs = [
            s / MAX_STEPS,
            (MAX_STEPS - s) / MAX_STEPS,
            self._failing_input(),
            0.0 if self.tests_run else 1.0,
            n_failed / len(window) if window else 0.0,
            raw["steps_without_progress"] / MAX_STEPS,
            advice_turn / MAX_ADVICE_TURNS,
        ]
        return {
            "inputs": inputs,
            "names": list(INPUT_NAMES),
            "raw": raw,
            "tests_unparsed": self.last_test_unparsed,
        }
