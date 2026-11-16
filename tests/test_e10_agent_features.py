"""Offline test of the agent server's E10 feature hook (spec §3.1, gate G4).

Drives AgentSession.execute_step with a scripted fake student and a fake
container, and checks the features returned for the next advice turn. No GPU,
network, Docker or model. Skipped if the agent server's own imports (Flask,
mini-swe-agent's dependencies) are missing.

Run: python -m unittest discover -s tests -p "test_e10_agent_features.py"
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from advisor_models.swe_smith import agent_server
except Exception:  # pragma: no cover - depends on the machine
    agent_server = None

from advisor_models.swe_smith.features import diff_hash


class ExecutionTimeoutError(Exception):
    """Same name as mini-swe-agent's, which is what the server checks."""


class FakeModel:
    n_calls = 0
    cost = 0.0


class FakeStudent:
    """Replays (command, returncode, output) steps; a None command is a format error."""

    def __init__(self, script, container):
        self.script = list(script)
        self.container = container
        self.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
        self.model = FakeModel()

    def add_advice(self, advice):
        self.messages[-1]["content"] += "\n<advice>" + advice + "</advice>"

    def add_message(self, role, content, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})

    def step(self):
        command, returncode, output, new_diff = self.script.pop(0)
        self.model.n_calls += 1
        if command is None:
            self.add_message("assistant", "I will look around.")
            raise ValueError("format error: no bash block")
        self.add_message("assistant", f"THOUGHT\n```bash\n{command}\n```")
        if new_diff is not None:
            self.container.diff = new_diff
        if returncode == "timeout":
            raise ExecutionTimeoutError("timed out")
        self.add_message("user", f"<returncode>{returncode}</returncode>\n<output>\n{output}</output>")
        return {"output": output, "returncode": returncode}


class FakeContainer:
    def __init__(self):
        self.diff = ""

    def execute(self, command, cwd=None):
        assert command.startswith("git diff "), command
        return {"output": self.diff, "returncode": 0}

    def cleanup(self):
        pass


@unittest.skipIf(agent_server is None, "agent server dependencies not installed")
class TestAgentServerFeatures(unittest.TestCase):
    def _session(self, script):
        # __init__ starts the container and the agent; replace both with fakes.
        with mock.patch.object(agent_server.AgentSession, "_initialize", lambda self: None):
            session = agent_server.AgentSession(session_id="t", instance={"instance_id": "x"}, model_name="m")
        session.docker_env = FakeContainer()
        session.original_commit = "abc123"
        session.agent = FakeStudent(script, session.docker_env)
        session._build_observation = lambda: "obs"
        session._log_advisor_feedback = lambda advice: None
        session._log_step_completion = lambda actions, observation: None
        return session

    def test_features_for_next_turn(self):
        pytest_fail = "=== 2 failed, 3 passed in 0.1s ===\n"
        script = [
            ("ls", 0, "a.py\n", None),
            (None, None, None, None),                      # format error: a step, no command
            ("python -m pytest", 1, pytest_fail, None),    # 3a -> 2/20: progress at step 3
            ("sed -i s/a/b/ a.py", 0, "", "diff --git a/a.py"),  # diff changes: progress at step 4
            ("sleep 999", "timeout", None, None),          # timed out: counts as failed
        ]
        session = self._session(script)
        with mock.patch.object(agent_server, "TerminatingException", RuntimeError):
            result = session.execute_step("try the tests", max_steps_per_call=5)

        self.assertFalse(result["terminated"])
        feats = result["features"]
        raw = feats["raw"]
        self.assertEqual(raw["advice_turn"], 1)
        self.assertEqual(raw["steps_used"], 5)
        self.assertEqual(raw["failing_tests"], 2)
        self.assertTrue(raw["tests_run"])
        self.assertEqual(raw["commands_in_window"], 4)   # ls, pytest, sed, sleep
        self.assertEqual(raw["failed_commands_in_window"], 2)  # pytest (1), sleep (timeout)
        self.assertEqual(raw["steps_without_progress"], 1)  # last progress at step 4
        self.assertEqual(feats["inputs"][0], 5 / 40)
        self.assertEqual(feats["inputs"][6], 1 / 9)
        self.assertEqual(session.features.diff_hash, diff_hash("diff --git a/a.py"))

    def test_no_features_when_terminated(self):
        session = self._session([("ls", 0, "", None)])
        session.max_steps = 0  # the student is already at its step cap
        result = session.execute_step("advice", max_steps_per_call=5)
        self.assertTrue(result["terminated"])
        self.assertNotIn("features", result)


if __name__ == "__main__":
    unittest.main()
