"""Offline tests of the E10 saved trajectories (spec §6, gate G4).

The trajectory writer's schema on fixture rollouts, the advice-turn split, the
GRPO group statistics, the gzip round trip, the student files, the tensor
snapshot's right alignment, and the env reporting its data. No GPU, network,
Docker or model. The tensor test needs torch and the env test needs
skyrl-gym; each is skipped otherwise.

Run: python -m unittest discover -s tests -p "test_e10_trajectories.py"
"""

import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advisor_models.swe_smith import trajectories as T


def decode(ids):
    return " ".join(str(i) for i in ids)


def _features(turn, steps):
    return {"inputs": [steps / 40] * 7, "names": [], "raw": {"steps_used": steps, "advice_turn": turn},
            "tests_unparsed": False}


def _row(instance, rep, reward, *, env=True, tokens=True, stop="stop"):
    # Response: advice(3) obs(2) advice(2) obs(1) advice(1)
    response = [101, 102, 103, 900, 901, 104, 105, 902, 106]
    mask = [1, 1, 1, 0, 0, 1, 1, 0, 1]
    n = len(response)
    row = {
        "key": f"{instance}_{rep}",
        "instance_id": instance,
        "repetition_id": rep,
        "generation": {"prompt_ids": [1, 2, 3, 4], "response_ids": response, "loss_mask": mask,
                       "stop_reason": stop, "reward": reward},
        "env": {"issue_id": "jd__tenacity." + instance, "session_id": f"sess{instance}{rep}xx",
                "turn_features": [_features(0, 0), _features(1, 5), _features(2, 10)],
                "student_steps": 12, "resolved": reward > 0 or stop != "stop", "env_reward": 0.7,
                "outcome": "completed", "test_report": "1 passed"} if env else None,
        "tokens": {
            "action_log_probs": [-0.1 * (i + 1) for i in range(n)],
            "base_action_log_probs": [-0.2 * (i + 1) for i in range(n)],
            "values": [0.5 + 0.01 * i for i in range(n)],
            "advantages": [0.3] * n,
            "returns": [0.7] * n,
            "rollout_logprobs": None,
        } if tokens else {},
    }
    return row


class TestTurns(unittest.TestCase):
    def test_advice_turns(self):
        self.assertEqual(T.advice_turns([1, 1, 1, 0, 0, 1, 1, 0, 1]), [(0, 3), (5, 7), (8, 9)])
        self.assertEqual(T.advice_turns([0, 0]), [])
        self.assertEqual(T.advice_turns([]), [])
        self.assertEqual(T.advice_turns([0, 1, 1]), [(1, 3)])

    def test_grpo_group_stats_like_torch(self):
        self.assertEqual(T.grpo_group_stats([0.9]), (0.0, 1.0))
        mean, std = T.grpo_group_stats([1.0, 0.0, 0.0, 1.0])
        self.assertAlmostEqual(mean, 0.5)
        self.assertAlmostEqual(std, 0.5773502691896258)  # torch.std, unbiased


class TestRecords(unittest.TestCase):
    def setUp(self):
        self.rows = [_row("a", 0, 0.9), _row("a", 1, 0.0), _row("b", 0, 0.0, stop="length"),
                     _row("b", 1, 0.8, env=False)]
        self.records = T.build_records("exp-10_arm-grpolora", 7, self.rows, decode, "grpo")

    def test_one_record_per_advice_turn(self):
        self.assertEqual(len(self.records), 4 * 3)
        first = self.records[0]
        self.assertEqual((first["step"], first["group_index"], first["rollout_index"], first["turn_index"]),
                         (7, 0, 0, 0))
        self.assertEqual(self.records[6]["group_index"], 1)  # instance "b"
        self.assertEqual([r["is_last_turn"] for r in self.records[:3]], [False, False, True])

    def test_state_and_action(self):
        second = self.records[1]  # rollout a_0, turn 1: advice tokens 5..7
        self.assertEqual(second["state"]["prompt_tokens"], 4 + 5)
        self.assertEqual(second["state"]["text"], decode([1, 2, 3, 4, 101, 102, 103, 900, 901]))
        self.assertEqual(second["action"]["token_ids"], [104, 105])
        self.assertEqual(second["action"]["text"], "104 105")
        self.assertAlmostEqual(second["action"]["logprobs_policy"][0], -0.6)
        self.assertAlmostEqual(second["action"]["logprobs_reference"][1], -1.4)
        self.assertEqual(second["action"]["logprobs_policy_source"], "trainer_before_update")

    def test_training_fields(self):
        second = self.records[1]["training"]
        self.assertEqual(second["advantages"], [0.3, 0.3])
        self.assertAlmostEqual(second["critic_value"], 0.55)  # value at the turn's first token
        self.assertEqual(second["return"], 0.7)
        self.assertAlmostEqual(second["group_mean"], 0.45)  # group "a": 0.9, 0.0
        gae = T.build_records("r", 1, self.rows[:1], decode, "gae")[0]["training"]
        self.assertIsNone(gae["group_mean"])
        self.assertIsNone(gae["group_std"])

    def test_features_and_reward(self):
        second = self.records[1]
        self.assertEqual(second["features"]["raw"]["advice_turn"], 1)
        self.assertEqual(second["student_step"], 5)
        self.assertEqual(second["reward"]["reward"], 0.9)
        self.assertEqual(second["final_reward"], 0.9)
        cut = self.records[6]["reward"]  # b_0: stop "length", env 0.7 -> reward 0
        self.assertTrue(cut["non_stop"])
        self.assertTrue(cut["zeroed_by_non_stop"])

    def test_missing_env_still_records_tokens(self):
        r = self.records[9]  # b_1 has no env data
        self.assertIsNone(r["issue_id"])
        self.assertIsNone(r["features"])
        self.assertEqual(r["env_turns"], 0)
        self.assertEqual(r["action"]["token_ids"], [101, 102, 103])

    def test_rollout_logprobs_fallback(self):
        row = _row("c", 0, 0.0)
        row["tokens"]["action_log_probs"] = None
        row["tokens"]["rollout_logprobs"] = [-1.0] * 9
        rec = T.build_records("r", 1, [row], decode, "grpo")[0]
        self.assertEqual(rec["action"]["logprobs_policy_source"], "rollout")
        self.assertEqual(rec["action"]["logprobs_policy"], [-1.0, -1.0, -1.0])

    def test_spec_fields_present(self):
        r = self.records[0]
        for field in ["run_id", "step", "issue_id", "group_index", "rollout_index", "turn_index",
                      "student_step", "state", "action", "reward", "final_reward", "features",
                      "training", "student"]:
            self.assertIn(field, r)

    def test_write_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = T.write_step(Path(d) / "trajectories", 7, self.records)
            self.assertEqual(path.name, "step-0007.jsonl.gz")
            with gzip.open(path, "rt", encoding="utf-8") as f:
                back = [json.loads(line) for line in f]
        self.assertEqual(back, json.loads(json.dumps(self.records)))


class TestStudentFiles(unittest.TestCase):
    def test_copies_logs_and_reports(self):
        rows = [_row("a", 0, 0.9), _row("a", 1, 0.0)]
        rows[1]["env"]["test_report"] = None
        with tempfile.TemporaryDirectory() as d:
            logs = Path(d) / "agent-logs"
            logs.mkdir()
            (logs / "20261116_120000_jd__tenacity.a_sessa0xx.jsonl").write_text('{"x": 1}\n')
            out = Path(d) / "trajectories"
            counts = T.save_student_files(out, 7, rows, logs)
            step_dir = out / "students" / "step-0007"
            self.assertEqual((step_dir / "a_0.agent.jsonl").read_text(), '{"x": 1}\n')
            self.assertEqual((step_dir / "a_0.test_report.txt").read_text(), "1 passed")
            self.assertFalse((step_dir / "a_1.test_report.txt").exists())
        self.assertEqual(counts, {"agent_logs": 1, "agent_logs_missing": 1, "test_reports": 1})


@unittest.skipIf(importlib.util.find_spec("torch") is None, "torch not installed")
class TestSnapshot(unittest.TestCase):
    def test_right_aligned_rows(self):
        import torch

        data = {
            "advantages": torch.tensor([[0.0, 0.0, 1.0, 2.0], [3.0, 4.0, 5.0, 6.0]]),
            "values": None,
        }
        snap = T.TensorSnapshot(data)
        self.assertEqual(snap.row(0, 2)["advantages"], [1.0, 2.0])
        self.assertEqual(snap.row(1, 4)["advantages"], [3.0, 4.0, 5.0, 6.0])
        self.assertIsNone(snap.row(0, 2)["values"])
        self.assertEqual(snap.row(0, 0)["advantages"], [])


@unittest.skipIf(importlib.util.find_spec("skyrl_gym") is None, "skyrl-gym not installed")
class TestEnvReports(unittest.TestCase):
    def _run(self, steps, eval_score, close_early=False):
        from advisor_models.swe_smith import env as env_module
        from swe_smith_fake_rollout import FakeRemoteAgentClient, extras_for

        rollout = {"instance_id": "jd__tenacity.x", "steps": steps, "patch": "diff", "eval_score": eval_score}
        extras = extras_for(rollout)
        extras[T.EXTRAS_KEY] = "uid-1_3"
        fake = FakeRemoteAgentClient(rollout)
        with mock.patch.object(env_module, "RemoteAgentClient", lambda *a, **k: fake), \
             mock.patch.object(env_module, "compute_score", lambda p, i: (eval_score, "e", "report")):
            env = env_module.SWESmithEnv(None, extras)
            env.init([{"role": "user", "content": "p"}])
            for _ in steps:
                out = env.step("advice")
                if out["done"] or close_early:
                    break
            env.close()
        return T.REGISTRY.pop(["uid-1_3"])["uid-1_3"]["env"]

    def tearDown(self):
        T.REGISTRY.clear()

    def test_completed(self):
        feats = _features(1, 5)
        env = self._run([{"terminated": False, "total_steps": 5, "observation": "o", "features": feats},
                         {"terminated": True, "total_steps": 8}], eval_score=1.0)
        self.assertEqual(env["outcome"], "completed")
        self.assertTrue(env["resolved"])
        self.assertAlmostEqual(env["env_reward"], 0.5 + 0.5 * 32 / 40)
        self.assertEqual(env["student_steps"], 8)
        self.assertEqual(env["test_report"], "report")
        self.assertEqual(env["session_id"], "session-jd__tenacity.x")
        self.assertEqual(len(env["turn_features"]), 2)
        self.assertEqual(env["turn_features"][1], feats)

    def test_cut_by_trainer(self):
        env = self._run([{"terminated": False, "total_steps": 5, "observation": "o"}] * 3,
                        eval_score=1.0, close_early=True)
        self.assertEqual(env["outcome"], "cut")
        self.assertIsNone(env["resolved"])


if __name__ == "__main__":
    unittest.main()
