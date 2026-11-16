"""Offline test of the E10 trajectory hooks in the generator and the trainer (spec §6, G4).

Runs one fake training step through the real hook code: the generator records
a batch, envs report, the trainer snapshots its tensors, and the writer
callback joins them and writes the step file. Upstream SkyRL's base classes
are replaced by minimal stand-ins (only the methods the hooks call), so this
runs without SkyRL; with upstream installed, the generator test in
test_rollout_counts.py covers the real base class. Needs torch (loguru is
stubbed if missing).

Run: python -m unittest discover -s tests -p "test_e10_trajectory_hooks.py"
"""

import gzip
import importlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HAVE_TORCH = importlib.util.find_spec("torch") is not None


@dataclass
class TrajectoryID:
    instance_id: str
    repetition_id: int

    def to_string(self):
        return f"{self.instance_id}_{self.repetition_id}"


@dataclass
class BatchMetadata:
    global_step: int
    training_phase: str


class _StubGenerator:
    def _setup_env_extras(self, env_class, env_extras, sampling_params, trajectory_id):
        return env_extras


class _StubTrainer:
    def compute_advantages_and_returns(self, data):
        return data


def _stub_modules():
    mods = {}
    for name in ["skyrl", "skyrl.train", "skyrl.train.generators", "skyrl.train.generators.base",
                 "skyrl.train.generators.skyrl_gym_generator", "skyrl.train.trainer",
                 "skyrl.train.utils", "skyrl.train.utils.callbacks"]:
        mods[name] = types.ModuleType(name)
    mods["skyrl.train.generators.base"].GeneratorInput = dict
    mods["skyrl.train.generators.base"].GeneratorOutput = dict
    mods["skyrl.train.generators.skyrl_gym_generator"].SkyRLGymGenerator = _StubGenerator
    mods["skyrl.train.trainer"].RayPPOTrainer = _StubTrainer
    mods["skyrl.train.utils.callbacks"].TrainingCallback = object
    if importlib.util.find_spec("loguru") is None:
        mods["loguru"] = types.ModuleType("loguru")
        mods["loguru"].logger = types.SimpleNamespace(info=lambda *a, **k: None)
    return mods


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestHooks(unittest.TestCase):
    def setUp(self):
        self.patch = mock.patch.dict(sys.modules, _stub_modules())
        self.patch.start()
        for name in ["advisor_models.swe_smith.generator_upstream", "advisor_models.swe_smith.trainer_upstream"]:
            sys.modules.pop(name, None)
        self.gen = importlib.import_module("advisor_models.swe_smith.generator_upstream")
        self.trn = importlib.import_module("advisor_models.swe_smith.trainer_upstream")
        from advisor_models.swe_smith import trajectories
        self.T = trajectories
        self.T.REGISTRY.clear()

    def tearDown(self):
        self.T.REGISTRY.clear()
        for name in ["advisor_models.swe_smith.generator_upstream", "advisor_models.swe_smith.trainer_upstream"]:
            sys.modules.pop(name, None)
        self.patch.stop()

    def test_env_extras_carry_the_key(self):
        g = self.gen.E10SkyRLGymGenerator.__new__(self.gen.E10SkyRLGymGenerator)
        extras = g._setup_env_extras("swe_smith", {}, None, TrajectoryID("uid", 2))
        self.assertEqual(extras[self.T.EXTRAS_KEY], "uid_2")

    def test_eval_batches_are_not_recorded(self):
        batch = {"batch_metadata": BatchMetadata(3, "eval"), "trajectory_ids": [TrajectoryID("u", 0)]}
        self.gen.E10SkyRLGymGenerator._record_trajectories(batch, {"response_ids": [[1]]})
        self.assertIsNone(self.T.REGISTRY.pop_order())

    def test_one_step_end_to_end(self):
        import torch

        ids = [TrajectoryID("u1", 0), TrajectoryID("u1", 1), TrajectoryID("u2", 0)]
        # Responses of length 5, 3 and 4; loss masks include one observation run each.
        output = {
            "prompt_token_ids": [[1, 2], [1, 2], [3]],
            "response_ids": [[10, 11, 90, 12, 13], [20, 91, 21], [30, 31, 92, 32]],
            "loss_masks": [[1, 1, 0, 1, 1], [1, 0, 1], [1, 1, 0, 1]],
            "stop_reasons": ["stop", "stop", "length"],
            "rewards": [0.9, [0.0, 0.0, 0.0], 0.0],
        }
        batch = {"batch_metadata": BatchMetadata(5, "train"), "trajectory_ids": ids}
        self.gen.E10SkyRLGymGenerator._record_trajectories(batch, output)

        for tid, outcome in [(ids[0], "completed"), (ids[2], "cut")]:
            self.T.record_env(tid.to_string(), {"issue_id": "iss-" + tid.instance_id,
                                                "session_id": None, "outcome": outcome,
                                                "turn_features": [], "env_reward": 0.7})

        # Right-aligned (batch, max_response=5) tensors, plus one padding row.
        adv = torch.zeros(4, 5)
        adv[0, :] = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
        adv[1, 2:] = torch.tensor([-1.0, 0.0, -1.0])
        adv[2, 1:] = torch.tensor([2.0, 2.0, 0.0, 2.0])
        trainer = self.trn.E10RayPPOTrainer()
        trainer.tokenizer = types.SimpleNamespace(decode=lambda xs: ",".join(map(str, xs)))
        trainer.compute_advantages_and_returns({"advantages": adv, "action_log_probs": -adv.abs(),
                                                "values": None})

        with tempfile.TemporaryDirectory() as d:
            cfg = types.SimpleNamespace(trainer=types.SimpleNamespace(
                export_path=d, run_name="exp-10_arm-grpolora_split-tenacitytrain_seed-0",
                algorithm=types.SimpleNamespace(advantage_estimator="grpo")))
            writer = self.trn.E10TrajectoryWriter(cfg)
            cb_input = types.SimpleNamespace(global_step=5, trajectory_ids=None)
            writer.on_step_end(trainer, cb_input, None)
            with gzip.open(Path(d) / "trajectories" / "step-0005.jsonl.gz", "rt") as f:
                records = [json.loads(line) for line in f]

        # 2 + 2 + 2 advice turns.
        self.assertEqual(len(records), 6)
        by = {(r["trajectory_key"], r["turn_index"]): r for r in records}
        self.assertEqual(by[("u1_0", 1)]["action"]["token_ids"], [12, 13])
        self.assertEqual(by[("u1_0", 1)]["training"]["advantages"], [1.0, 1.0])
        self.assertEqual(by[("u1_1", 0)]["training"]["advantages"], [-1.0])
        self.assertEqual(by[("u1_1", 0)]["final_reward"], 0.0)  # per-token rewards summed
        self.assertEqual(by[("u2_0", 1)]["action"]["token_ids"], [32])
        self.assertEqual(by[("u2_0", 1)]["training"]["advantages"], [2.0])
        self.assertEqual(by[("u2_0", 0)]["state"]["text"], "3")
        self.assertEqual(by[("u2_0", 0)]["group_index"], 1)
        self.assertEqual(by[("u2_0", 0)]["issue_id"], "iss-u2")
        self.assertIsNone(by[("u1_1", 0)]["issue_id"])  # this env never reported
        self.assertIsNone(by[("u1_0", 0)]["training"]["critic_value"])
        self.assertEqual(by[("u1_0", 0)]["run_id"], "exp-10_arm-grpolora_split-tenacitytrain_seed-0")
        self.assertIsNone(trainer.e10_snapshot)
        self.assertIsNone(self.T.REGISTRY.pop_order())

    def test_step_without_snapshot_fails(self):
        self.T.REGISTRY.set_order(["u_0"])
        trainer = self.trn.E10RayPPOTrainer()
        cfg = types.SimpleNamespace(trainer=types.SimpleNamespace(
            export_path="/nonexistent", run_name="r",
            algorithm=types.SimpleNamespace(advantage_estimator="grpo")))
        with self.assertRaises(RuntimeError):
            self.trn.E10TrajectoryWriter(cfg).on_step_end(
                trainer, types.SimpleNamespace(global_step=1, trajectory_ids=None), None)


if __name__ == "__main__":
    unittest.main()
