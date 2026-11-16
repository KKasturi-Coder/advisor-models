"""E10 trainer: upstream SkyRL's RayPPOTrainer plus saved trajectories (spec §6).

Training is upstream's, unchanged. Two additions, both bookkeeping:

  E10RayPPOTrainer      keeps CPU copies of the per-token log-probs, values,
                        advantages and returns right after
                        ``compute_advantages_and_returns``. Upstream later
                        rescales ``advantages`` in place for its loss
                        reduction, so the copy is what the estimator gave.
  E10TrajectoryWriter   on ``on_step_end``, joins that copy with the
                        generator's and the env's per-rollout data and writes
                        ``<run_dir>/trajectories/step-<NNNN>.jsonl.gz`` plus
                        the students' session logs and test reports.

A step whose records cannot be written fails the run: a run without its
trajectories is incomplete (spec §6, G3).

Environment variables:
  E10_RUN_DIR         run directory (default: trainer.export_path)
  E10_RUN_ID          run id in every record (default: trainer.run_name)
  E10_AGENT_LOG_DIR   the agent server's --log-dir, to copy session logs from
                      (unset: the records say where the logs would be, and the
                      step's log line counts them as missing)
"""

import os
from pathlib import Path

from loguru import logger
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils.callbacks import TrainingCallback

from . import trajectories


class E10RayPPOTrainer(RayPPOTrainer):
    e10_snapshot = None

    def compute_advantages_and_returns(self, data):
        data = super().compute_advantages_and_returns(data)
        self.e10_snapshot = trajectories.TensorSnapshot(data)
        return data


class E10TrajectoryWriter(TrainingCallback):
    def __init__(self, cfg):
        self.run_dir = Path(os.environ.get("E10_RUN_DIR") or cfg.trainer.export_path)
        self.run_id = os.environ.get("E10_RUN_ID") or cfg.trainer.run_name
        log_dir = os.environ.get("E10_AGENT_LOG_DIR")
        self.agent_log_dir = Path(log_dir) if log_dir else None
        self.algorithm = cfg.trainer.algorithm.advantage_estimator
        self.trajectories_dir = self.run_dir / "trajectories"

    def on_step_end(self, trainer, callback_input, control):
        step = callback_input.global_step
        order = trajectories.REGISTRY.pop_order()
        if callback_input.trajectory_ids is not None:
            order = [tid.to_string() for tid in callback_input.trajectory_ids]
        snapshot = getattr(trainer, "e10_snapshot", None)
        if not order or snapshot is None:
            raise RuntimeError(f"E10 trajectories: step {step} has no batch order or tensor snapshot")

        parts = trajectories.REGISTRY.pop(order)
        rows, missing_env = [], 0
        for i, key in enumerate(order):
            gen = parts[key].get("generation")
            if gen is None:
                raise RuntimeError(f"E10 trajectories: step {step}, rollout {key} has no generation data")
            env = parts[key].get("env")
            missing_env += env is None
            instance_id, _, repetition = key.rpartition("_")
            rows.append({
                "key": key,
                "instance_id": instance_id,
                "repetition_id": int(repetition),
                "generation": gen,
                "env": env,
                "tokens": snapshot.row(i, len(gen["response_ids"])),
            })
        trainer.e10_snapshot = None

        records = trajectories.build_records(self.run_id, step, rows, trainer.tokenizer.decode, self.algorithm)
        path = trajectories.write_step(self.trajectories_dir, step, records)
        student = trajectories.save_student_files(self.trajectories_dir, step, rows, self.agent_log_dir)
        leftover = trajectories.REGISTRY.clear()
        logger.info(
            f"E10 trajectories: step={step} rollouts={len(rows)} records={len(records)} "
            f"missing_env={missing_env} agent_logs={student['agent_logs']} "
            f"agent_logs_missing={student['agent_logs_missing']} test_reports={student['test_reports']} "
            f"leftover={leftover} bytes={path.stat().st_size} file={path}"
        )
