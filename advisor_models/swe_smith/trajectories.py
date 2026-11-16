"""E10 saved trajectories (spec §6): one record per advice turn, per training step.

Every training rollout is written as state, action and reward, so the training
data can be re-analysed (or reused for offline RL) without rerunning:

    <run_dir>/trajectories/step-<NNNN>.jsonl.gz      one record per advice turn
    <run_dir>/trajectories/students/step-<NNNN>/     per rollout, the agent
        <key>.agent.jsonl                            server's session log and
        <key>.test_report.txt                        the grader's test report

Three parts of the trainer each hold a piece, joined here by trajectory key
(``<instance_id>_<repetition_id>``, upstream's ``TrajectoryID.to_string()``):

  generator  token ids (prompt, response), the response's loss mask, the stop
             reason and the scalar reward after ``zero_reward_on_non_stop``
             (``record_generation``)
  env        the issue id, the six features at each advice turn (§3.1), the
             student's steps, resolved, the env's reward before zeroing, the
             test report (``record_env``)
  trainer    per-token policy and reference log-probs, critic values,
             advantages and returns, snapshotted right after
             ``compute_advantages_and_returns`` (``TensorSnapshot``), before
             upstream rescales the advantages for its loss reduction

Advice turns are the runs of trained tokens (loss mask 1) in a rollout's
response; the observations between them have loss mask 0. The state at a turn
is the decoded prompt plus the response up to that turn: exactly the tokens the
advisor was conditioned on.

The record building and writing are pure Python (no torch, no SkyRL), so they
are unit-tested offline. ``TensorSnapshot.row`` is the only part that touches
tensors.
"""

import gzip
import json
import shutil
import statistics
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1

# Key the generator puts in env_extras so the env knows its trajectory.
EXTRAS_KEY = "e10_trajectory_key"

TOKEN_FIELDS = ("action_log_probs", "base_action_log_probs", "values", "advantages", "returns",
                "rollout_logprobs")


class _Registry:
    """Per-rollout data from the generator and the env, until the step's records are written."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        self._order: Optional[List[str]] = None

    def set_order(self, keys: Sequence[str]) -> None:
        """Keys of the last training batch, in output (= batch row) order."""
        with self._lock:
            self._order = list(keys)

    def pop_order(self) -> Optional[List[str]]:
        with self._lock:
            order, self._order = self._order, None
            return order

    def update(self, key: str, part: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._data.setdefault(key, {})[part] = value

    def pop(self, keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: self._data.pop(k, {}) for k in keys}

    def clear(self) -> int:
        """Drop everything left (e.g. rollouts from trainer evals); returns how many."""
        with self._lock:
            n = len(self._data)
            self._data.clear()
            return n


REGISTRY = _Registry()


def record_generation(key: str, prompt_ids: List[int], response_ids: List[int],
                      loss_mask: List[int], stop_reason: str, reward: float) -> None:
    REGISTRY.update(key, "generation", {
        "prompt_ids": list(prompt_ids),
        "response_ids": list(response_ids),
        "loss_mask": list(loss_mask),
        "stop_reason": stop_reason,
        "reward": float(reward),
    })


def record_env(key: str, env_record: Dict[str, Any]) -> None:
    REGISTRY.update(key, "env", env_record)


def advice_turns(loss_mask: Sequence[int]) -> List[Tuple[int, int]]:
    """[start, end) of each run of trained tokens in a response."""
    turns, start = [], None
    for i, m in enumerate(loss_mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            turns.append((start, i))
            start = None
    if start is not None:
        turns.append((start, len(loss_mask)))
    return turns


def grpo_group_stats(rewards: Sequence[float]) -> Tuple[float, float]:
    """Group mean and std as upstream's GRPO computes them (torch.std: unbiased; one sample: 0, 1)."""
    if len(rewards) <= 1:
        return 0.0, 1.0
    return statistics.fmean(rewards), statistics.stdev(rewards)


def _slice(values: Optional[Sequence[float]], start: int, end: int) -> Optional[List[float]]:
    return None if values is None else [float(v) for v in values[start:end]]


def _at(values: Optional[Sequence[float]], i: int) -> Optional[float]:
    return None if values is None or i >= len(values) else float(values[i])


def build_records(
    run_id: str,
    step: int,
    rows: Sequence[Dict[str, Any]],
    decode: Callable[[List[int]], str],
    algorithm: str,
) -> List[Dict[str, Any]]:
    """Advice-turn records for one training step.

    Args:
        run_id: the E10 run id.
        step: the trainer's global step.
        rows: one per rollout, in batch order, each with
            key, instance_id, repetition_id,
            generation (from ``record_generation``),
            env (from ``record_env``; may be empty if the env never reported),
            tokens: per-token lists over the response for each of TOKEN_FIELDS
              present (missing or None if the trainer did not compute it).
        decode: token ids -> text (the policy's tokenizer).
        algorithm: the advantage estimator ("grpo", "gae", ...).
    """
    groups: Dict[str, List[float]] = {}
    group_index: Dict[str, int] = {}
    for row in rows:
        if row["instance_id"] not in group_index:
            group_index[row["instance_id"]] = len(group_index)
        groups.setdefault(row["instance_id"], []).append(row["generation"]["reward"])

    records = []
    for row in rows:
        gen, env, tok = row["generation"], row.get("env") or {}, row.get("tokens") or {}
        prompt_ids, response_ids = gen["prompt_ids"], gen["response_ids"]
        turns = advice_turns(gen["loss_mask"])
        turn_features = env.get("turn_features") or []
        policy_lp = tok.get("action_log_probs")
        logprob_source = "trainer_before_update"
        if policy_lp is None:
            policy_lp, logprob_source = tok.get("rollout_logprobs"), "rollout"
        mean, std = grpo_group_stats(groups[row["instance_id"]])
        reward = {
            "resolved": env.get("resolved"),
            "student_steps": env.get("student_steps"),
            "env_reward": env.get("env_reward"),
            "reward": gen["reward"],
            "stop_reason": gen["stop_reason"],
            # zero_reward_on_non_stop: a non-"stop" rollout's reward is set to 0;
            # "zeroed" is when that removed a positive env reward.
            "non_stop": gen["stop_reason"] != "stop",
            "zeroed_by_non_stop": bool(gen["stop_reason"] != "stop"
                                       and (env.get("env_reward") or 0.0) > 0.0
                                       and gen["reward"] == 0.0),
        }
        for t, (start, end) in enumerate(turns):
            features = turn_features[t] if t < len(turn_features) else None
            state_ids = prompt_ids + response_ids[:start]
            records.append({
                "schema": SCHEMA_VERSION,
                "run_id": run_id,
                "step": step,
                "trajectory_key": row["key"],
                "issue_id": env.get("issue_id"),
                "group_index": group_index[row["instance_id"]],
                "rollout_index": row["repetition_id"],
                "turn_index": t,
                "n_turns": len(turns),
                "env_turns": len(turn_features),
                "student_step": (features or {}).get("raw", {}).get("steps_used"),
                "state": {
                    "text": decode(state_ids),
                    "prompt_tokens": len(state_ids),
                },
                "action": {
                    "text": decode(response_ids[start:end]),
                    "token_ids": list(response_ids[start:end]),
                    "logprobs_policy": _slice(policy_lp, start, end),
                    "logprobs_policy_source": logprob_source,
                    "logprobs_reference": _slice(tok.get("base_action_log_probs"), start, end),
                },
                "reward": reward,
                "final_reward": gen["reward"],
                "is_last_turn": t == len(turns) - 1,
                "features": features,
                "training": {
                    "algorithm": algorithm,
                    "advantages": _slice(tok.get("advantages"), start, end),
                    "critic_value": _at(tok.get("values"), start),
                    "return": _at(tok.get("returns"), start),
                    "group_mean": mean if algorithm == "grpo" else None,
                    "group_std": std if algorithm == "grpo" else None,
                },
                "student": {
                    "session_id": env.get("session_id"),
                    "agent_log": f"students/step-{step:04d}/{row['key']}.agent.jsonl",
                    "test_report": f"students/step-{step:04d}/{row['key']}.test_report.txt",
                },
            })
    return records


def write_step(trajectories_dir: Path, step: int, records: List[Dict[str, Any]]) -> Path:
    """Write one step's records as gzipped JSON Lines; returns the file path."""
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    path = trajectories_dir / f"step-{step:04d}.jsonl.gz"
    tmp = path.with_suffix(".gz.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def save_student_files(trajectories_dir: Path, step: int, rows: Sequence[Dict[str, Any]],
                       agent_log_dir: Optional[Path]) -> Dict[str, int]:
    """Copy each rollout's agent session log and write its test report beside the records."""
    out = trajectories_dir / "students" / f"step-{step:04d}"
    out.mkdir(parents=True, exist_ok=True)
    counts = {"agent_logs": 0, "agent_logs_missing": 0, "test_reports": 0}
    for row in rows:
        env = row.get("env") or {}
        report = env.get("test_report")
        if report is not None:
            (out / f"{row['key']}.test_report.txt").write_text(report, encoding="utf-8")
            counts["test_reports"] += 1
        session_id = env.get("session_id")
        matches = (sorted(agent_log_dir.glob(f"*_{session_id[:8]}.jsonl"))
                   if agent_log_dir is not None and session_id else [])
        if matches:
            shutil.copyfile(matches[-1], out / f"{row['key']}.agent.jsonl")
            counts["agent_logs"] += 1
        else:
            counts["agent_logs_missing"] += 1
    return counts


class TensorSnapshot:
    """CPU copies of the per-token training tensors, right after advantages are computed."""

    def __init__(self, data: Any) -> None:
        self.tensors = {}
        for name in TOKEN_FIELDS:
            try:
                value = data[name]
            except (KeyError, TypeError):
                value = None
            if value is not None:
                self.tensors[name] = value.detach().float().cpu().clone()

    def row(self, i: int, response_len: int) -> Dict[str, Optional[List[float]]]:
        """Row i's values over its response; response tensors are right-aligned."""
        out: Dict[str, Optional[List[float]]] = {name: None for name in TOKEN_FIELDS}
        for name, t in self.tensors.items():
            out[name] = t[i, t.shape[1] - response_len:].tolist() if response_len else []
        return out
