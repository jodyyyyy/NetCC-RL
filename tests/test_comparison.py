"""Common protocol and action-bound regression checks."""

from pathlib import Path

import numpy as np
import pytest
import torch

from netcc_rl.catalog import ALGORITHMS, entries
from netcc_rl.cli import read_config
from netcc_rl.algorithms.baselines.cmat.task_adapter import TaskAdapter
from netcc_rl.algorithms.baselines.cmat.gpu_batch import ContinuousBatch
from netcc_rl.comparison_output import checkpoint_index, evaluation, training_row

ROOT = Path(__file__).resolve().parents[1] / "configs"


def test_config_catalog_and_common_schema():
    rows = list(entries(ROOT, validate=True))
    assert len(rows) == len({(r["environment"], r["identity"]) for r in rows}) == 60
    for name in ("kuramoto", "sis", "swing"):
        configs = [read_config(ROOT/"comparison"/f"{name}-{algo}.yaml") for algo in ALGORITHMS]
        assert all(c.keys() == configs[0].keys() for c in configs)
        assert all(c["drivers"] == configs[0]["drivers"] for c in configs)
        assert len({c["snapshot_sha256"] for c in configs}) == 1


def test_unified_cmat_sis_protocol_and_cost_once():
    selected = read_config(ROOT/"comparison/sis-cmat.yaml")
    primary = TaskAdapter(selected)
    old = TaskAdapter(read_config(ROOT/"historical/sis-cmat.yaml"))
    assert primary.controls == old.controls
    assert primary.snapshot["reward_variant"] == "sis_control_cost"
    assert old.snapshot["reward_variant"] != primary.snapshot["reward_variant"]
    assert (primary.low, primary.high, old.low, old.high) == (-1, 1, -1, .2)
    assert primary.identity != old.identity
    env = primary.make_env()
    try:
        for action, intensity in ((-1., 0.), (1., 1.)):
            env.reset(seed=202600)
            _, reward, terminated, truncated, info = env.step(np.full(primary.action_dim, action, np.float32))
            assert not (terminated or truncated)
            assert info["mean_control_intensity"] == intensity
            assert info["control_cost"] == pytest.approx(.1*intensity)
            assert reward == pytest.approx(info["infection_reward"] - .1*intensity)
            simulator = ContinuousBatch(primary, 2, device="cpu")
            simulator.reset()
            _, batch_reward, _, _, stats = simulator.step(torch.full((2,primary.action_dim), action))
            torch.testing.assert_close(stats["mean_control_intensity"],torch.full((2,),intensity))
            torch.testing.assert_close(batch_reward,stats["infection_reward"]-.1*intensity)
    finally:
        env.close()


def test_comparison_result_schema(tmp_path):
    cfg = read_config(ROOT / "comparison/sis-cmat.yaml")
    assert training_row(cfg, 32, {"loss": .5})["real_steps"] == 32
    report = evaluation(cfg, [{"seed": 0, "return": 1}, {"seed": 1, "return": 3}])
    assert report["mean_return"] == 2
    weights = tmp_path / "final.pt"
    weights.write_bytes(b"weights")
    checkpoint_index(cfg,tmp_path,weights,32,0)
    assert (tmp_path / "checkpoint.json").is_file()
