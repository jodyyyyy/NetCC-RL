"""Same-device numerical and checkpoint checks for EXP017 graph execution."""

from copy import deepcopy
import os
import json
from pathlib import Path
import pytest
import torch
from netcc_rl.environments import make_env
from netcc_rl.algorithms.proposed import TDMPC2
from netcc_rl.algorithms.proposed.common.buffer import Buffer
from netcc_rl.algorithms.proposed.runner import record
from netcc_rl.matrix import ALL_ARMS, MATRIX, resolved


@pytest.mark.skipif(
    os.environ.get("NETCC_RL_CUDA_ACCEPTANCE") != "1",
    reason="explicit audited GPU required",
)
@pytest.mark.parametrize("name", ["kuramoto", "sis", "swing"])
@pytest.mark.parametrize("arm", ALL_ARMS)
def test_graph_update_matches_eager(name, arm, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(174)
    cfg = resolved(name, arm, "cuda:0")
    cfg.compile = False
    cfg.learner_cuda_graphs = False
    if arm in MATRIX["arms"]:
        cfg.batch_size = 2
    cfg.steps = cfg.buffer_size = 10
    env = make_env(cfg)
    eager = TDMPC2(cfg)
    accelerated_cfg = deepcopy(cfg)
    accelerated_cfg.learner_cuda_graphs = True
    accelerated_cfg.compile = os.environ.get("EXP017_COMPILE_ACCEPTANCE") == "1"
    accelerated = TDMPC2(accelerated_cfg)
    accelerated.model.load_state_dict(eager.model.state_dict())
    obs = env.reset(seed=174)
    rows = [record(obs, action_dim=cfg.action_dim)]
    for _ in range(5):
        action = torch.zeros(cfg.action_dim)
        obs, reward, _, info = env.step(action)
        rows.append(
            record(obs, action, reward, info["terminated"], action_dim=cfg.action_dim)
        )
    buf = Buffer(cfg)
    buf.add(torch.cat(rows))
    diagnostics = []
    for i, starts in enumerate(([0, 1], [0, 4], [0, 1], [0, 4])):
        batch = buf.sample(start_indices=starts * (cfg.batch_size // 2))
        rng = torch.cuda.get_rng_state()
        expected = eager._update(
            *batch["model_batch"], model_valid_mask=batch["model_valid_mask"]
        )
        torch.cuda.set_rng_state(rng)
        actual = accelerated._update(
            *batch["model_batch"], model_valid_mask=batch["model_valid_mask"]
        )
        for k in ("total_loss", "value_loss", "pi_loss", "grad_norm", "pi_grad_norm"):
            torch.testing.assert_close(
                actual[k],
                expected[k],
                rtol=2e-4,
                atol=2e-5,
                msg=lambda msg: f"{name}/{arm}/{i}/{k}: {msg}",
            )
        # Fused FP32 reductions can change near-zero encoder gradients;
        # Adam may amplify those into a few small parameter differences.
        # Keep loss/norm/action/Q checks strict and record parameter drift.
        differences = {}
        for k, value in eager.model.state_dict().items():
            if torch.is_tensor(value):
                differences[k] = float(
                    (accelerated.model.state_dict()[k] - value).abs().max()
                )
        with torch.no_grad():
            probe = batch["model_batch"][0][:-1]
            expected_nodes, expected_z = eager.model.encode_actor_features(probe)
            actual_nodes, actual_z = accelerated.model.encode_actor_features(probe)
            noise = torch.zeros(*expected_z.shape[:-1], cfg.action_dim, device="cuda:0")
            expected_action, _ = eager.model.pi(
                expected_z, node_states=expected_nodes, noise=noise
            )
            actual_action, _ = accelerated.model.pi(
                actual_z, node_states=actual_nodes, noise=noise
            )
            # Q(avg) randomly selects two ensemble heads; match those indices.
            q_rng = torch.cuda.get_rng_state()
            expected_q = eager.model.Q(
                expected_z, expected_action, None, return_type="avg"
            )
            torch.cuda.set_rng_state(q_rng)
            actual_q = accelerated.model.Q(
                actual_z, actual_action, None, return_type="avg"
            )
        diagnostics.append(
            dict(
                update=i,
                max_parameter_difference=max(differences.values()),
                worst_parameters=sorted(
                    differences.items(), key=lambda item: item[1], reverse=True
                )[:5],
                action_max_difference=float(
                    (expected_action - actual_action).abs().max()
                ),
                q_max_difference=float((expected_q - actual_q).abs().max()),
                losses={
                    k: float((actual[k] - expected[k]).abs())
                    for k in (
                        "total_loss",
                        "value_loss",
                        "pi_loss",
                        "grad_norm",
                        "pi_grad_norm",
                    )
                },
            )
        )
        if os.environ.get("NETCC_RL_ACCEL_EVIDENCE"):
            output = Path(os.environ["NETCC_RL_ACCEL_EVIDENCE"])
            output.mkdir(parents=True, exist_ok=True)
            (output / f"{name}-{arm}.json").write_text(
                json.dumps(diagnostics, indent=2) + "\n"
            )
        torch.testing.assert_close(actual_action, expected_action, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(
            actual_q,
            expected_q,
            rtol=2e-4,
            atol=1e-4 if accelerated_cfg.compile else 2e-5,
        )
        torch.testing.assert_close(
            accelerated.scale.value, eager.scale.value, rtol=2e-4, atol=2e-5
        )
        accelerated_parameters = dict(accelerated.model.named_parameters())
        for optimizer, other_optimizer in (
            (eager.optim, accelerated.optim),
            (eager.pi_optim, accelerated.pi_optim),
        ):
            for key, parameter in eager.model.named_parameters():
                if parameter not in optimizer.state:
                    continue
                for state_key, state_value in optimizer.state[parameter].items():
                    torch.testing.assert_close(
                        other_optimizer.state[accelerated_parameters[key]][state_key],
                        state_value,
                        rtol=2e-4,
                        atol=2e-6,
                    )
        for k, value in eager.model.state_dict().items():
            if torch.is_tensor(value):
                torch.testing.assert_close(
                    accelerated.model.state_dict()[k],
                    value,
                    rtol=2e-4,
                    atol=1e-4 if accelerated_cfg.compile else 2e-5,
                    msg=lambda msg: f"{name}/{arm}/{i}/{k}: {msg}",
                )
    meta = accelerated._learner_runtime.metadata()
    assert meta["graphs"] and meta["updates"] == 4
    if accelerated_cfg.compile:
        assert "encoder" in meta["compiled_modules"]
        assert "actor" in meta["compiled_modules"]
    assert meta["counts"]["graph_cache_hits"] > 0
    assert all(torch.isfinite(x) for x in actual.values())
    accelerated.save(tmp_path / "weights.pt")
    assert not any("_orig_mod" in k for k in accelerated.model.state_dict())
    (tmp_path / "acceleration-runtime.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    env.close()
