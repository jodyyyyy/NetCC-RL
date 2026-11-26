"""EXP-017 real Replay acceptance and factor contracts, on CPU and explicit CUDA."""

from copy import deepcopy
from pathlib import Path
import json
import os
import subprocess
import sys
import numpy as np
import pytest
import torch
import yaml
from netcc_rl.environments import make_env
from netcc_rl.algorithms.proposed import TDMPC2
from netcc_rl.algorithms.proposed.config import (
    load_config,
    actor_metadata,
    validate,
    validate_experiment_output,
)
from netcc_rl.algorithms.proposed.common.buffer import Buffer
from netcc_rl.algorithms.proposed.common.adaptive_architecture import digest
from netcc_rl.algorithms.proposed.common.node_features import node_features
from netcc_rl.algorithms.proposed.runner import record
from netcc_rl.matrix import (
    ALL_ARMS,
    MATRIX,
    ROOT,
    encoder_parameter_comparison,
    resolved,
)


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(174)


DEVICES = [
    "cpu",
    pytest.param(
        "cuda:0",
        marks=pytest.mark.skipif(
            os.environ.get("NETCC_RL_CUDA_ACCEPTANCE") != "1"
            or not torch.cuda.is_available(),
            reason="requires explicit CUDA acceptance on audited UUID",
        ),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", MATRIX["environments"])
@pytest.mark.parametrize("arm", ALL_ARMS)
def test_real_replay_all_arms(name, arm, device, tmp_path, monkeypatch):
    cfg = resolved(name, arm, device)
    cfg.steps = cfg.buffer_size = 10
    cfg.batch_size = 2
    env = make_env(cfg)
    agent = TDMPC2(cfg)
    model = agent.model
    obs = env.reset(seed=174)
    batch = obs.to(device).expand(2, -1)
    h, z = model.encode_features(batch)
    n, f = cfg.architecture_resolved["N"], cfg.architecture_resolved["F"]
    assert z.shape == (2, 64) and torch.isfinite(z).all()
    enc = model._encoder["state"]
    hcount = None
    if cfg.state_encoder_type == "global_mlp":
        assert h is None
        comparison = encoder_parameter_comparison(cfg)
        assert comparison["global_mlp_total_model_parameters"] == model.total_params
        assert enc.net[0].out_features == enc.net[1].in_features == 512
        assert enc.net[-1].dim == 8
        torch.testing.assert_close(
            z.unflatten(-1, (8, 8)).sum(-1), torch.ones_like(z[:, :8])
        )
    else:
        assert h.shape == (2, n, 64) and torch.isfinite(h).all()
        assert cfg.architecture_resolved["readout_dims"] == [128, 256, 64]
        hcount = sum(
            p.numel()
            for k, p in enc.named_parameters()
            if not k.startswith("global_readout")
        )
        assert hcount == cfg.architecture_resolved["h_encoder_parameters"]
        torch.testing.assert_close(
            z, enc.global_readout(torch.cat((h.mean(-2), h.amax(-2)), -1))
        )
    local, z = model.encode_actor_features(batch)
    torch.testing.assert_close(local, node_features(batch, cfg))
    seen = []
    handle = model._pi.net[0].register_forward_pre_hook(
        lambda m, args: seen.append(args[0].detach().clone())
    )
    eps = torch.randn(2, cfg.action_dim, device=device)
    action, info = model.pi(z, node_states=local, noise=eps)
    handle.remove()
    assert action.shape == info["log_std"].shape == (2, cfg.action_dim)
    torch.testing.assert_close(info["scaled_entropy"], info["entropy"] / cfg.action_dim)
    mean, raw = model._pi(z, node_states=local).chunk(2, -1)
    ref, refinfo = model._squashed_gaussian(mean, model.policy_log_std(raw), eps)
    torch.testing.assert_close(action, ref)
    torch.testing.assert_close(info["entropy"], refinfo["entropy"])
    if cfg.exp017_factors["A"] == "A0":
        assert seen[0].shape == (2, cfg.action_dim, f + 64)
        torch.testing.assert_close(
            seen[0][..., :f], local.index_select(-2, model._pi.driver_nodes)
        )
        torch.testing.assert_close(
            seen[0][..., f:], z[:, None, :].expand(2, cfg.action_dim, 64)
        )
    else:
        torch.testing.assert_close(seen[0], z)
        torch.testing.assert_close(
            model._pi(z, node_states=local), model._pi(z, node_states=local + 99)
        )
    for eval_mode in (False, True):
        torch.testing.assert_close(
            agent.act(batch, noise=eps, eval_mode=eval_mode, return_cpu=False), action
        )
    rows = [record(obs, action_dim=cfg.action_dim)]
    for step in range(5):
        # deterministic, normalized policy action; genuine environment physics/reward
        a = agent.act(obs, noise=torch.zeros(cfg.action_dim, device=device))
        obs, reward, done, envinfo = env.step(a)
        assert torch.isfinite(obs).all() and torch.isfinite(reward)
        if name == "sis":
            u = float(((a + 1) / 2).mean())
            assert float(envinfo["control_cost"]) == pytest.approx(0.1 * u, abs=1e-7)
            assert float(reward) == pytest.approx(
                float(envinfo["infection_reward"]) - 0.1 * u, abs=1e-7
            )
        rows.append(
            record(obs, a, reward, envinfo["terminated"], action_dim=cfg.action_dim)
        )
    buf = Buffer(cfg)
    buf.add(torch.cat(rows))
    calls = []
    if cfg.dynamics_type == "disabled":
        assert model._dynamics is model._reward is model._termination is None
        assert not any(
            k.startswith(("_dynamics.", "_reward.", "_termination."))
            for k in model.state_dict()
        )

        def forbidden(*a, **kw):
            raise AssertionError("imagined/model prediction in D0")

        monkeypatch.setattr(model, "next", forbidden)
        monkeypatch.setattr(model, "reward", forbidden)
        monkeypatch.setattr(model, "termination", forbidden)
    else:
        orig = model.next

        def capture(*a, **kw):
            calls.append(1)
            return orig(*a, **kw)

        monkeypatch.setattr(model, "next", capture)
    results = []
    # Full windows and tail windows, including actual optimizer steps.
    for starts in ([0, 1], [4, 4]):
        data = buf.sample(start_indices=starts)
        values = data["model_batch"]
        mask = data["model_valid_mask"]
        # Poison padding: valid observations and loss must remain finite.
        if not bool(mask.all()):
            o, a, r, t, task = values
            ov = torch.cat((torch.ones_like(mask[:1]), mask), 0)
            values = (
                torch.where(ov[..., None], o, float("nan")),
                *(torch.where(mask[..., None], x, float("nan")) for x in (a, r, t)),
                task,
            )
        before = [p.detach().clone() for p in model._encoder.parameters()]
        pibefore = [p.detach().clone() for p in model._pi.parameters()]
        metrics = agent._update(*values, model_valid_mask=mask)
        assert all(torch.isfinite(v) for v in metrics.values())
        # Initially zero Q output weights imply zero encoder gradients on D0's first step.
        if cfg.dynamics_type != "disabled" or starts == [4, 4]:
            assert any(
                not torch.equal(p, q)
                for p, q in zip(before, model._encoder.parameters())
            )
        assert any(
            not torch.equal(p, q) for p, q in zip(pibefore, model._pi.parameters())
        )
        if cfg.dynamics_type == "disabled":
            assert not set(metrics) & {
                "consistency_loss",
                "reward_loss",
                "termination_loss",
                "value_loss_rollout",
            }
            torch.testing.assert_close(
                metrics["total_loss"], cfg.value_coef * metrics["value_loss_encoder"]
            )
        results.append({k: float(v) for k, v in metrics.items()})
    if cfg.dynamics_type != "disabled":
        assert len(calls) == 6
        assert {"consistency_loss", "reward_loss", "value_loss_rollout"} <= results[
            0
        ].keys()
    path = tmp_path / "weights.pt"
    agent.save(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    assert checkpoint["metadata"]["exp017_factors"] == cfg.exp017_factors
    assert checkpoint["metadata"]["model_horizon"] == (
        3 if cfg.dynamics_type != "disabled" else None
    )
    expected = agent.act(batch, noise=eps, return_cpu=False)
    assert agent.load(checkpoint)["policy_restored"]
    torch.testing.assert_close(expected, agent.act(batch, noise=eps, return_cpu=False))
    # Reject each other factorial identity even when tensor shapes happen to match.
    for other in ALL_ARMS:
        if other == arm:
            continue
        bad = deepcopy(checkpoint)
        bad["metadata"]["exp017_arm"] = other
        with pytest.raises(ValueError, match="identity differs"):
            agent.load(bad)
    bad = deepcopy(checkpoint)
    bad["metadata"]["state_encoder_type"] = (
        "gatv2_prior" if cfg.state_encoder_type == "global_mlp" else "global_mlp"
    )
    with pytest.raises(ValueError, match="state_encoder_type identity differs"):
        agent.load(bad)
    bad = deepcopy(checkpoint)
    bad["config"]["exp017_factors"]["G"] = "invalid"
    with pytest.raises(ValueError):
        agent.load(bad)
    saved_config = tmp_path / "config.yaml"
    saved_config.write_text(yaml.safe_dump(vars(cfg)))
    restored = load_config(saved_config)
    make_env(restored).close()
    assert digest(restored.architecture_resolved) == digest(cfg.architecture_resolved)
    env.close()
    evidence = os.environ.get("NETCC_RL_EXP017_EVIDENCE")
    if evidence:
        dest = Path(evidence)
        dest.mkdir(parents=True, exist_ok=True)
        row = dict(
            environment=name,
            arm=arm,
            device=device,
            factors=cfg.exp017_factors,
            architecture=cfg.architecture_resolved,
            metadata=actor_metadata(cfg),
            h_encoder_parameters=hcount,
            total_model_parameters=model.total_params,
            H=list(h.shape) if h is not None else None,
            z=list(z.shape),
            action=list(action.shape),
            metrics=results,
            resolved_config_sha256=digest(vars(cfg)),
            checkpoint_roundtrip=True,
        )
        (dest / f'{name}-{arm}-{device.replace(":","_")}.json').write_text(
            json.dumps(row, indent=2) + "\n"
        )
        (dest / f'{name}-{arm}-{device.replace(":","_")}.yaml').write_text(
            yaml.safe_dump(vars(cfg))
        )


@pytest.mark.parametrize("name", MATRIX["environments"])
def test_g0_no_adjacency_or_cross_node_dependency(name, monkeypatch):
    from netcc_rl.algorithms.proposed.common.exp017 import MatchedNodeMLP
    from netcc_rl.algorithms.proposed.common import node_features as nf

    cfg = resolved(name, "E1")

    def forbidden(*args, **kwargs):
        raise AssertionError("G0 accessed topology")

    monkeypatch.setattr(nf, "prior_adjacency", forbidden)
    enc = MatchedNodeMLP(cfg)
    assert not list(enc.buffers())
    obs = torch.randn(2, cfg.obs_shape["state"][0])
    h, z = enc.forward_features(obs)
    # Change the physical adjacency and add misleading edge indices in config.
    other = deepcopy(cfg)
    snapshot = (
        other.graph_resolved
        or other.mpcsc_resolved["fixed_actuators"]["physical_system"]
    )
    snapshot["adjacency"] = np.zeros((cfg.architecture_resolved["N"],) * 2).tolist()
    other.edge_index = [[0], [0]]
    alt = MatchedNodeMLP(other)
    alt.load_state_dict(enc.state_dict())
    torch.testing.assert_close(h, alt.encode_nodes(obs))
    torch.testing.assert_close(z, alt(obs))
    # Permuting native nodes permutes H exactly; modifying node 0 cannot alter H[1:].
    n, raw = (
        cfg.architecture_resolved["N"],
        cfg.obs_shape["state"][0] // cfg.architecture_resolved["N"],
    )
    perm = torch.randperm(n)
    reordered = obs.reshape(2, n, raw)[:, perm].reshape_as(obs)
    torch.testing.assert_close(enc.encode_nodes(reordered), h[:, perm])
    changed = obs.clone()
    changed[:, :raw] += 1
    torch.testing.assert_close(enc.encode_nodes(changed)[:, 1:], h[:, 1:])
    # Every MLP parameter participates in the node computation; no dummy capacity.
    grads = torch.autograd.grad(
        (h * torch.randn_like(h)).sum(), tuple(enc.node_encoder.parameters())
    )
    assert all(torch.isfinite(g).all() and bool(g.abs().sum()) for g in grads)


@pytest.mark.parametrize("name", MATRIX["environments"])
@pytest.mark.parametrize("arm", ["E2", "E4", "E6", "E7", *MATRIX["global_mlp_arms"]])
def test_actual_next_observation_and_bellman_mask(name, arm, monkeypatch):
    agent = TDMPC2(resolved(name, arm))
    cfg = agent.cfg
    obs = torch.randn(cfg.horizon + 1, 2, cfg.obs_shape["state"][0])
    a = torch.zeros(cfg.horizon, 2, cfg.action_dim)
    reward = torch.arange(2, 2 + 2 * cfg.horizon, dtype=obs.dtype).reshape(
        cfg.horizon, 2, 1
    )
    terminated = torch.tensor([[[0.0], [1.0]]]).expand_as(reward)
    expected = agent.model.encode(obs[1:], None).detach()
    seen = []
    original = agent._td_target

    def capture(z, r, t, task, **kw):
        torch.testing.assert_close(z, expected)
        target = original(z, r, t, task, **kw)
        seen.append(target)
        return target

    monkeypatch.setattr(agent, "_td_target", capture)
    original_q = agent.model.Q

    def q(z, *args, **kwargs):
        if kwargs.get("target"):
            torch.testing.assert_close(z, expected)
            return z.new_full((*z.shape[:-1], 1), 10.0)
        return original_q(z, *args, **kwargs)

    monkeypatch.setattr(agent.model, "Q", q)
    agent._update(obs, a, reward, terminated)
    torch.testing.assert_close(seen[0], reward + (1 - terminated) * agent.discount * 10)


@pytest.mark.parametrize("name", MATRIX["environments"])
def test_e0_exact_main_modules_initialization_and_update(name):
    from netcc_rl.algorithms.baselines.environment_profile import resolved as main_config

    cfg = resolved(name, "E0")
    main = main_config(name, "cpu")
    for c in (main, cfg):
        c.batch_size = 2
        c.graph_checkpoint = False
    torch.manual_seed(174)
    baseline = TDMPC2(main)
    torch.manual_seed(174)
    experiment = TDMPC2(cfg)
    for key, value in baseline.model.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(
                value, experiment.model.state_dict()[key], rtol=0, atol=0
            )
    obs = torch.randn(4, 2, cfg.obs_shape["state"][0])
    action = torch.randn(3, 2, cfg.action_dim).tanh()
    reward = torch.randn(3, 2, 1)
    terminated = torch.zeros_like(reward)
    torch.manual_seed(174)
    a = baseline._update(obs, action, reward, terminated)
    torch.manual_seed(174)
    b = experiment._update(obs, action, reward, terminated)
    assert a.keys() == b.keys()
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)


@pytest.mark.parametrize("name", MATRIX["environments"])
@pytest.mark.parametrize("arm", ["E7", *MATRIX["global_mlp_arms"]])
def test_evaluation_cli_mlp_arms(name, arm, tmp_path):
    cfg = resolved(name, arm)
    cfg.eval_episodes = 1
    cfg.eval_policy = "gaussian_mean"
    agent = TDMPC2(cfg)
    agent.save(tmp_path / "weights.pt")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(vars(cfg)))
    output = tmp_path / "evaluation.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "netcc_rl.algorithms.proposed.evaluate",
            "--checkpoint",
            str(tmp_path / "weights.pt"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert np.isfinite(data["episode_return"]) and len(data["episodes"]) == 1
    assert data["checkpoint_load"]["policy_restored"]
    assert data["exp017_arm"] == arm
    if arm != "E7":
        return
    # Exercise the public training entrypoint for one interaction, with no learner campaign.
    work_dir = tmp_path / "entrypoint-smoke"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "netcc_rl.algorithms.proposed.train",
            "--config",
            str(tmp_path / "config.yaml"),
            "steps=1",
            "buffer_size=10",
            "warmup_steps=5",
            f"work_dir={work_dir}",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((work_dir / "run-manifest.json").read_text())
    assert manifest["exp017_arm"] == "E7" and manifest["dynamics_enabled"] is False
    assert (work_dir / "final.pt").is_file()


def test_matrix_full_factorial_identities_and_production_isolation():
    from itertools import product

    # The post-hoc script uses historical factor levels and arm IDs together.
    factor_names = ("G", "D", "A")
    historical_levels = [tuple(MATRIX["factors"][f]) for f in factor_names]
    historical_arms = MATRIX["arms"].values()
    assert {tuple(arm[f] for f in factor_names) for arm in historical_arms} == set(
        product(*historical_levels)
    )
    assert len(historical_arms) == 8
    encoder_levels = (*historical_levels[0], *MATRIX["global_mlp_encoders"])
    assert {tuple(arm[f] for f in factor_names) for arm in ALL_ARMS.values()} == set(
        product(encoder_levels, *historical_levels[1:])
    )
    assert len(list((ROOT.parent / "configs/ablation").glob("*/*.yaml"))) == 36
    identities = set()
    for name in MATRIX["environments"]:
        for arm in ALL_ARMS:
            cfg = resolved(name, arm)
            identities.add(digest(cfg.architecture_resolved))
            cfg.work_dir = str(ROOT.parent / "test-runs" / name / arm)
            validate_experiment_output(cfg)
            cfg.work_dir = "/tmp/artifacts/EXP-016/wrong"
            with pytest.raises(ValueError):
                validate_experiment_output(cfg)
    assert len(identities) == 36
    for override in [
        "state_encoder_type=matched_node_mlp",
        "state_encoder_type=global_mlp",
        "actor_architecture=global_latent",
        "dynamics_type=disabled",
    ]:
        with pytest.raises(ValueError):
            load_config(overrides=[override])
    cfg = resolved("sis", "E7")
    cfg.exp017_factors["D"] = "D1"
    with pytest.raises(ValueError):
        validate(cfg)


def test_frozen_base_hash_and_same_shape_environment_checkpoint_rejection(monkeypatch):
    from netcc_rl import matrix

    matrix._base.cache_clear()
    with monkeypatch.context() as m:
        m.setitem(matrix.MATRIX["base_config_sha256"], "sis", "invalid")
        with pytest.raises(ValueError, match="base config changed"):
            resolved("sis", "E0")
    matrix._base.cache_clear()
    sis = TDMPC2(resolved("sis", "E0"))
    kuramoto = TDMPC2(resolved("kuramoto", "E0"))
    # Same N/Z/action/Q dimensions must not bypass the environment identity.
    checkpoint = dict(
        metadata=actor_metadata(sis.cfg),
        config=deepcopy(vars(sis.cfg)),
        model=sis.model.state_dict(),
    )
    with pytest.raises(ValueError, match="identity differs"):
        kuramoto.load(checkpoint)


@pytest.mark.parametrize("name", MATRIX["environments"])
def test_global_mlp_native_input_and_topology_independence(name, monkeypatch):
    from torch import nn
    from netcc_rl.algorithms.proposed.common.exp017 import GlobalMLPEncoder
    from netcc_rl.algorithms.proposed.common.layers import NormedLinear, SimNorm
    from netcc_rl.algorithms.proposed.common import node_features as nf

    cfg = resolved(name, "E8")

    def forbidden(*args, **kwargs):
        raise AssertionError("Global MLP accessed topology")

    monkeypatch.setattr(nf, "prior_adjacency", forbidden)
    encoder = GlobalMLPEncoder(cfg)
    assert not list(encoder.buffers())
    assert [type(layer) for layer in encoder.net] == [NormedLinear, nn.Linear, SimNorm]
    assert type(encoder.net[0].act) is nn.Mish
    assert encoder.net[0].ln.elementwise_affine and encoder.net[0].dropout is None
    obs = torch.randn(2, 3, cfg.obs_shape["state"][0])
    native = node_features(obs, cfg).flatten(-2)
    if name == "kuramoto":
        torch.testing.assert_close(native[..., ::2], obs.sin())
        torch.testing.assert_close(native[..., 1::2], obs.cos())
    else:
        torch.testing.assert_close(native, obs)
    seen = []
    hook = encoder.net[0].register_forward_pre_hook(
        lambda module, args: seen.append(args[0])
    )
    z = encoder(obs)
    hook.remove()
    torch.testing.assert_close(seen[0], native)
    assert z.shape == (2, 3, 64)
    other = deepcopy(cfg)
    snapshot = (
        other.graph_resolved
        or other.mpcsc_resolved["fixed_actuators"]["physical_system"]
    )
    del snapshot["adjacency"]
    other.edge_index = [[0], [0]]
    alternative = GlobalMLPEncoder(other)
    alternative.load_state_dict(encoder.state_dict())
    torch.testing.assert_close(alternative(obs), z, atol=0, rtol=0)
    gradients = torch.autograd.grad(
        (z * torch.randn_like(z)).sum(), tuple(encoder.parameters())
    )
    assert all(torch.isfinite(g).all() and bool(g.abs().sum()) for g in gradients)


@pytest.mark.parametrize("name", MATRIX["environments"])
@pytest.mark.parametrize("arm", MATRIX["global_mlp_arms"])
def test_global_mlp_config_matches_control(name, arm):
    cfg = resolved(name, arm, "cuda:0")
    control = resolved(name, MATRIX["global_mlp_controls"][arm], "cuda:0")
    expected_differences = {
        "exp017_arm",
        "exp017_factors",
        "state_encoder_type",
        "architecture_resolved",
    }
    assert {
        key for key, value in vars(cfg).items() if value != vars(control)[key]
    } == expected_differences
    comparison = encoder_parameter_comparison(cfg)
    expected_input, expected_gatv2, expected_mlp = {
        "kuramoto": (200, 125952, 136768),
        "sis": (100, 125824, 85568),
        "swing": (190, 126336, 131648),
    }[name]
    assert cfg.architecture_resolved["global_mlp_dims"] == [expected_input, 512, 64]
    assert comparison["hidden_width"] == 512
    assert comparison["d_obs"] == expected_input
    assert comparison["gatv2_encoder_parameters"] == expected_gatv2
    assert comparison["global_mlp_encoder_parameters"] == expected_mlp
    assert comparison["relative_difference"] == pytest.approx(
        (expected_mlp - expected_gatv2) / expected_gatv2
    )
