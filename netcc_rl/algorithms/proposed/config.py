"""Explicit Gaussian-only mainline configuration and checkpoint contracts."""

import argparse
import math
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from types import SimpleNamespace
import yaml
from .common.adaptive_architecture import PROFILE as ADAPTIVE_PROFILE

GAUSSIAN_POLICY_CONTRACT = "tanh_affine_joint_density_arctan_std_mean_action_entropy"
MAINLINE_CONTRACT = "gaussian_single_step_native_states_g2_v1"


def resolve_gaussian_bounds(cfg):
    get = cfg.get if isinstance(cfg, Mapping) else lambda k: getattr(cfg, k, None)
    for key, value in [("gaussian_action_low", -1.0), ("gaussian_action_high", 1.0)]:
        supplied = get(key)
        if supplied is not None and (isinstance(supplied, bool) or supplied != value):
            raise ValueError(
                f"{key} must be {value} for the effective Gaussian task; train from scratch"
            )
        if isinstance(cfg, MutableMapping):
            cfg[key] = value
        else:
            setattr(cfg, key, value)


def resolve_model_widths(cfg, observation_shape):
    from .common.adaptive_architecture import resolve

    if isinstance(cfg, Mapping):
        temporary = SimpleNamespace(**cfg)
        resolve(temporary)
        cfg.update(vars(temporary))
    else:
        resolve(cfg)
    return cfg


def validate(cfg):
    from netcc_rl.environments import supported_task

    if not supported_task(cfg.task):
        raise ValueError("main registry supports only Kuramoto, SIS and Swing")
    resolve_gaussian_bounds(cfg)
    required = dict(
        actor_type="gaussian",
        actor_architecture="shared_native_state",
        model_width_profile=ADAPTIVE_PROFILE,
        state_encoder_type="gatv2_prior",
        graph_num_layers=2,
        graph_num_heads=4,
        graph_global_readout="mean_max",
        exp016_arm="G2",
        dynamics_type="global_mlp",
        horizon=3,
        num_q=5,
        actor_latent_source="encoder",
        critic_latent_source="rollout",
        obs="state",
        task_dim=0,
        multitask=False,
        mpc=False,
    )
    if getattr(cfg, "exp017_arm", None) is not None:
        from netcc_rl.matrix import validate as validate_matrix

        validate_matrix(cfg)
        for key in (
            "actor_architecture",
            "state_encoder_type",
            "dynamics_type",
            "horizon",
            "critic_latent_source",
        ):
            required[key] = getattr(cfg, key)
    for key, value in required.items():
        if getattr(cfg, key, None) != value:
            raise ValueError(f"mainline requires {key}={value}")
    for key in (
        "episodic",
        "compile",
        "retain_best_checkpoint",
        "learner_cuda_graphs",
        "graph_checkpoint",
    ):
        if type(getattr(cfg, key)) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in (
        "batch_size",
        "steps",
        "buffer_size",
        "num_bins",
        "latent_dim",
        "simnorm_dim",
        "mlp_dim",
        "enc_dim",
        "num_enc_layers",
        "torch_threads",
        "eval_episodes",
        "eval_freq",
        "log_freq",
        "save_freq",
        "profile_interval",
        "profile_window",
    ):
        if type(getattr(cfg, key)) is not int or getattr(cfg, key) <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("warmup_steps", "initial_updates", "seed", "eval_seed"):
        if type(getattr(cfg, key)) is not int or getattr(cfg, key) < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("lr", "enc_lr_scale", "grad_clip_norm"):
        if not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in (
        "reward_coef",
        "value_coef",
        "termination_coef",
        "consistency_coef",
        "entropy_coef",
        "critic_encoder_aux_coef",
    ):
        if not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) < 0:
            raise ValueError(f"{key} must be nonnegative and finite")
    if not 0 < cfg.rho <= 1 or not 0 < cfg.tau <= 1 or not cfg.vmin < cfg.vmax:
        raise ValueError("invalid rho/tau/value range")
    if (
        not 0 <= cfg.dropout < 1
        or not math.isfinite(cfg.log_std_min)
        or not math.isfinite(cfg.log_std_max)
        or not cfg.log_std_min < cfg.log_std_max
    ):
        raise ValueError("invalid dropout/Gaussian standard deviation")
    if not 0 <= cfg.discount_min <= cfg.discount_max < 1:
        raise ValueError("invalid discounts")
    if cfg.num_bins < 2 or cfg.latent_dim % cfg.simnorm_dim:
        raise ValueError("invalid value bins/SimNorm grouping")
    if cfg.eval_policy not in {"stochastic", "gaussian_mean"}:
        raise ValueError("invalid eval_policy")
    if cfg.compile and cfg.learner_cuda_graphs and not getattr(cfg, "exp017_arm", None):
        raise ValueError("compile and CUDA Graph learner are mutually exclusive")
    if hasattr(cfg, "obs_shape"):
        from .common.adaptive_architecture import validate_identity

        validate_identity(cfg)
    from .common.utd import validate_schedule
    from .common.vector_runner import validate_collector

    validate_schedule(cfg)
    validate_collector(cfg)
    if cfg.graph_backend not in {"pyg", "turbo"}:
        raise ValueError("graph_backend must be pyg or turbo")
    if cfg.graph_backend == "turbo":
        if cfg.state_encoder_type not in {"gatv2_full", "gatv2_prior"}:
            raise ValueError("Turbo backend requires a GATv2 encoder")
        if cfg.graph_attention_dropout != 0:
            raise ValueError("Turbo GATv2 does not support attention dropout")
        if cfg.enc_dim // cfg.graph_num_heads not in {32, 64, 128, 256}:
            raise ValueError("Turbo GATv2 head width must be 32, 64, 128, or 256")
    for name in (
        "turbo_symmetric_csr",
        "turbo_fused_projection",
        "turbo_save_output_for_backward",
    ):
        if type(getattr(cfg, name, False)) is not bool:
            raise ValueError(f"{name} must be boolean")
    for name, allowed, default in (
        ("grad_A_reduce_row_chunk_size", {256, 512, 1024, 2048}, 512),
        ("forward_light_warps", {1, 2, 4}, 1),
        ("backward_light_warps", {1, 2, 4}, 1),
        ("forward_heavy_warps", {8, 16, 32}, 8),
        ("backward_heavy_warps", {8, 16, 32}, 8),
    ):
        value = getattr(cfg, "turbo_" + name, default)
        if type(value) is not int or value not in allowed:
            raise ValueError(f"invalid turbo_{name}")
    for name in ("turbo_forward_quantile", "turbo_backward_quantile"):
        value = getattr(cfg, name, 0.99)
        if isinstance(value, bool) or value not in {-1, 0.90, 0.95, 0.99}:
            raise ValueError(f"invalid {name}")
    if type(cfg.graph_batch_size) is not int or cfg.graph_batch_size < 1:
        raise ValueError("graph_batch_size must be a positive integer")
    if type(cfg.graph_checkpoint) is not bool:
        raise ValueError("graph_checkpoint must be boolean")
    if not 0 <= cfg.graph_attention_dropout < 1:
        raise ValueError("graph_attention_dropout must be in [0,1)")
    if not math.isfinite(cfg.graph_prior_bias_init):
        raise ValueError("graph_prior_bias_init must be finite")
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    return cfg


def load_config(path=None, overrides=(), *, profile=None, ablation=None):
    data = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
    if path or ablation:
        supplied = (yaml.safe_load(Path(path).read_text()) if path else
                    dict(zip(("exp017_environment", "exp017_arm"), ablation)))
        metadata = {
            "bin_size",
            "obs_shape",
            "action_dim",
            "episode_length",
            "seed_steps",
        }
        if isinstance(supplied, dict) and "exp017_environment" in supplied:
            from netcc_rl.matrix import expand

            if "exp017_factors" not in supplied:
                supplied = expand(supplied)
            metadata |= {
                "exp017_environment",
                "exp017_arm",
                "exp017_factors",
                "exp017_base_sha256",
            }
        if not isinstance(supplied, dict) or set(supplied) - set(data) - metadata:
            raise ValueError(
                "configuration must use current mainline settings; historical configs require the preserved research branch"
            )
        data.update(supplied)
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep or key not in data:
            raise ValueError(f"unknown mainline override {override!r}")
        data[key] = yaml.safe_load(value)
    if profile:
        data.update(profile)
    return validate(SimpleNamespace(**data))


def episode_discount(cfg, episode_length):
    if (
        isinstance(episode_length, bool)
        or not isinstance(episode_length, (int, float))
        or not math.isfinite(episode_length)
        or episode_length <= 0
    ):
        raise ValueError("episode_length must be positive and finite")
    return min(max(1 - 1 / episode_length, cfg.discount_min), cfg.discount_max)


def actor_metadata(cfg):
    keys = (
        "actor_type",
        "actor_architecture",
        "state_encoder_type",
        "model_width_profile",
        "graph_num_layers",
        "graph_num_heads",
        "graph_global_readout",
        "exp016_arm",
        "graph_backend",
        "graph_node_dim",
        "latent_dim",
        "enc_dim",
        "mlp_dim",
        "num_q",
        "num_bins",
        "gaussian_action_low",
        "gaussian_action_high",
        "architecture_resolved",
        "collector_backend",
        "collector_batch_size",
        "collector_protocol",
        "learner_cuda_graphs",
    )
    result = {k: getattr(cfg, k) for k in keys}
    result.update(
        mainline_contract=MAINLINE_CONTRACT,
        gaussian_policy_contract=GAUSSIAN_POLICY_CONTRACT,
        action_horizon=1,
        model_horizon=cfg.horizon,
        gaussian_entropy_objective="joint_entropy_divided_by_action_dim",
        replay_boundary_version="single_step_real_transition_tail_v1",
        model_width_rule="Z=2^ceil(log2(D/3)); W=4Z; GAT=2Z; node=Z",
        training_seed=cfg.seed,
        replay_capacity_unit="real_transitions",
    )
    if hasattr(cfg, "episode_length"):
        result.update(
            discount_episode_length=cfg.episode_length,
            effective_gamma=episode_discount(cfg, cfg.episode_length),
        )
    if getattr(cfg, "exp017_arm", None):
        from .common.adaptive_architecture import digest

        result.update(
            exp017_arm=cfg.exp017_arm,
            exp017_environment=cfg.exp017_environment,
            exp017_factors=cfg.exp017_factors,
            exp017_identity_sha256=digest(cfg.architecture_resolved),
            dynamics_enabled=cfg.dynamics_type != "disabled",
            model_horizon=cfg.horizon if cfg.dynamics_type != "disabled" else None,
            replay_horizon=cfg.horizon,
            mainline_contract="exp017_factorial_gaussian_v1",
        )
    return result


def experiment_name(cfg):
    if getattr(cfg, "exp017_arm", None):
        return f"exp017_{cfg.exp017_environment}_{cfg.exp017_arm}"
    return f"gaussian_K1_{cfg.model_width_profile}_G2_z{cfg.latent_dim}_h{cfg.horizon}"


def parse_args(description, *, evaluation=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path)
    if evaluation:
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--output", type=Path)
    parser.add_argument("overrides", nargs="*", metavar="key=value")
    return parser.parse_args()


def validate_experiment_output(cfg):
    if not cfg.work_dir:
        raise ValueError("an explicit output directory is required")
