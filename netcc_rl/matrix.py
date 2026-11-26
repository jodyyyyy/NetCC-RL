"""Resolve EXP-017 configurations and enumerate commands without starting training."""

import argparse
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import json
import shlex
import yaml
from netcc_rl.algorithms.proposed.common.adaptive_architecture import digest

ROOT = Path(__file__).resolve().parent
MATRIX = yaml.safe_load((ROOT / "ablation.yaml").read_text())
ALL_ARMS = {**MATRIX["arms"], **MATRIX["global_mlp_arms"]}
ENCODERS = {**MATRIX["factors"]["G"], **MATRIX["global_mlp_encoders"]}


@lru_cache(None)
def _base(name):
    value = yaml.safe_load(
        (ROOT / "algorithms/proposed/configs/exp016" / f"{name}.yaml").read_text()
    )
    if digest(value) != MATRIX["base_config_sha256"][name]:
        raise ValueError(
            "EXP-017 base config changed from frozen main; explicit migration required"
        )
    return value


def expand(spec):
    if set(spec) != {"exp017_environment", "exp017_arm"}:
        raise ValueError("EXP-017 config must select exactly environment and arm")
    name, arm = spec["exp017_environment"], spec["exp017_arm"]
    if name not in MATRIX["environments"] or arm not in ALL_ARMS:
        raise ValueError("unknown EXP-017 environment/arm")
    factors = ALL_ARMS[arm]
    base = deepcopy(_base(name))
    base.update(
        exp017_environment=name,
        exp017_arm=arm,
        exp017_factors=deepcopy(factors),
        exp017_base_sha256=digest(base),
        graph_backend="pyg",
        graph_checkpoint=False,
        learner_cuda_graphs=True,
        compile=True,
        collector_backend="cpu_serial",
        collector_batch_size=1,
        collector_protocol=None,
    )
    base.update(
        state_encoder_type=ENCODERS[factors["G"]],
        dynamics_type=MATRIX["factors"]["D"][factors["D"]],
        actor_architecture=MATRIX["factors"]["A"][factors["A"]],
    )
    if factors["D"] == "D0":
        base.update(
            horizon=1,
            critic_latent_source="encoder",
            critic_encoder_aux_coef=0.0,
            consistency_coef=0.0,
            reward_coef=0.0,
            termination_coef=0.0,
        )
    return base


def expected(cfg):
    return expand(
        dict(exp017_environment=cfg.exp017_environment, exp017_arm=cfg.exp017_arm)
    )


def validate(cfg):
    base = expected(cfg)
    # Preserve matched losses/physics/architecture. Runtime controls can be overridden.
    fixed = (
        "exp017_factors",
        "exp017_base_sha256",
        "actor_architecture",
        "state_encoder_type",
        "dynamics_type",
        "horizon",
        "critic_latent_source",
        "critic_encoder_aux_coef",
        "consistency_coef",
        "reward_coef",
        "termination_coef",
        "value_coef",
        "entropy_coef",
        "lr",
        "enc_lr_scale",
        "rho",
        "tau",
        "num_q",
        "num_bins",
        "vmin",
        "vmax",
        "dropout",
        "log_std_min",
        "log_std_max",
        "mpcsc_resolved",
        "graph_resolved",
        "episodic",
    )
    if cfg.exp017_arm in MATRIX["global_mlp_arms"]:
        fixed += ("simnorm_dim",)
    for k in fixed:
        if getattr(cfg, k, None) != base.get(k):
            raise ValueError(f"EXP-017 {k} differs from frozen matrix")
    if cfg.graph_backend != "pyg":
        raise ValueError("EXP-017 preserves the PyG encoder backend")
    if cfg.collector_backend != "cpu_serial":
        raise ValueError("EXP-017 uses the normal serial real-transition runner")
    # Canonical environment task can be normalized by make_env; snapshot is fixed above.
    if cfg.task != base["task"]:
        raise ValueError("EXP-017 task differs from frozen matrix")


def resolved(name, arm, device="cpu"):
    from netcc_rl.algorithms.proposed.config import load_config
    from netcc_rl.environments import make_env

    cfg = load_config(
        overrides=[f"device={device}"]
        + (["compile=false", "learner_cuda_graphs=false"] if device == "cpu" else []),
        ablation=(name, arm),
    )
    make_env(cfg).close()
    return cfg


def encoder_parameter_comparison(cfg):
    """Measure both complete encoders and total model counts on CPU."""
    from netcc_rl.algorithms.proposed.common.adaptive_world_model import AdaptiveWorldModel

    name, arm = cfg.exp017_environment, cfg.exp017_arm
    control_arm = MATRIX["global_mlp_controls"][arm]
    control_cfg = resolved(name, control_arm)
    control = AdaptiveWorldModel(control_cfg)
    model = AdaptiveWorldModel(cfg)
    target, actual = (
        sum(p.numel() for p in m._encoder.parameters() if p.requires_grad)
        for m in (control, model)
    )
    architecture = cfg.architecture_resolved
    if (target, actual) != (
        architecture["complete_gatv2_parameters"],
        architecture["encoder_parameters"],
    ):
        raise ValueError(
            f"{name}/{arm}: instantiated encoder counts differ from the resolved architecture"
        )
    difference = actual - target
    if control.total_params - target != model.total_params - actual:
        raise ValueError(
            f"{name}/{arm}: non-encoder parameter count differs from control"
        )
    input_dim, width, _ = architecture["global_mlp_dims"]
    return dict(
        environment=name,
        arm=arm,
        control_arm=control_arm,
        d_obs=input_dim,
        hidden_width=width,
        gatv2_encoder_parameters=target,
        global_mlp_encoder_parameters=actual,
        parameter_difference=difference,
        relative_difference=difference / target,
        absolute_error=abs(difference),
        relative_error=abs(difference) / target,
        gatv2_total_model_parameters=control.total_params,
        global_mlp_total_model_parameters=model.total_params,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--set", choices=("historical", "global-mlp", "all"), default="historical"
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help="only print training commands, with per-arm work_dir under this root",
    )
    args = parser.parse_args()
    arms = {
        "historical": MATRIX["arms"],
        "global-mlp": MATRIX["global_mlp_arms"],
        "all": ALL_ARMS,
    }[args.set]
    rows = []
    comparisons = []
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    for name in MATRIX["environments"]:
        for arm in arms:
            cfg = resolved(name, arm, args.device)
            row = dict(
                environment=name,
                arm=arm,
                factors=cfg.exp017_factors,
                identity_sha256=digest(cfg.architecture_resolved),
                config_sha256=digest(vars(cfg)),
            )
            if arm in MATRIX["global_mlp_arms"]:
                comparisons.append(encoder_parameter_comparison(cfg))
                row["configuration_set"] = "global-mlp"
            elif args.set == "all":
                row["configuration_set"] = "historical"
            if args.run_root:
                row["training_command"] = shlex.join(
                    [
                        "python",
                        "-m",
                        "netcc_rl.algorithms.proposed.train",
                        "--config",
                        str(
                            Path("configs/ablation")
                            / name
                            / f"{arm}.yaml"
                        ),
                        f"device={args.device}",
                        *(
                            ["compile=false", "learner_cuda_graphs=false"]
                            if args.device == "cpu"
                            else []
                        ),
                        f"work_dir={args.run_root / name / arm}",
                    ]
                )
            rows.append(row)
            if args.output:
                (args.output / f"{name}-{arm}.yaml").write_text(
                    yaml.safe_dump(vars(cfg), sort_keys=True)
                )
    if args.output:
        (args.output / "index.json").write_text(json.dumps(rows, indent=2) + "\n")
        if comparisons:
            (args.output / "parameter-counts.json").write_text(
                json.dumps(comparisons, indent=2) + "\n"
            )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
