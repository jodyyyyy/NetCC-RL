"""Versioned EXP-016 architecture; raw observation widths and frozen graph identity."""

import hashlib
import json
import math

PROFILE = "exp016_adaptive_v1"
READOUTS = {"G2": "mean_max"}


def adaptive_dimensions(d):
    """Exact 2**ceil(log2(D/3)), 4Z, including the fractional D=1 result."""
    if type(d) is not int or d <= 0:
        raise ValueError("D must be a positive integer")
    z = 0.5 if d == 1 else 1 << (((d + 2) // 3) - 1).bit_length()
    return z, 4 * z


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def architecture(cfg):
    from .node_features import node_layout, fixed_driver_nodes, prior_adjacency

    n, _, f, task = node_layout(cfg)
    if task not in {"mpcsc-sis", "mpcsc-kuramoto", "graph-swing"}:
        raise ValueError("unsupported adaptive graph task")
    d = math.prod(cfg.obs_shape["state"])
    z, w = adaptive_dimensions(d)
    if type(z) is not int or z % 2 or z % cfg.simnorm_dim:
        raise ValueError(
            f"adaptive Z={z} requires integer widths, four-head divisibility and SimNorm grouping"
        )
    mode = READOUTS.get(cfg.exp016_arm)
    if mode != cfg.graph_global_readout:
        raise ValueError("adaptive profile requires matching G2 Mean+Max readout")
    adjacency = prior_adjacency(cfg).tolist()
    drivers = fixed_driver_nodes(cfg)
    snapshot = getattr(cfg, "graph_resolved", None) or cfg.mpcsc_resolved
    result = dict(
        version=PROFILE,
        D=d,
        F=f,
        N=n,
        A=cfg.action_dim,
        Z=z,
        W=w,
        task=task,
        node_order=list(range(n)),
        driver_nodes=drivers,
        topology_sha256=digest(adjacency),
        environment_sha256=digest(snapshot),
        topology_rule="binary physical connectivity + self loops in GAT",
        gat_input=[f, 2 * z],
        gat_layers=2,
        gat_heads=4,
        gat_per_head=z // 2,
        node_projection=[2 * z, z],
        readout=mode,
        readout_dims=[2 * z, w, z],
        actor_local=(
            "native sin/cos + G2 global latent"
            if task == "mpcsc-kuramoto"
            else "native state + G2 global latent"
        ),
        actor_dims=[f + z, w, w, 2],
        dynamics_dims=[z + cfg.action_dim, w, w, z],
        reward_dims=[z + cfg.action_dim, w, w, cfg.num_bins],
        critic_dims=[z + cfg.action_dim, w, w, cfg.num_bins],
        num_q=cfg.num_q,
        termination_dims=[z, w, w, 1] if cfg.episodic else None,
        episodic=cfg.episodic,
        simnorm_dim=cfg.simnorm_dim,
    )

    if getattr(cfg, "exp017_arm", None):
        result.update(
            exp017_arm=cfg.exp017_arm,
            exp017_environment=cfg.exp017_environment,
            exp017_factors=cfg.exp017_factors,
            exp017_base_sha256=cfg.exp017_base_sha256,
            encoder=cfg.state_encoder_type,
            dynamics_enabled=cfg.dynamics_type != "disabled",
            replay_horizon=cfg.horizon,
        )
        if cfg.state_encoder_type == "global_mlp":
            from netcc_rl.matrix import MATRIX

            target = MATRIX["complete_gatv2_parameters"][cfg.exp017_environment]
            input_dim = n * f
            hidden_dim = MATRIX["global_mlp_hidden_width"]
            count = (input_dim + z + 3) * hidden_dim + z
            for key in (
                "gat_input",
                "gat_layers",
                "gat_heads",
                "gat_per_head",
                "node_projection",
                "readout",
                "readout_dims",
            ):
                del result[key]
            result.update(
                global_mlp_dims=[input_dim, hidden_dim, z],
                encoder_parameters=count,
                complete_gatv2_parameters=target,
                actor_local="native node state + global MLP latent",
            )
        else:
            from .exp017 import matched_widths

            widths, count = matched_widths(f, 2 * z, z, cfg.graph_num_layers)
            result.update(
                h_encoder_parameters=count,
                node_mlp_dims=(
                    [f, *widths, z]
                    if cfg.state_encoder_type == "matched_node_mlp"
                    else None
                ),
            )
        if cfg.actor_architecture == "global_latent":
            result.update(
                actor_local="global latent only",
                actor_dims=[z, w, w, 2 * cfg.action_dim],
            )
        if cfg.dynamics_type == "disabled":
            result.update(dynamics_dims=None, reward_dims=None, termination_dims=None)
    return result


def resolve(cfg):
    """Recompute from environment; never silently overwrite a saved identity."""
    resolved = architecture(cfg)
    saved = getattr(cfg, "architecture_resolved", None)
    if saved is not None and saved != resolved:
        raise ValueError(
            "adaptive architecture/environment identity differs from saved resolution"
        )
    cfg.architecture_resolved = resolved
    cfg.latent_dim = cfg.graph_node_dim = resolved["Z"]
    cfg.mlp_dim = resolved["W"]
    cfg.enc_dim = 2 * resolved["Z"]
    return resolved


def validate_identity(cfg):
    expected = architecture(cfg)
    if getattr(cfg, "architecture_resolved", None) != expected:
        raise ValueError(
            "adaptive architecture/environment identity missing or inconsistent"
        )
    dims = (cfg.latent_dim, cfg.graph_node_dim, cfg.mlp_dim, cfg.enc_dim)
    if dims != (expected["Z"], expected["Z"], expected["W"], 2 * expected["Z"]):
        raise ValueError("adaptive resolved dimensions differ from canonical widths")
    return expected
