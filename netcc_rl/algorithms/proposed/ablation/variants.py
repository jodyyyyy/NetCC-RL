"""Encoder, actor and learner variants for the ablation factor design."""

from functools import lru_cache
import torch
from torch import nn
from netcc_rl.algorithms.proposed.models.layers import NormedLinear, SimNorm, mlp
from netcc_rl.algorithms.proposed.models.node_features import node_features, node_layout


@lru_cache(None)
def matched_widths(features, graph_width, node_width, layers=2):
    """Search three hidden widths in [W/2,3W], minimizing distance to GAT W.

    Each GAT block has two affine W->W maps, W attention coefficients,
    output bias W and LayerNorm 2W. The shared readout is excluded.
    All MLP affines have biases and all four outputs have LayerNorm.
    """
    w, z = graph_width, node_width
    target = (features + 1) * w + layers * (2 * w * w + 6 * w) + (w + 3) * z
    solutions = []
    for a in range(w // 2, 3 * w + 1):
        for b in range(w // 2, 3 * w + 1):
            remaining = target - (features + 3) * a - (a + 3) * b - 3 * z
            c, remainder = divmod(remaining, b + 3 + z)
            if remainder == 0 and w // 2 <= c <= 3 * w:
                solutions.append(
                    (
                        max(abs(x - w) for x in (a, b, c)),
                        sum(abs(x - w) for x in (a, b, c)),
                        a,
                        b,
                        c,
                    )
                )
    if not solutions:
        raise ValueError(
            "BLOCKED: no exact parameter-matched ordinary node MLP in resolver search"
        )
    return tuple(min(solutions)[2:]), target


class MatchedNodeMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        f = node_layout(cfg)[2]
        widths, target = matched_widths(
            f, cfg.enc_dim, cfg.graph_node_dim, cfg.graph_num_layers
        )
        dims = (f, *widths, cfg.graph_node_dim)
        blocks = []
        for i, (before, after) in enumerate(zip(dims, dims[1:])):
            blocks.extend([nn.Linear(before, after), nn.LayerNorm(after)])
            if i < len(dims) - 2:
                blocks.append(nn.Mish())
        self.node_encoder = nn.Sequential(*blocks)
        assert sum(p.numel() for p in self.node_encoder.parameters()) == target
        # The same builder, shape, normalization and forward path as the reference graph encoder.
        self.global_readout = mlp(
            2 * cfg.graph_node_dim, [cfg.mlp_dim], cfg.latent_dim, act=SimNorm(cfg)
        )

    def encode_nodes(self, obs):
        return self.node_encoder(node_features(obs, self.cfg))

    def encode_global(self, nodes):
        return self.global_readout(torch.cat((nodes.mean(-2), nodes.amax(-2)), -1))

    def forward_features(self, obs):
        h = self.encode_nodes(obs)
        return h, self.encode_global(h)

    def forward(self, obs):
        return self.encode_global(self.encode_nodes(obs))


class GlobalMLPEncoder(nn.Module):
    """Map flattened native features in physical node order directly to z."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        input_dim, hidden_dim, latent_dim = cfg.architecture_resolved["global_mlp_dims"]
        self.net = nn.Sequential(
            NormedLinear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            SimNorm(cfg),
        )

    def forward(self, obs):
        return self.net(node_features(obs, self.cfg).flatten(-2))

    def forward_features(self, obs):
        return None, self(obs)


class GlobalGaussianActor(nn.Module):
    """Joint Gaussian logits from z alone, in the environment's driver order."""

    def __init__(self, cfg):
        super().__init__()
        self.width = node_layout(cfg)[2]  # caller's native-state tensor contract
        self.net = mlp(cfg.latent_dim, 2 * [cfg.mlp_dim], 2 * cfg.action_dim)

    def forward(self, z, *, node_states=None):
        return self.net(z)
