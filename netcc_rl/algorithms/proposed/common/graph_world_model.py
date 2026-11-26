"""GAT node embeddings, global Mean+Max readout and native-state shared Gaussian policy."""

import torch
from torch import nn

from .gatv2_encoder import GATv2StateEncoder
from .layers import SimNorm, mlp
from .node_features import fixed_driver_nodes, node_layout


class NodeGATv2Encoder(GATv2StateEncoder):
    """Reuse EXP-011 candidate graph/batching; retain projected normalized nodes."""

    def __init__(self, cfg):
        super().__init__(cfg)
        del self.output_projection
        self.node_projection = nn.Sequential(
            nn.Linear(cfg.enc_dim, cfg.graph_node_dim), nn.LayerNorm(cfg.graph_node_dim)
        )
        self.global_readout = mlp(
            2 * cfg.graph_node_dim, [cfg.mlp_dim], cfg.latent_dim, act=SimNorm(cfg)
        )

    def encode_nodes(self, obs):
        return self.node_projection(super().encode_nodes(obs))

    def encode_global(self, nodes):
        pooled = torch.cat((nodes.mean(-2), nodes.amax(-2)), dim=-1)
        return self.global_readout(pooled)

    def forward_features(self, obs):
        nodes = self.encode_nodes(obs)
        return nodes, self.encode_global(nodes)

    def forward(self, obs):
        return self.encode_global(self.encode_nodes(obs))


class SharedNativeGaussianActor(nn.Module):
    """One shared head on native node state plus G2 global latent."""

    def __init__(self, cfg):
        super().__init__()
        self.nodes = node_layout(cfg)[0]
        self.width = node_layout(cfg)[2]
        self.register_buffer(
            "driver_nodes", torch.tensor(fixed_driver_nodes(cfg)), persistent=False
        )
        self.net = mlp(self.width + cfg.latent_dim, 2 * [cfg.mlp_dim], 2)

    def forward(self, z, *, node_states=None):
        if node_states is None or node_states.shape != (
            *z.shape[:-1],
            self.nodes,
            self.width,
        ):
            raise ValueError(
                "native-state actor requires raw node states aligned with z"
            )
        local = node_states.index_select(-2, self.driver_nodes)
        global_z = z.unsqueeze(-2).expand(*local.shape[:-1], z.shape[-1])
        mean, log_std = self.net(torch.cat((local, global_z), -1)).unbind(-1)
        return torch.cat((mean, log_std), -1)
