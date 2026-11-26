"""PyG GATv2 with source-to-target COO edges and mean graph pooling."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from netcc_rl.algorithms.proposed.models.layers import SimNorm, mlp
from netcc_rl.algorithms.proposed.models.node_features import node_features, node_layout, prior_adjacency


def candidate_edges(cfg):
    """Build a static candidate graph, with exactly one self loop per node.

    P1 consumes the frozen symmetric physical adjacency as connectivity only;
    physical weights are not edge features. No dense attention mask is used.
    """
    n, _, _, _ = node_layout(cfg)
    if cfg.state_encoder_type == 'gatv2_full':
        allowed = torch.ones(n, n, dtype=torch.bool)
    elif cfg.state_encoder_type == 'gatv2_prior':
        allowed = prior_adjacency(cfg).bool() | torch.eye(n, dtype=torch.bool)
    else:
        raise ValueError('GATv2 requires gatv2_full or gatv2_prior')
    return allowed.nonzero().T.contiguous()


class GATv2Block(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        self.conv = GATv2Conv(width, width // heads, heads=heads,
                             dropout=dropout, add_self_loops=False,
                             share_weights=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, x, edges):
        return self.norm(x + self.conv(x, edges))


class GATv2StateEncoder(nn.Module):
    """Two configurable residual GATv2 layers, node embeddings, global SimNorm.

    Graph microbatches bound edge tensor memory. Activation checkpointing is
    exact for dropout=0; PyTorch preserves RNG state when dropout is enabled.
    Frozen profiles prepare fixed COO tensors before execution. Device or dtype moves require preparation again.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        from netcc_rl.environments.graph.frozen import for_config
        self._frozen=for_config(cfg)
        self._static_ready=False
        self._static_edges={}
        self.static_preparations=0
        self.static_materializations=0
        self.node_count, _, features, _ = node_layout(cfg)
        self.input_projection = nn.Linear(features, cfg.enc_dim)
        self.blocks = nn.ModuleList(GATv2Block(cfg.enc_dim, cfg.graph_num_heads,
                                   cfg.graph_attention_dropout)
                                   for _ in range(cfg.graph_num_layers))
        self.output_projection = mlp(cfg.enc_dim,
            max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim], cfg.latent_dim,
            act=SimNorm(cfg))
        # Rebuilt from the checkpoint's frozen environment, without serializing topology buffers.
        if self._frozen is not None and cfg.state_encoder_type!=self._frozen['encoder']:
            raise ValueError('frozen encoder topology differs')
        edges=(torch.tensor(self._frozen['candidate_edges'],dtype=torch.long).T.contiguous()
               if self._frozen is not None else candidate_edges(cfg))
        self.register_buffer('edge_index', edges, persistent=False)

    def _apply(self, fn, recurse=True):
        self._static_edges.clear()
        self._static_ready=False
        return super()._apply(fn, recurse=recurse)

    def prepare_static_graphs(self):
        """Once before execution/capture: materialize frozen indices, no sorting."""
        if self._frozen is None or self._static_ready:return
        if self.edge_index.device.type=='cuda' and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('static graph preparation cannot run inside capture')
        if self.edge_index.dtype!=torch.long or self.input_projection.weight.dtype!=torch.float32:
            raise ValueError('frozen graph supports int64 COO / FP32 model only')
        device=self.edge_index.device;n=self.node_count
        for shape in self._frozen["batch_shapes"]:
            b = int(shape)
            offsets = torch.arange(b, device=device, dtype=torch.int32) * n
            self._static_edges[b] = (self.edge_index[:, None, :] + offsets.to(torch.long)[None, :, None]).reshape(2, -1)
            self.static_materializations+=1
        self.static_preparations+=1
        self._static_ready=True

    def batched_edges(self, graphs):
        if self._frozen is not None:
            if not self._static_ready or graphs not in self._static_edges:
                raise RuntimeError('unknown or unprepared frozen COO shape')
            return self._static_edges[graphs]
        offsets = torch.arange(graphs, device=self.edge_index.device) * self.node_count
        return (self.edge_index[:, None, :] + offsets[None, :, None]).reshape(2, -1)

    def _nodes(self, nodes):
        batch = nodes.shape[0]
        x = self.input_projection(nodes).reshape(-1, self.cfg.enc_dim)
        edges = self.batched_edges(batch)
        for block in self.blocks:
            x = block(x, edges)
        return x.reshape(batch, self.node_count, self.cfg.enc_dim)

    def encode_nodes(self, obs):
        """Return [..., nodes, width] with independent graph batch indexing."""
        leading = obs.shape[:-1]
        nodes = node_features(obs, self.cfg).reshape(-1, self.node_count, self.input_projection.in_features)
        outputs = []
        for chunk in nodes.split(self.cfg.graph_batch_size):
            if self.cfg.graph_checkpoint and torch.is_grad_enabled():
                outputs.append(checkpoint(self._nodes, chunk, use_reentrant=False,
                                          preserve_rng_state=self.cfg.graph_attention_dropout != 0))
            else:
                outputs.append(self._nodes(chunk))
        return torch.cat(outputs).reshape(*leading, self.node_count, self.cfg.enc_dim)

    def forward(self, obs):
        return self.output_projection(self.encode_nodes(obs).mean(-2))
