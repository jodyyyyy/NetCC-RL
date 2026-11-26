"""EXP-011 PyG baseline: source -> target COO edges and mean graph pooling."""
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .layers import SimNorm, mlp
from .node_features import node_features, node_layout, prior_adjacency


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


def symmetric_candidate_edges(edges, nodes):
    """Validate simple per-graph COO; return exact symmetry, never guess.

    Called only at CSR construction, outside capture. Batch offsets then prove
    that each copied edge remains within its own graph.
    """
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
        raise ValueError('candidate edges must be int64 [2, E]')
    pairs = edges.detach().cpu().T.tolist()
    if any(not (0 <= u < nodes and 0 <= v < nodes) for u, v in pairs):
        raise ValueError('candidate edge outside graph (cross-graph edge)')
    unique = set(map(tuple, pairs))
    if len(unique) != len(pairs):
        raise ValueError('duplicate candidate edges')
    if any((i, i) not in unique for i in range(nodes)):
        raise ValueError('exactly one self-loop per node required')
    return all((v, u) in unique for u, v in unique)


class GATv2Block(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        self.conv = GATv2Conv(width, width // heads, heads=heads,
                             dropout=dropout, add_self_loops=False,
                             share_weights=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, x, edges, *, turbo_graph=None, fused_projection=False, kernel_options=None):
        if turbo_graph is None:
            attended = self.conv(x, edges)
        else:
            from .turbo_gatv2 import turbo_aggregate
            attended = turbo_aggregate(self.conv, x, turbo_graph, fused_projection=fused_projection,
                                       kernel_options=kernel_options)
        return self.norm(x + attended)


class GATv2StateEncoder(nn.Module):
    """Two configurable residual GATv2 layers, node embeddings, global SimNorm.

    Graph microbatches bound edge tensor memory. Activation checkpointing is
    exact for dropout=0; PyTorch preserves RNG state when dropout is enabled.
    Frozen profiles prepare permanent CSR owners before execution; legacy profiles retain their bounded cache. Device/dtype moves require preparation again.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backend = getattr(cfg, 'graph_backend', 'pyg')
        if self.backend not in {'pyg', 'turbo'}:
            raise ValueError('graph_backend must be pyg or turbo')
        if self.backend == 'turbo' and cfg.graph_attention_dropout != 0:
            raise ValueError('Turbo GATv2 does not support attention dropout')
        if self.backend == 'turbo' and cfg.enc_dim // cfg.graph_num_heads not in {32, 64, 128, 256}:
            raise ValueError('Turbo GATv2 head width must be 32, 64, 128, or 256')
        self._turbo_graphs = OrderedDict()
        from netcc_rl.environments.graph.frozen import for_config
        self._frozen=for_config(cfg)
        self._static_ready=False
        self._static_edges={}
        self.static_preparations=0
        self.static_materializations=0
        self.turbo_symmetric_csr = getattr(cfg, 'turbo_symmetric_csr', False)
        self.turbo_fused_projection = getattr(cfg, 'turbo_fused_projection', False)
        self.turbo_kernel_options = {
            name: getattr(cfg, 'turbo_' + name, default) for name, default in (
                ('grad_A_reduce_row_chunk_size', 512), ('forward_light_warps', 1),
                ('forward_heavy_warps', 8), ('backward_light_warps', 1),
                ('backward_heavy_warps', 8))}
        if getattr(cfg, 'turbo_save_output_for_backward', False):
            self.turbo_kernel_options['save_output_for_backward'] = True
        self.turbo_quantiles = (getattr(cfg, 'turbo_forward_quantile', .99),
                                getattr(cfg, 'turbo_backward_quantile', .99))
        self.node_count, _, features, _ = node_layout(cfg)
        self.input_projection = nn.Linear(features, cfg.enc_dim)
        self.blocks = nn.ModuleList(GATv2Block(cfg.enc_dim, cfg.graph_num_heads,
                                   cfg.graph_attention_dropout)
                                   for _ in range(cfg.graph_num_layers))
        self.output_projection = mlp(cfg.enc_dim,
            max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim], cfg.latent_dim,
            act=SimNorm(cfg))
        # Rebuilt from the checkpoint's frozen environment, like EXP-007 buffers.
        if self._frozen is not None and cfg.state_encoder_type!=self._frozen['encoder']:
            raise ValueError('frozen encoder topology differs')
        edges=(torch.tensor(self._frozen['candidate_edges'],dtype=torch.long).T.contiguous()
               if self._frozen is not None else candidate_edges(cfg))
        self.register_buffer('edge_index', edges, persistent=False)

    def _apply(self, fn, recurse=True):
        self._turbo_graphs.clear()
        self._static_edges.clear()
        self._static_ready=False
        return super()._apply(fn, recurse=recurse)

    def prepare_static_graphs(self):
        """Once before execution/capture: materialize frozen indices, no sorting."""
        if self._frozen is None or self._static_ready:return
        if self.edge_index.device.type=='cuda' and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('static graph preparation cannot run inside capture')
        if self.turbo_quantiles!=tuple(self._frozen['quantiles']):
            raise ValueError('bucket quantiles differ from frozen preprocessing')
        if self.edge_index.dtype!=torch.long or self.input_projection.weight.dtype!=torch.float32:
            raise ValueError('frozen graph supports int64 COO / FP32 model only')
        device=self.edge_index.device;n=self.node_count
        for shape,buckets in self._frozen['shapes'].items():
            b=int(shape);offsets=torch.arange(b,device=device,dtype=torch.int32)*n
            if self.backend=='turbo':
                if device.type!='cuda':raise ValueError('Turbo static preparation requires CUDA')
                from .verified_turbo import verified_turbo
                fields={}
                for direction in ('forward','backward'):
                    csr=self._frozen['csr'][direction];e=len(csr['indices'])
                    ptr=torch.tensor(csr['indptr'][:-1],device=device,dtype=torch.int32)
                    fields[direction+'_indptr']=torch.cat(((ptr[None,:]+torch.arange(b,device=device,dtype=torch.int32)[:,None]*e).flatten(),torch.tensor([b*e],device=device,dtype=torch.int32)))
                    col=torch.tensor(csr['indices'],device=device,dtype=torch.int32)
                    fields[direction+'_indices']=(col[None,:]+offsets[:,None]).flatten()
                    for size in ('light','heavy'):
                        name=direction+'_'+size+'_nodes'
                        ids=torch.tensor(buckets[name],device=device,dtype=torch.int32)
                        fields[name]=(ids[None,:]+offsets[:,None]).flatten()
                graph=verified_turbo().AdjacencyForwardBackwardWithNodeBuckets(**fields,
                    is_directed=not(self.turbo_symmetric_csr and self._frozen['symmetric']))
                key=(b,device,self.edge_index.data_ptr(),self.edge_index._version)
                self._turbo_graphs[key]=graph
            else:
                self._static_edges[b]=(self.edge_index[:,None,:]+offsets.to(torch.long)[None,:,None]).reshape(2,-1)
            self.static_materializations+=1
        self.static_preparations+=1
        self._static_ready=True

    def turbo_graph(self, graphs):
        if self.edge_index.device.type != 'cuda':
            raise ValueError('Turbo GATv2 requires CUDA; no silent PyG fallback')
        key = (graphs, self.edge_index.device, self.edge_index.data_ptr(), self.edge_index._version)
        if self._frozen is not None and (not self._static_ready or key not in self._turbo_graphs):
            raise RuntimeError('unknown or unprepared frozen graph shape; explicit preprocessing/preparation required')
        if key not in self._turbo_graphs:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Turbo CSR cache miss during CUDA Graph capture; prime all shapes first')
            symmetric = symmetric_candidate_edges(self.edge_index, self.node_count)
            from .verified_turbo import verified_turbo
            AdjacencyForwardBackwardWithNodeBuckets = verified_turbo().AdjacencyForwardBackwardWithNodeBuckets
            self._turbo_graphs[key] = AdjacencyForwardBackwardWithNodeBuckets.from_edge_list(
                self.batched_edges(graphs), num_nodes=graphs*self.node_count,
                index_dtype=torch.int32, is_directed=not (self.turbo_symmetric_csr and symmetric))
            if self.turbo_quantiles != (.99, .99):
                self._turbo_graphs[key] = self._turbo_graphs[key].repartition(
                    forward_huge_degree_threshold_quantile=self.turbo_quantiles[0],
                    backward_huge_degree_threshold_quantile=self.turbo_quantiles[1])
            # Four graph batch sizes bound memory across inference and tail shapes.
            if len(self._turbo_graphs) > 4:
                self._turbo_graphs.popitem(last=False)
        self._turbo_graphs.move_to_end(key)
        return self._turbo_graphs[key]

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
        if self.backend == 'turbo':
            graph = self.turbo_graph(batch)
            for block in self.blocks:
                x = block(x, None, turbo_graph=graph,
                          fused_projection=self.turbo_fused_projection,
                          kernel_options=self.turbo_kernel_options)
        else:
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
