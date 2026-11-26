"""Encoder, actor and learner variants for the EXP-017 factor design."""

from functools import lru_cache
import torch
from torch import nn
from .layers import NormedLinear, SimNorm, mlp
from .node_features import node_features, node_layout


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
        # The same builder, shape, normalization and forward path as main G2.
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


def model_free_update(agent, obs, action, reward, terminated, task, model_valid_mask):
    """A9 real-transition critic update, then unchanged detached-latent actor."""
    agent._validate_batch(obs, action, reward, terminated, task)
    valid = (
        torch.ones(action.shape[:2], device=action.device, dtype=torch.bool)
        if model_valid_mask is None
        else model_valid_mask
    )
    if (
        valid.dtype != torch.bool
        or valid.shape != action.shape[:2]
        or not bool(valid.any())
    ):
        raise ValueError("model-free update requires boolean valid real transitions")
    obs_valid = torch.cat((torch.ones_like(valid[:1]), valid), 0)
    obs = torch.where(obs_valid[..., None], obs, 0)
    action, reward, terminated = (
        torch.where(valid[..., None], x, 0) for x in (action, reward, terminated)
    )
    model = agent.model
    agent.optim.zero_grad(set_to_none=True)
    with torch.no_grad():
        next_nodes, next_z = model.encode_actor_features(obs[1:])
        targets = agent._td_target(
            next_z, reward, terminated, task, next_nodes=next_nodes
        )
    model.train()
    z = model.encode(obs[:-1], task)
    value = agent._value_loss(z, action, targets, valid, task)
    loss = agent.cfg.value_coef * value
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model._clip_parameters, agent.cfg.grad_clip_norm)
    agent._check_finite_step(loss, norm, "model-free critic")
    agent.optim.step()
    agent.optim.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        nodes, z = model.encode_actor_features(obs[:-1])
    info = agent.update_pi(z, action, actor_nodes=nodes, model_valid_mask=valid)
    model.soft_update_target_Q()
    info.update(
        value_loss=value,
        value_loss_encoder=value,
        value_loss_total=value,
        total_loss=loss,
        grad_norm=norm,
        valid_transition_count=valid.sum(),
        valid_transition_fraction=valid.float().mean(),
    )
    return {k: v.detach().float().mean() for k, v in info.items()}


def _native_norm_inductor_backend(graph, inputs):
    """Retain ATen normalization/nonlinearity numerics inside Inductor graphs.

    B512 repeated-state probes expose amplified action drift when these FP32
    operations are decomposed. Keep their native kernels (also in backward),
    while Inductor compiles the remaining tensor operations. The outer runtime
    still captures all native and generated CUDA kernels together.
    """
    from torch._inductor.compile_fx import compile_fx
    from torch._inductor.decomposition import decompositions, select_decomp_table
    from torch._inductor.lowering import lowerings, make_fallback

    native = (
        torch.ops.aten.native_layer_norm.default,
        torch.ops.aten.native_layer_norm_backward.default,
        torch.ops.aten.mish.default,
        torch.ops.aten.mish_backward.default,
    )
    table = dict(select_decomp_table())
    for op in native:
        table.pop(op, None)
        if op not in lowerings:
            # Register the native lowering without changing the default global
            # decomposition policy used by other compile entry points.
            previous = decompositions.pop(op, None)
            try:
                make_fallback(op, warn=False)
            finally:
                if previous is not None:
                    decompositions[op] = previous
    return compile_fx(
        graph,
        inputs,
        config_patches={"triton.cudagraphs": False},
        decompositions=table,
    )


def compile_modules(model):
    """Inductor tensor kernels inside the existing segmented CUDA Graph runtime.

    Compile bound forward methods, preserving state_dict keys and parameter
    ownership. Inductor's own CUDA Graph wrapper is disabled to avoid nesting
    graph pools: LearnerRuntime captures forward/backward/optimizer segments.
    fullgraph=True and suppress_errors=False make unsupported kernels fail.
    """
    if torch._dynamo.config.suppress_errors:
        raise ValueError("EXP017 compilation requires suppress_errors=false")
    names = []
    modules = [
        ("encoder", model._encoder["state"]),
        ("dynamics", model._dynamics),
        ("reward", model._reward),
        ("actor", model._pi),
    ]
    for name, module in modules:
        if module is not None:
            module.forward = torch.compile(
                module.forward, fullgraph=True, backend=_native_norm_inductor_backend
            )
            names.append(name)
    return names


def model_free_backward(agent, obs, action, reward, terminated, valid):
    """Identical D0 loss, with backward captured by the segmented runtime."""
    agent.optim.zero_grad(set_to_none=False)
    model = agent.model
    model.eval()
    with torch.no_grad():
        nodes, next_z = model.encode_actor_features(obs[1:])
        target = agent._td_target(next_z, reward, terminated, None, next_nodes=nodes)
    model.train()
    z = model.encode(obs[:-1], None)
    value = agent._value_loss(z, action, target, valid, None)
    loss = agent.cfg.value_coef * value
    loss.backward()
    return dict(
        value_loss=value.detach(),
        value_loss_encoder=value.detach(),
        value_loss_total=value.detach(),
        total_loss=loss.detach(),
    )
