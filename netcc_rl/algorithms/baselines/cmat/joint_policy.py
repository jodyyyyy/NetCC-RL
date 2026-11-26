"""Independent paper-based continuous MAT, CMAT and JointPPO adaptations.

No upstream implementation is incorporated. Rectangular attention maps the 100
physical state tokens to the frozen 50 actuator slots. Raw phases enter only the
parameter-free sin/cos transform. Policies use bounded tanh-normal actions.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from .phase_features import phase_features
TRANSFORMERS = ('cmat',)


class Compression(nn.Module):
    """Learned attention pooling, preserving gradients through every input token."""
    def __init__(self, width, heads=4):
        super().__init__()
        self.weights = nn.Linear(width, heads)
        self.output = nn.Sequential(nn.Linear(heads * width, width), nn.GELU())

    def forward(self, tokens):
        weights = self.weights(tokens).softmax(dim=-2)
        return self.output(torch.einsum('bnh,bnd->bhd', weights, tokens).flatten(1))


def position(length, width, device):
    i = torch.arange(length, device=device)[:, None]
    scale = torch.exp(torch.arange(0, width, 2, device=device) * (-math.log(10000.) / width))
    result = torch.zeros(length, width, device=device)
    result[:, 0::2] = (i * scale).sin()
    result[:, 1::2] = (i * scale).cos()
    return result


def _attention_heads(tensor, heads):
    batch, length, width = tensor.shape
    if width % heads:
        raise ValueError("attention width must be divisible by its head count")
    return tensor.reshape(batch, length, heads, width // heads).transpose(1, 2)


def _attention_output(attention, query, key, value):
    attended = F.scaled_dot_product_attention(
        query, key, value,
        dropout_p=float(attention.dropout) if attention.training else 0.0,
        is_causal=False,
    )
    batch, _, query_length, _ = attended.shape
    attended = attended.transpose(1, 2).contiguous().reshape(
        batch, query_length, attention.embed_dim
    )
    return F.linear(attended, attention.out_proj.weight, attention.out_proj.bias)


def init_incremental_decoder_cache(decoder, memory):
    """Project each decoder layer's cross-attention K/V once per CMAT pass.

    The cache contains no detached tensors: gradients still flow through the
    encoder memory and every autoregressive consensus step.
    """
    caches = []
    for layer in decoder.layers:
        if layer.norm_first:
            raise ValueError("incremental CMAT decoding requires post-norm decoder layers")
        for attention in (layer.self_attn, layer.multihead_attn):
            if (
                not attention.batch_first
                or attention.in_proj_weight is None
                or attention.bias_k is not None
                or attention.bias_v is not None
                or attention.add_zero_attn
            ):
                raise ValueError("unsupported MultiheadAttention configuration for CMAT cache")

        cross_attention = layer.multihead_attn
        _, key_weight, value_weight = cross_attention.in_proj_weight.chunk(3, dim=0)
        if cross_attention.in_proj_bias is None:
            key_bias = value_bias = None
        else:
            _, key_bias, value_bias = cross_attention.in_proj_bias.chunk(3, dim=0)
        memory_key = _attention_heads(
            F.linear(memory, key_weight, key_bias), cross_attention.num_heads
        )
        memory_value = _attention_heads(
            F.linear(memory, value_weight, value_bias), cross_attention.num_heads
        )
        # (self K, self V, cross-attention K, cross-attention V)
        caches.append((None, None, memory_key, memory_value))
    return caches


def incremental_decoder_step(decoder, token, caches):
    """Decode one new target token using per-layer self- and cross-attention KV."""
    hidden = token
    next_caches = []
    for layer, (past_key, past_value, memory_key, memory_value) in zip(
        decoder.layers, caches
    ):
        self_attention = layer.self_attn
        query, key, value = F.linear(
            hidden, self_attention.in_proj_weight, self_attention.in_proj_bias
        ).chunk(3, dim=-1)
        query = _attention_heads(query, self_attention.num_heads)
        key = _attention_heads(key, self_attention.num_heads)
        value = _attention_heads(value, self_attention.num_heads)
        all_keys = key if past_key is None else torch.cat((past_key, key), dim=2)
        all_values = value if past_value is None else torch.cat((past_value, value), dim=2)

        self_output = _attention_output(self_attention, query, all_keys, all_values)
        hidden = layer.norm1(hidden + layer.dropout1(self_output))

        cross_attention = layer.multihead_attn
        query_weight = cross_attention.in_proj_weight[:cross_attention.embed_dim]
        query_bias = (
            None if cross_attention.in_proj_bias is None
            else cross_attention.in_proj_bias[:cross_attention.embed_dim]
        )
        cross_query = _attention_heads(
            F.linear(hidden, query_weight, query_bias), cross_attention.num_heads
        )
        cross_output = _attention_output(
            cross_attention, cross_query, memory_key, memory_value
        )
        hidden = layer.norm2(hidden + layer.dropout2(cross_output))

        feed_forward = layer.linear2(
            layer.dropout(layer.activation(layer.linear1(hidden)))
        )
        hidden = layer.norm3(hidden + layer.dropout3(feed_forward))
        next_caches.append((all_keys, all_values, memory_key, memory_value))

    if decoder.norm is not None:
        hidden = decoder.norm(hidden)
    return hidden, next_caches


def incremental_consensus(decoder, memory, initial, latent_projection, steps):
    """Build the same consensus trajectory with incremental Transformer decoding.

    The original full-prefix decoder reprojects all past target tokens and all
    encoder-memory K/V at every step. This version computes memory K/V once and
    each target token's self-attention K/V once, while retaining full autograd
    history (no cache detach and no change to the number of consensus steps).
    """
    if steps < 0:
        raise ValueError("consensus steps must be non-negative")
    caches = init_incremental_decoder_cache(decoder, memory)
    positions = position(steps, initial.shape[-1], memory.device)
    trajectory = [initial]
    current = initial.unsqueeze(1)
    for step in range(steps):
        token = current + positions[step:step + 1].unsqueeze(0)
        decoded, caches = incremental_decoder_step(decoder, token, caches)
        current = latent_projection(decoded)
        trajectory.append(current.squeeze(1))
    return torch.stack(trajectory, dim=1)


class JointPolicy(nn.Module):
    def __init__(self, algorithm, width=64, heads=1, blocks=1, consensus_steps=50, *, controls):
        super().__init__()
        if algorithm not in TRANSFORMERS:
            raise ValueError(algorithm)
        self.algorithm = algorithm
        self.width = width
        self.consensus_steps = consensus_steps
        self.register_buffer('drivers', torch.tensor(controls, dtype=torch.long))
        self.embedding = nn.Linear(2, width)
        encoder = nn.TransformerEncoderLayer(width, heads, width * 2, dropout=0.,
                                            activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, blocks, enable_nested_tensor=False)
        decoder = nn.TransformerDecoderLayer(width, heads, width * 2, dropout=0.,
                                            activation='gelu', batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder, blocks)
        self.log_std = nn.Parameter(torch.ones(50))
        if algorithm == 'cmat':
            self.critic_compressor = Compression(width)
            self.actor_compressor = Compression(width)
            self.latent_projection = nn.Linear(width, width)
            self.actor = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1))
        else:
            self.previous_action = nn.Linear(1, width)
            self.actor = nn.Linear(width, 1)
        self.critic = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def encode(self, raw):
        if raw.shape[-1] != 100 or not torch.isfinite(raw).all():
            raise ValueError('expected finite raw 100-node state')
        tokens = phase_features(raw).reshape(*raw.shape[:-1], 100, 2)
        embedded = self.embedding(tokens)
        # CMAT intentionally has no physical-token positional encoding.
        if self.algorithm != 'cmat':
            embedded = embedded + position(100, self.width, raw.device)
        return self.encoder(embedded)

    def decode(self, target, memory):
        n = target.shape[1]
        mask = torch.ones(n, n, dtype=torch.bool, device=target.device).triu(1)
        return self.decoder(target, memory, tgt_mask=mask)

    def consensus(self, memory):
        initial = self.critic_compressor(memory)
        trajectory = incremental_consensus(
            self.decoder, memory, initial, self.latent_projection, self.consensus_steps
        )
        return initial, self.actor_compressor(trajectory)

    def components(self, memory):
        if self.algorithm == 'cmat':
            initial, consensus = self.consensus(memory)
            own = memory[:, self.drivers]
            mean = self.actor(torch.cat((own, consensus[:, None].expand_as(own)), -1)).squeeze(-1)
            return mean, self.critic(initial)
        values = self.critic(memory[:, self.drivers]).squeeze(-1) if self.algorithm == 'mat' else self.critic(memory.mean(1))
        return None, values

    def conditional_mean(self, memory, actions):
        # Teacher forcing: slot i depends only on actions < i; BOS is zero.
        previous = torch.cat((torch.zeros_like(actions[:, :1]), actions[:, :-1]), 1)
        target = memory[:, self.drivers[:actions.shape[1]]] + self.previous_action(previous[..., None])
        return self.actor(self.decode(target, memory)).squeeze(-1)

    def distribution(self, mean):
        return Normal(mean, self.log_std.sigmoid() * .5)

    @staticmethod
    def log_probability(distribution, latent):
        # Stable tanh Jacobian, including saturated samples (no inverse tanh).
        correction = 2 * (math.log(2) - latent - torch.nn.functional.softplus(-2 * latent))
        return distribution.log_prob(latent) - correction

    def forward(self, raw, latent):
        memory = self.encode(raw)
        mean, values = self.components(memory)
        if mean is None:
            mean = self.conditional_mean(memory, latent.tanh())
        dist = self.distribution(mean)
        # Conditional bounded entropy at teacher-forced rollout prefixes.
        # For AR policies this is a PPO regularizer, not an unbiased estimate
        # of full current-policy joint entropy (prefixes come from rollout).
        entropy = -self.log_probability(dist, dist.rsample())
        return self.log_probability(dist, latent), values, entropy

    @torch.no_grad()
    def act(self, raw, deterministic=False):
        memory = self.encode(raw)
        mean, values = self.components(memory)
        if mean is not None:
            distribution = self.distribution(mean)
            latent = mean if deterministic else distribution.sample()
        else:
            latents = []
            for i in range(50):
                prefix = torch.stack(latents, 1).tanh() if latents else raw.new_empty((len(raw), 0))
                # Last placeholder cannot affect its own conditional mean.
                actions = torch.cat((prefix, raw.new_zeros((len(raw), 1))), 1)
                mu = self.conditional_mean(memory, actions)[:, -1]
                dist = Normal(mu, self.log_std[i].sigmoid() * .5)
                latents.append(mu if deterministic else dist.sample())
            latent = torch.stack(latents, 1)
            distribution = self.distribution(self.conditional_mean(memory, latent.tanh()))
        return latent.tanh(), latent, self.log_probability(distribution, latent), values

    @torch.no_grad()
    def predict(self, observation):
        raw = torch.as_tensor(np.asarray(observation), dtype=torch.float32,
                              device=self.log_std.device).reshape(1, 100)
        return self.act(raw, deterministic=True)[0][0].cpu().numpy()


def surrogate(algorithm, logp, old_logp, advantage, clip):
    if algorithm != 'mat':
        logp = logp.sum(-1, keepdim=True)
        old_logp = old_logp.sum(-1, keepdim=True)
    ratio = (logp - old_logp).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
