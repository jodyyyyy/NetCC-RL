"""Independent paper-based continuous MAT, CMAT and JointPPO adaptations.

No upstream implementation is incorporated. Rectangular attention maps the 100
physical state tokens to the frozen 50 actuator slots. Raw phases enter only the
parameter-free sin/cos transform. Policies use bounded tanh-normal actions.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from .environment import LocalGraph
from .phase_features import phase_features, NODE_COUNT, NODE_FEATURES, RAW_DIM
from .roster import TRANSFORMERS


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


class JointPolicy(nn.Module):
    def __init__(self, algorithm, width=64, heads=1, blocks=1, consensus_steps=50):
        super().__init__()
        if algorithm not in TRANSFORMERS:
            raise ValueError(algorithm)
        self.algorithm = algorithm
        self.width = width
        self.consensus_steps = consensus_steps
        self.register_buffer('drivers', torch.tensor(LocalGraph().drivers, dtype=torch.long))
        self.embedding = nn.Linear(NODE_FEATURES, width)
        encoder = nn.TransformerEncoderLayer(width, heads, width * 2, dropout=0.,
                                            activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, blocks, enable_nested_tensor=False)
        decoder = nn.TransformerDecoderLayer(width, heads, width * 2, dropout=0.,
                                            activation='gelu', batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder, blocks)
        self.log_std = nn.Parameter(torch.ones(len(self.drivers)))
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
        if raw.shape[-1] != RAW_DIM or not torch.isfinite(raw).all():
            raise ValueError('expected finite raw 100-node state')
        tokens = phase_features(raw).reshape(*raw.shape[:-1], NODE_COUNT, NODE_FEATURES)
        embedded = self.embedding(tokens)
        # CMAT intentionally has no physical-token positional encoding.
        if self.algorithm != 'cmat':
            embedded = embedded + position(NODE_COUNT, self.width, raw.device)
        return self.encoder(embedded)

    def decode(self, target, memory):
        n = target.shape[1]
        mask = torch.ones(n, n, dtype=torch.bool, device=target.device).triu(1)
        return self.decoder(target, memory, tgt_mask=mask)

    def consensus(self, memory):
        initial = self.critic_compressor(memory)
        trajectory = initial[:, None, :]
        for _ in range(self.consensus_steps):
            target = trajectory + position(trajectory.shape[1], self.width, memory.device)
            next_latent = self.latent_projection(self.decode(target, memory)[:, -1:])
            trajectory = torch.cat((trajectory, next_latent), dim=1)
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
            for i in range(len(self.drivers)):
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
                              device=self.log_std.device).reshape(1, RAW_DIM)
        return self.act(raw, deterministic=True)[0][0].cpu().numpy()


def surrogate(algorithm, logp, old_logp, advantage, clip):
    if algorithm != 'mat':
        logp = logp.sum(-1, keepdim=True)
        old_logp = old_logp.sum(-1, keepdim=True)
    ratio = (logp - old_logp).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
