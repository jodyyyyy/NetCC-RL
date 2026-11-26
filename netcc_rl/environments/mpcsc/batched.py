"""GPU-parallel MPCSC physics; unchanged CPU implementation remains the oracle.

Random resets use independent counter streams per slot, reproducing the original
reset distributions rather than Python/Torch CPU RNG bitstreams. Tests supply
identical physical states for trajectory equivalence. Boids retains sequential
node updates; each node operation advances all environments in parallel on GPU.
"""

import math
import torch
from .config import from_snapshot
from .env import MPCSCEnv


class BatchedMPCSC:

    def __init__(self, snapshot, batch_size, *, device="cuda:0", seed=0):
        self.config = from_snapshot(snapshot)
        self.p = self.config.params
        self.task = self.config.task
        self.device = torch.device(device)
        self.batch_size = batch_size
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        oracle = MPCSCEnv(self.config, seed=seed)
        oracle.reset(seed=seed)
        b = oracle.backend
        self.n = self.p["node_num"]
        self.drivers = torch.tensor(oracle.driver_nodes, device=self.device)
        self.action_dim = len(self.drivers)
        self.elapsed = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.keys = (
            torch.arange(batch_size, device=self.device, dtype=torch.long)
            + seed * 104729
        )
        self.counter = torch.zeros_like(self.elapsed)
        self.adjacency = b.adj_matrix.to(self.device)
        from ..graph.frozen import artifact

        layout = artifact(snapshot)
        self.degree = (
            torch.tensor(
                layout["degrees"], device=self.device, dtype=self.adjacency.dtype
            )
            if layout
            else self.adjacency.sum(-1).clamp_min(1)
        )
        if self.task == "mpcsc-kuramoto":
            self.omegas = b.init_omegas.to(self.device)
        else:
            self.default = torch.zeros(self.n, device=self.device)
            self.default[b.default_initial_I_node_list] = 1
        self.state = torch.zeros(batch_size, self.n, device=self.device)
        self.reset()
        oracle.close()

    def uniform(self, width):
        x = (
            self.keys[:, None] * 747796405
            + self.counter[:, None] * 2891336453
            + torch.arange(width, device=self.device) * 277803737
            & 4294967295
        )
        x = (x ^ x >> 16) * 2246822519 & 4294967295
        x = (x ^ x >> 13) * 3266489917 & 4294967295
        return ((x ^ x >> 16).double() + 0.5) / 4294967296.0

    def reset(self, mask=None, *, initial_state=None):
        mask = (
            torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
            if mask is None
            else mask
        )
        if initial_state is not None:
            new = initial_state.to(self.device, dtype=self.state.dtype)
        elif self.task == "mpcsc-kuramoto":
            values = (
                torch.arange(1, self.n + 1, device=self.device)
                * (2.0 / self.n)
                * math.pi
            )
            new = (
                values[self.uniform(self.n).argsort(-1)]
                if self.config.random_init
                else values.expand_as(self.state)
            )
        elif self.task == "mpcsc-sis":
            new = (
                torch.zeros_like(self.state).scatter_(
                    1,
                    self.uniform(self.n).argsort(-1)[:, : self.p["initial_I_node_num"]],
                    1,
                )
                if self.config.random_init
                else self.default.expand_as(self.state)
            )
        if new.shape != self.state.shape:
            raise ValueError("invalid initial state shape")
        self.state = torch.where(
            mask.reshape(self.batch_size, *[1] * (self.state.ndim - 1)), new, self.state
        )
        self.elapsed = torch.where(mask, 0, self.elapsed)
        self.counter += mask.long()
        return self.observe()

    def observe(self):
        return self.state.clone()

    def step(self, action, active=None):
        if (
            action.shape != (self.batch_size, self.action_dim)
            or action.device != self.device
        ):
            raise ValueError("device action shape mismatch")
        action = action.float().clamp(-1, 1)
        active = (
            torch.ones_like(self.elapsed, dtype=torch.bool)
            if active is None
            else active
        )
        u = torch.zeros_like(self.state)
        native = (action + 1) / 2 if self.task == "mpcsc-sis" else action
        u[:, self.drivers] = native
        if self.task == "mpcsc-kuramoto":
            coupling = (
                torch.sin(self.state[:, None, :] - self.state[:, :, None])
                * self.adjacency
            ).sum(-1) * self.p["coupling"]
            new = self.state + 0.01 * (self.omegas + coupling + 20 * u)
            reward = (
                new.cos().sum(-1).square() + new.sin().sum(-1).square()
            ).sqrt() / self.n - self.p.get(
                "kuramoto_action_cost_coef", 0
            ) * action.square().mean(
                -1
            )
        else:
            pressure = self.state @ self.adjacency.T / self.degree
            derivative = (
                -self.p["gamma"] * self.state
                + (self.p["beta"] - u).relu() * (1 - self.state) * pressure
            )
            new = (self.state + 0.1 * derivative).clamp(0, 1)
            reward = -new.mean(-1) - self.p.get(
                "sis_control_cost_coef", 0.1
            ) * native.mean(-1)
        if self.p["noise_std"]:
            raise ValueError(
                "batched noisy MPCSC is not yet supported; no silent noise omission"
            )
        self.state = torch.where(
            active.reshape(self.batch_size, *[1] * (self.state.ndim - 1)),
            new,
            self.state,
        )
        self.elapsed += active.long()
        self.counter += active.long()
        info = {}
        if self.task == "mpcsc-sis":
            intensity = native.mean(-1)
            cost = self.p.get("sis_control_cost_coef", 0.1) * intensity
            info = dict(
                infection_reward=-new.mean(-1),
                control_cost=cost,
                total_reward=reward,
                mean_control_intensity=intensity,
            )
        return (
            self.observe(),
            torch.where(active, reward.float(), 0),
            torch.zeros_like(active),
            self.elapsed >= self.config.max_episode_steps,
            info,
        )
