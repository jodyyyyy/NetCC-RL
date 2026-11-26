"""Small graph control systems, FP32 Torch physics with leading batch dimensions.

These are documented synthetic baselines, not IEEE/traffic simulator replicas.
"""

from copy import deepcopy
import hashlib
import json
import math
import gymnasium as gym
import numpy as np
import torch

TASKS = ("graph-swing",)
DEFAULTS = dict(
    n=None,
    topology=None,
    dt=None,
    substeps=4,
    episode_length=150,
    control_fraction=0.5,
    coupling=2.0,
    inertia=2.0,
    damping=1.0,
    acceleration=3.0,
    actuator_tau=0.5,
    target_speed=20.0,
    target_gap=15.0,
    disturbance=0.1,
    cell_length=0.5,
    lanes=2.0,
    capacity=1800.0,
    free_speed=90.0,
    wave_speed=20.0,
    jam_density=150.0,
    demand=2000.0,
    ramp_demand=200.0,
    ramp_capacity=600.0,
    effort=0.01,
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def resolve(task, params=None, *, selection_profile=None):
    if task not in TASKS:
        raise ValueError(f"unsupported graph task {task}")
    params = dict(params or {})
    if params.keys() - DEFAULTS.keys():
        raise ValueError(f"unknown graph parameters: {params.keys() - DEFAULTS.keys()}")
    p = {**DEFAULTS, **params}
    p["n"] = p["n"] if p["n"] is not None else 39
    p["topology"] = p["topology"] or "ring"
    p["dt"] = p["dt"] if p["dt"] is not None else 0.05
    for key in ("n", "substeps", "episode_length"):
        if type(p[key]) is not int or p[key] < (3 if key == "n" else 1):
            raise ValueError(f"{key} must be a positive integer (n>=3)")
    for key, value in p.items():
        if key not in ("n", "substeps", "episode_length", "topology"):
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or (not math.isfinite(value))
                or (value < 0)
            ):
                raise ValueError(f"invalid finite nonnegative parameter {key}")
    for key in (
        "dt",
        "inertia",
        "actuator_tau",
        "cell_length",
        "lanes",
        "capacity",
        "free_speed",
        "wave_speed",
        "jam_density",
        "target_gap",
    ):
        if p[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 < p["control_fraction"] <= 1:
        raise ValueError("invalid control_fraction")
    if p["dt"] / p["substeps"] > 0.05 and True:
        raise ValueError("integration substep must be <= .05 seconds")
    allowed = {"graph-swing": {"ring", "chain", "ieee39-core38-v1"}}
    if p["topology"] not in allowed[task]:
        raise ValueError("unsupported topology")
    n = p["n"]
    adj = torch.zeros(n, n)
    for i in range(n - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    if p["topology"] == "ring":
        adj[-1, 0] = 1
        adj[0, -1] = 1
    topology_source = None
    if p["topology"] == "ieee39-core38-v1":
        from .topology import ieee39_core38

        if n != 38:
            raise ValueError("IEEE39 core topology requires n=38")
        a, topology_source = ieee39_core38()
        adj = torch.from_numpy(a)
    candidates = list(range(n))
    drivers = candidates[:: max(1, round(1 / p["control_fraction"]))]
    f = 5
    snapshot = dict(
        version=1,
        task=task,
        params=p,
        adjacency=adj.tolist(),
        driver_nodes=drivers,
        raw_features=f,
        node_order=list(range(n)),
    )
    if selection_profile is not None:
        if p["control_fraction"] != 0.5:
            raise ValueError("invalid research selection profile/budget")
        if selection_profile == "grounded_distance_v1" and topology_source:
            from .topology import grounded_distance_greedy

            drivers, selection = grounded_distance_greedy(adj.numpy(), n // 2)
        else:
            raise ValueError("invalid research selection profile/budget")
        snapshot.update(
            version=2,
            driver_nodes=drivers,
            selection_profile=selection_profile,
            selection=selection,
            driver_sha256=digest(drivers),
            topology_sha256=digest(adj.tolist()),
        )
    if topology_source:
        snapshot.update(
            version=3,
            environment_version="graph-swing-ieee39-core38-v1",
            topology_source=topology_source,
        )
    snapshot["sha256"] = digest(snapshot)
    return snapshot


class GraphDynamics:
    """Device-native batch simulator. reset is explicit; step never auto-resets.

    action is normalized [-1,1], including CTM (mapped to release ratio [0,1]).
    step returns tensors with the supplied batch shape; partial batches can use
    a boolean active mask without advancing uncollected slots or RNG streams.
    """

    def __init__(
        self, task, *, params=None, snapshot=None, device="cpu", batch_size=1, seed=0
    ):
        if task not in TASKS:
            raise ValueError(f"unsupported graph task {task}")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        if snapshot is not None:
            from .frozen import validate_snapshot

            self.snapshot, frozen = validate_snapshot(task, snapshot, params)
        else:
            if (params or {}).get("topology") == "ieee39-core38-v1":
                raise ValueError(
                    "versioned IEEE39 topology requires a preprocessed frozen snapshot"
                )
            self.snapshot = resolve(task, params)
            from .frozen import artifact

            frozen = artifact(self.snapshot)
        self.task, self.p, self.device, self.batch_size = (
            task,
            self.snapshot["params"],
            torch.device(device),
            batch_size,
        )
        self.n = self.p["n"]
        self.f = self.snapshot["raw_features"]
        self.drivers = torch.tensor(self.snapshot["driver_nodes"], device=self.device)
        self.action_dim = len(self.drivers)
        self.adjacency = torch.tensor(self.snapshot["adjacency"], device=self.device)
        self.edges = (
            torch.tensor(
                frozen["physical_edges"], device=self.device, dtype=torch.long
            ).T.contiguous()
            if frozen
            else self.adjacency.nonzero().T.contiguous()
        )
        self.edge_features = (
            torch.tensor(
                [
                    self.p["coupling"],
                    self.p["capacity"] * self.p["lanes"],
                    self.p["cell_length"],
                    self.p["lanes"],
                ],
                device=self.device,
            )
            .expand(self.edges.shape[1], 4)
            .clone()
        )
        self.elapsed = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.state = torch.zeros(batch_size, self.n, 3, device=self.device)
        self.reset_counter = torch.zeros_like(self.elapsed)
        self.reset(seed=seed)

    def reset(self, seed=None, *, mask=None):
        """Reset selected slots with device-native, independent counter streams.

        Other slots retain their state, elapsed time and next random sample.
        Explicit reseeding applies to the complete batch only. This changes the
        old shared-generator reset bitstream, not the reset distributions.
        """
        if seed is not None and mask is not None:
            raise ValueError("explicit seed requires a full-batch reset")
        mask = (
            torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
            if mask is None
            else torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        )
        if mask.shape != (self.batch_size,):
            raise ValueError("invalid reset mask")
        if seed is not None:
            self.reset_keys = (
                torch.arange(self.batch_size, device=self.device, dtype=torch.long)
                + int(seed) * 104729
            )
            self.reset_counter.zero_()
        x = (
            self.reset_keys[:, None] * 747796405
            + self.reset_counter[:, None] * 2891336453
            + torch.arange(self.n, device=self.device) * 277803737
            & 4294967295
        )
        x = (x ^ x >> 16) * 2246822519 & 4294967295
        x = (x ^ x >> 13) * 3266489917 & 4294967295
        noise = (((x ^ x >> 16).double() + 0.5) / 4294967296.0 - 0.5).float()
        state = torch.zeros_like(self.state)
        state[..., 0] = 0.2 * noise
        state[..., 1] = 0.1 * noise
        injection = 0.1 * torch.sin(
            torch.arange(self.n, device=self.device) * 2 * math.pi / self.n
        )
        self.injection = injection - injection.mean()
        self.state = torch.where(mask[:, None, None], state, self.state)
        self.elapsed = torch.where(mask, 0, self.elapsed)
        self.reset_counter += mask.long()
        return self.observe()

    def observe(self):
        constants = torch.stack(
            (
                torch.full_like(self.injection, self.p["inertia"]),
                torch.full_like(self.injection, self.p["damping"]),
                self.injection,
            ),
            -1,
        )
        obs = torch.cat(
            (self.state[..., :2], constants.expand(self.batch_size, -1, -1)), -1
        )
        return obs.flatten(1).clone()

    def step(self, action, active=None):
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        if (
            action.shape != (self.batch_size, self.action_dim)
            or not torch.isfinite(action).all()
        ):
            raise ValueError("action must have finite batch x action_dim values")
        action = action.clamp(-1, 1)
        active = (
            torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
            if active is None
            else torch.as_tensor(active, device=self.device, dtype=torch.bool)
        )
        if active.shape != (self.batch_size,):
            raise ValueError("invalid active mask")
        if ((self.elapsed >= self.p["episode_length"]) & active).any():
            raise RuntimeError("reset required after truncation")
        u = torch.zeros(self.batch_size, self.n, device=self.device)
        u[:, self.drivers] = action
        old = self.state

        def rhs(s):
            theta, omega = (s[..., 0], s[..., 1])
            coupling = (
                torch.sin(theta.unsqueeze(-2) - theta.unsqueeze(-1)) * self.adjacency
            ).sum(-1) * self.p["coupling"]
            return torch.stack(
                (
                    omega,
                    (self.injection + u + coupling - self.p["damping"] * omega)
                    / self.p["inertia"],
                ),
                -1,
            )

        s = old[..., :2]
        h = self.p["dt"] / self.p["substeps"]
        for _ in range(self.p["substeps"]):
            k1 = rhs(s)
            k2 = rhs(s + h * k1 / 2)
            k3 = rhs(s + h * k2 / 2)
            k4 = rhs(s + h * k3)
            s = s + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        new = torch.cat((s, old[..., 2:]), -1)
        error = s[..., 1].square().mean(-1) + (
            1 - torch.cos(s[..., 0] - s[..., 0].mean(-1, keepdim=True))
        ).mean(-1)
        reward = -error - self.p["effort"] * action.square().mean(-1)
        self.state = torch.where(active[:, None, None], new, old)
        self.elapsed += active.long()
        return (
            self.observe(),
            torch.where(active, reward, 0),
            torch.zeros_like(active),
            self.elapsed >= self.p["episode_length"],
            {},
        )


class GraphEnv(gym.Env):

    def __init__(self, task, *, params=None, snapshot=None, seed=0):
        self.sim = GraphDynamics(task, params=params, snapshot=snapshot, seed=seed or 0)
        self.action_space = gym.spaces.Box(-1, 1, (self.sim.action_dim,), np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.sim.n * self.sim.f,), np.float32
        )
        self.max_episode_steps = self.sim.p["episode_length"]
        self.no_control_action = np.full(self.sim.action_dim, 0.0, np.float32)
        self.action_space.seed(seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        return (self.sim.reset(seed)[0].cpu().numpy(), {})

    def step(self, action):
        obs, r, t, tr, info = self.sim.step(torch.as_tensor(action).reshape(1, -1))
        return (obs[0].cpu().numpy(), float(r[0]), bool(t[0]), bool(tr[0]), info)

    def snapshot(self):
        return deepcopy(self.sim.snapshot)
