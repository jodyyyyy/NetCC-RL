"""Environment-side adapters for the official TD-MPC2 four-return interface."""

from collections.abc import Mapping, MutableMapping
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .config import from_snapshot, load_config, resolved_task_identity, task_identity
from .env import MPCSCEnv


class TDMPC2NumpyAdapter(gym.Wrapper):
    """Use with the upstream TensorWrapper, which expects old-style reset/step."""

    @property
    def max_episode_steps(self):
        return self.env.max_episode_steps

    def reset(self, **kwargs):
        observation, _ = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info.update(terminated=terminated, truncated=truncated)
        info["TimeLimit.truncated"] = truncated and not terminated
        return observation, reward, terminated or truncated, info


class TDMPC2TensorAdapter(gym.Wrapper):
    """Same data contract as upstream TensorWrapper, without learner imports."""

    @property
    def max_episode_steps(self):
        return self.env.max_episode_steps

    @property
    def no_control_action(self):
        return torch.from_numpy(self.unwrapped.no_control_action)

    def reset(self, task_idx=None, *, seed=None, options=None):
        if task_idx not in (None, 0):
            raise ValueError("MPCSC factory creates single-task environments")
        return torch.from_numpy(self.env.reset(seed=seed, options=options))

    def rand_act(self):
        return torch.from_numpy(self.action_space.sample().astype(np.float32))

    def step(self, action):
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        observation, reward, done, info = self.env.step(action)
        info["terminated"] = torch.tensor(float(info["terminated"]), dtype=torch.float32)
        return torch.from_numpy(observation), torch.tensor(reward, dtype=torch.float32), done, info


def make_gym_env(task="mpcsc-kuramoto", *, seed=None, params=None, random_init=None):
    return MPCSCEnv(load_config(task, params=params, random_init=random_init), seed=seed)


def _get(cfg, key, default=None):
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def _set(cfg, key, value):
    if isinstance(cfg, MutableMapping):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def make_env(cfg: Any):
    """Drop-in environment factory for an upstream single-task agent/trainer.

    Only environment metadata are filled in cfg. No policy is created or run.
    """
    if _get(cfg, "multitask", False):
        raise ValueError("multitask MPCSC environments are not part of this migration")
    if _get(cfg, "obs", "state") != "state":
        raise ValueError("MPCSC migration currently supports state observations only")
    snapshot = _get(cfg, 'mpcsc_resolved')
    task = str(_get(cfg, 'task', 'mpcsc-kuramoto'))
    if snapshot:
        resolved = from_snapshot(snapshot)
        overrides = _get(cfg, 'mpcsc_params') or {}
        if any(key not in resolved.params or resolved.params[key] != value for key, value in overrides.items()):
            raise ValueError('mpcsc_params conflict with the saved environment snapshot')
        identity = task_identity(task, overrides)
        if identity is None:
            if task != _get(cfg, 'mpcsc_task_source'):
                raise ValueError('path task cannot be verified from the saved environment snapshot; '
                                 'for a legacy path config, explicitly set task='
                                 + resolved_task_identity(resolved))
            identity = resolved_task_identity(resolved)
        if identity != resolved_task_identity(resolved):
            raise ValueError('task conflicts with the saved environment snapshot (including reward variant)')
        randomized = _get(cfg, 'mpcsc_random_init')
        if randomized is not None and randomized != resolved.random_init:
            raise ValueError('mpcsc_random_init conflicts with the saved environment snapshot')
    else:
        resolved = load_config(task,
                               params=_get(cfg, 'mpcsc_params'),
                               random_init=_get(cfg, 'mpcsc_random_init'))
    raw = MPCSCEnv(resolved, seed=_get(cfg, 'seed'))
    if not snapshot or not _get(cfg, 'mpcsc_task_source'):
        _set(cfg, 'mpcsc_task_source', task)
    _set(cfg, 'task', resolved_task_identity(resolved))
    _set(cfg, 'mpcsc_resolved', resolved.snapshot())
    env = TDMPC2TensorAdapter(TDMPC2NumpyAdapter(raw))
    _set(cfg, "obs_shape", {"state": env.observation_space.shape})
    _set(cfg, "action_dim", env.action_space.shape[0])
    _set(cfg, "episode_length", env.max_episode_steps)
    _set(cfg, "seed_steps", max(1000, 5 * env.max_episode_steps))
    # One resolver is shared by training, evaluation, and benchmark entry points.
    from netcc_rl.algorithms.proposed.config import resolve_gaussian_bounds, resolve_model_widths
    resolve_gaussian_bounds(cfg)
    resolve_model_widths(cfg, env.observation_space.shape)
    return env
