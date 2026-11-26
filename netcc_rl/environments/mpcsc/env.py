"""Gymnasium environment exposing MPCSC dynamics in normalized action space."""

from typing import Any
import gymnasium as gym
import numpy as np
from .config import EnvironmentConfig


class MPCSCEnv(gym.Env):
    metadata = {"render_modes": []}
    render_mode = None

    def __init__(self, config: EnvironmentConfig, *, seed: int | None = None):
        super().__init__()
        self.config = config
        self.task = config.task
        if self.task not in ("mpcsc-kuramoto", "mpcsc-sis"):
            raise ValueError("unsupported mainline MPCSC task")
        if config.fixed_actuators is not None:
            from .fixed_actuators import build_backend

            self.backend = build_backend(
                self.task, config.params, config.fixed_actuators
            )
        elif self.task == "mpcsc-kuramoto":
            from .dynamics.kuramoto import KuramotoDynamics

            self.backend = KuramotoDynamics(
                **{
                    k: v
                    for k, v in config.params.items()
                    if k != "kuramoto_action_cost_coef"
                }
            )
        elif self.task == "mpcsc-sis":
            from .dynamics.sis import SISDynamics

            self.backend = SISDynamics(
                **{
                    k: v
                    for k, v in config.params.items()
                    if k != "sis_control_cost_coef"
                }
            )
        else:
            raise ValueError("unsupported mainline MPCSC task")
        p = config.params
        self.obs_dim = p["node_num"]
        self.action_dim = p["driver_node_num"]
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, (self.action_dim,), dtype=np.float32
        )
        if self.task == "mpcsc-sis":
            self.observation_space = gym.spaces.Box(
                0.0, 1.0, (self.obs_dim,), dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, (self.obs_dim,), dtype=np.float32
            )
        self._pending_seed = seed
        self._elapsed_steps = 0
        self._needs_reset = True
        self._first_sync_threshold_step = None
        self.action_space.seed(p["seed"] if seed is None else seed)

    @property
    def max_episode_steps(self) -> int:
        return self.config.max_episode_steps

    @property
    def driver_nodes(self) -> tuple[int, ...]:
        nodes = self.backend.driver_node_list
        return tuple(nodes)

    @property
    def no_control_action(self) -> np.ndarray:
        """SIS needs -1 in normalized coordinates to express zero intervention."""
        value = -1.0 if self.task == "mpcsc-sis" else 0.0
        return np.full(self.action_dim, value, dtype=np.float32)

    def _observation(self, observation: Any) -> np.ndarray:
        result = np.asarray(observation, dtype=np.float32).copy()
        if result.shape != (self.obs_dim,) or not np.isfinite(result).all():
            raise FloatingPointError(f"invalid {self.task} observation")
        return result

    def _info(self) -> dict[str, Any]:
        result = {
            "task": self.task,
            "elapsed_steps": self._elapsed_steps,
            "driver_nodes": self.driver_nodes,
            "environment_seed": self.config.params["seed"],
            "success": 0.0,
            "success_defined": False,
        }
        if self.config.fixed_actuators is not None:
            fixed = self.config.fixed_actuators
            result.update(
                actuator_mode="fixed_v2",
                driver_sha256=fixed["driver_sha256"],
                physical_sha256=fixed["physical_sha256"],
                population=self.config.params["node_num"],
                driver_count=self.action_dim,
            )
        return result

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        effective_seed = seed if seed is not None else self._pending_seed
        super().reset(seed=effective_seed)
        if effective_seed is not None:
            self.backend.reseed(effective_seed)
            self.action_space.seed(effective_seed)
        self._pending_seed = None
        options = {} if options is None else options
        if set(options) - {"random_init"}:
            raise ValueError("reset options only support random_init")
        random_init = options.get("random_init", self.config.random_init)
        if not isinstance(random_init, bool):
            raise ValueError("random_init must be boolean")
        observation = self.backend.reset(random_init=random_init)
        self._elapsed_steps = 0
        self._needs_reset = False
        self._first_sync_threshold_step = None
        info = self._info()
        info["random_init"] = random_init
        return (self._observation(observation), info)

    def to_native_action(self, action: Any) -> np.ndarray:
        normalized = np.asarray(action, dtype=np.float32)
        if normalized.shape != (self.action_dim,):
            raise ValueError(
                f"action must have shape {(self.action_dim,)}, got {normalized.shape}"
            )
        if not np.isfinite(normalized).all():
            raise ValueError("action must be finite")
        normalized = normalized.clip(-1, 1)
        return (normalized + 1) / 2 if self.task == "mpcsc-sis" else normalized

    def step(self, action: Any):
        if self._needs_reset:
            raise RuntimeError(
                "reset() is required before stepping or after truncation"
            )
        native = self.to_native_action(action)
        observation, reward, _, _ = self.backend.step(native)
        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self.max_episode_steps
        self._needs_reset = truncated
        reward = float(reward)
        if not np.isfinite(reward):
            raise FloatingPointError(f"invalid {self.task} reward")
        info = self._info()
        info.update(terminated=False, truncated=truncated, native_action=native.copy())
        if self.task == "mpcsc-kuramoto":
            action_energy = float(np.square(native.astype(np.float64)).mean())
            cost = (
                float(self.config.params.get("kuramoto_action_cost_coef", 0.0))
                * action_energy
            )
            info.update(
                synchronization_reward=reward,
                control_cost=cost,
                mean_squared_action=action_energy,
                sum_squared_action=action_energy * self.action_dim,
                total_reward=reward - cost,
            )
            reward -= cost
        if self.task == "mpcsc-sis":
            mean_control = float(native.mean())
            cost = self.config.params.get("sis_control_cost_coef", 0.0) * mean_control
            info.update(
                infection_reward=reward,
                control_cost=cost,
                mean_control_intensity=mean_control,
                total_reward=reward - cost,
            )
            reward -= cost
        if self.config.fixed_actuators is not None:
            energy = np.square(native.astype(np.float64))
            gain = 20.0 if self.task == "mpcsc-kuramoto" else 1.0
            info.update(
                mean_squared_action=float(energy.mean()),
                sum_squared_action=float(energy.sum()),
                mean_native_control=float(native.mean()),
                mean_absolute_native_control=float(np.abs(native).mean()),
                physical_control=(gain * native).copy(),
                physical_control_gain=gain,
                physical_control_units=(
                    "phase_velocity"
                    if self.task == "mpcsc-kuramoto"
                    else "infection_rate_reduction"
                ),
            )
            if self.task == "mpcsc-sis":
                info["infection_fraction"] = float(self.backend.I.mean())
            else:
                order = info.get("synchronization_reward", reward)
                if order >= 0.95 and self._first_sync_threshold_step is None:
                    self._first_sync_threshold_step = self._elapsed_steps
                info.update(
                    order_parameter=float(order),
                    synchronization_threshold=0.95,
                    first_sync_threshold_step=self._first_sync_threshold_step,
                )
        return (self._observation(observation), reward, False, truncated, info)

    def close(self):
        self._needs_reset = True
