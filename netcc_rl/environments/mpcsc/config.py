"""Load MPCSC-derived environment presets, using all nodes as default drivers."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import yaml

TASKS = ("mpcsc-kuramoto", "mpcsc-sis")
CONFIG_DIR = Path(__file__).with_name("configs")
VERSIONS = {"mpcsc-kuramoto": 1, "mpcsc-sis": 1}
KEYS = {
    "mpcsc-kuramoto": set(
        "net_type T node_num driver_node_num observe_node_num avg_degree coupling noise_std seed kuramoto_action_cost_coef".split()
    ),
    "mpcsc-sis": set(
        "net_type T node_num driver_node_num observe_node_num initial_I_node_num beta gamma noise_std seed sis_control_cost_coef".split()
    ),
}


def task_identity(task, params=None):
    """Resolve built-in names without reading a preset or a source YAML.

    Paths have no verifiable identity without the first resolution's provenance.
    SIS identity includes the effective reward variant, including explicit costs.
    """
    name = str(task).removeprefix("mpcsc-")
    aliases = {
        "kuramoto": "mpcsc-kuramoto",
        "sis": "mpcsc-sis",
        "sis-control-cost": "mpcsc-sis-control-cost",
        "sis_control_cost": "mpcsc-sis-control-cost",
        "kuramoto-control-cost": "mpcsc-kuramoto-control-cost",
        "kuramoto_control_cost": "mpcsc-kuramoto-control-cost",
    }
    identity = aliases.get(name)
    if identity in {"mpcsc-sis", "mpcsc-sis-control-cost"}:
        identity = (
            "mpcsc-sis-control-cost"
            if (params or {}).get("sis_control_cost_coef", 0.1) > 0
            else "mpcsc-sis"
        )
    if (
        identity in {"mpcsc-kuramoto", "mpcsc-kuramoto-control-cost"}
        and params
        and ("kuramoto_action_cost_coef" in params)
    ):
        identity = (
            "mpcsc-kuramoto-control-cost"
            if params["kuramoto_action_cost_coef"] > 0
            else "mpcsc-kuramoto"
        )
    return identity


def resolved_task_identity(resolved):
    if (
        resolved.task == "mpcsc-kuramoto"
        and resolved.params.get("kuramoto_action_cost_coef", 0.0) > 0
    ):
        return "mpcsc-kuramoto-control-cost"
    return (
        "mpcsc-sis-control-cost"
        if resolved.task == "mpcsc-sis" and resolved.params["sis_control_cost_coef"] > 0
        else resolved.task
    )


@dataclass(frozen=True)
class EnvironmentConfig:
    task: str
    params: dict[str, Any]
    random_init: bool
    max_episode_steps: int
    source_config: str
    fixed_actuators: dict[str, Any] | None = None

    def snapshot(self):
        constants = (
            {"dt": 0.01}
            if self.task == "mpcsc-kuramoto"
            else {"dt": 0.1, "avg_degree": 3.0} if self.task == "mpcsc-sis" else {}
        )
        sis_cost = self.params.get("sis_control_cost_coef", 0.1)
        kuramoto_cost = self.params.get("kuramoto_action_cost_coef", 0.0)
        reward = (
            "kuramoto_action_l2"
            if kuramoto_cost > 0
            else "sis_control_cost" if sis_cost > 0 else "original"
        )
        result = dict(
            snapshot_version=1,
            task=self.task,
            params=deepcopy(self.params),
            random_init=self.random_init,
            max_episode_steps=self.max_episode_steps,
            source_config=self.source_config,
            environment_version=VERSIONS[self.task],
            reward_variant=reward,
            dynamics_constants=constants,
        )
        if self.fixed_actuators is not None:
            result.update(
                snapshot_version=2,
                environment_version=VERSIONS[self.task] + 1,
                fixed_actuators=deepcopy(self.fixed_actuators),
            )
        signature = hashlib.sha256(
            json.dumps(result, sort_keys=True).encode()
        ).hexdigest()[:12]
        population = self.params["node_num"]
        drivers = self.params["driver_node_num"]
        result["environment_id"] = (
            f"{self.task}-{reward}-n{population}-d{drivers}-{signature}"
        )
        return result


def from_snapshot(snapshot):
    """Restore exact effective settings, including code/dynamics version checks."""
    if not isinstance(snapshot, dict) or snapshot.get("snapshot_version") not in (1, 2):
        raise ValueError("unsupported environment snapshot version")
    task = snapshot.get("task")
    version = snapshot["snapshot_version"]
    if (
        task not in TASKS
        or snapshot.get("environment_version") != VERSIONS[task] + version - 1
    ):
        raise ValueError("environment version differs")
    if (version == 2) != (snapshot["params"].get("actuator_mode") == "fixed_v2"):
        raise ValueError("snapshot version and actuator_mode differ")
    cfg = _validate(
        task,
        deepcopy(snapshot["params"]),
        snapshot["random_init"],
        snapshot["source_config"],
        fixed=deepcopy(snapshot.get("fixed_actuators")),
        restoring=version == 2,
    )
    if cfg.snapshot() != snapshot:
        raise ValueError("environment snapshot is inconsistent with effective settings")
    return cfg


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} is out of range")
    return int(value)


def load_config(
    task: str | Path = "mpcsc-kuramoto",
    *,
    params: Mapping[str, Any] | None = None,
    random_init: bool | None = None,
) -> EnvironmentConfig:
    """Accept a task name or a YAML path; reject unsupported parameter changes."""
    if params is not None and (not isinstance(params, Mapping)):
        raise ValueError("environment params must be a mapping or null")
    name = str(task).removeprefix("mpcsc-")
    name = "sis_control_cost" if name == "sis-control-cost" else name
    requested_kuramoto_cost = name in {"kuramoto-control-cost", "kuramoto_control_cost"}
    name = "kuramoto" if requested_kuramoto_cost else name
    path = (
        CONFIG_DIR / f"{name}.yaml"
        if name in {"kuramoto", "sis", "sis_control_cost"}
        else Path(task)
    )
    if task_identity(task) is None:
        raise ValueError("unsupported mainline MPCSC task")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or data.get("task") not in TASKS:
        raise ValueError("configuration must select an MPCSC task")
    values = deepcopy(data["params_to_env"])
    if data["task"] == "mpcsc-sis":
        values.setdefault("sis_control_cost_coef", 0.1)
    if params:
        population_key, driver_key = ("node_num", "driver_node_num")
        if (
            population_key in params
            and driver_key not in params
            and (values[driver_key] == values[population_key])
        ):
            values[driver_key] = params[population_key]
        if (
            population_key in params
            and "observe_node_num" in values
            and ("observe_node_num" not in params)
            and (values["observe_node_num"] == values[population_key])
        ):
            values["observe_node_num"] = params[population_key]
        values.update(params)
    if data["max_episode_steps"] != data["params_to_env"]["T"]:
        raise ValueError("max_episode_steps must equal params_to_env.T")
    randomized = data["random_init"] if random_init is None else random_init
    resolved = _validate(data["task"], values, randomized, data["source_config"])
    if (
        requested_kuramoto_cost
        and resolved.params.get("kuramoto_action_cost_coef", 0.0) <= 0
    ):
        raise ValueError(
            "kuramoto-control-cost requires a positive kuramoto_action_cost_coef"
        )
    return resolved


def _validate(task, values, randomized, source, *, fixed=None, restoring=False):
    from .fixed_actuators import CONTROL_KEYS, digest, prepare, validate_nodes

    optional = {"kuramoto_action_cost_coef"} if task == "mpcsc-kuramoto" else set()
    if set(values) - KEYS[task] - CONTROL_KEYS or KEYS[task] - set(values) - optional:
        raise ValueError("params_to_env has missing or unsupported keys")
    length = _positive_int(values["T"], "T")
    _positive_int(values["seed"], "environment seed", allow_zero=True)
    if not isinstance(randomized, bool):
        raise ValueError("random_init must be boolean")
    count = _positive_int(values["node_num"], "node_num")
    drivers = _positive_int(values["driver_node_num"], "driver_node_num")
    if values["net_type"] != "ER":
        raise ValueError("this migration supports the configured ER topology only")
    if _positive_int(values["observe_node_num"], "observe_node_num") != count:
        raise ValueError(
            "MPCSC returns the full state; partial observation is unsupported"
        )
    degree = values.get("avg_degree", 3.0)
    if count < 2 or not np.isfinite(degree) or (not 0 < degree <= count - 1):
        raise ValueError("ER expected degree must be in (0, node_num - 1]")
    if not np.isfinite(values["noise_std"]) or values["noise_std"] < 0:
        raise ValueError("noise_std must be non-negative and finite")
    if task == "mpcsc-sis":
        coef = values["sis_control_cost_coef"]
        if (
            isinstance(coef, bool)
            or not isinstance(coef, (int, float))
            or (not np.isfinite(coef))
            or (coef < 0)
        ):
            raise ValueError("sis_control_cost_coef must be nonnegative and finite")
        infected = _positive_int(
            values["initial_I_node_num"], "initial_I_node_num", allow_zero=True
        )
        if infected > count:
            raise ValueError("initial_I_node_num cannot exceed node_num")
        for key in ("beta", "gamma"):
            if not np.isfinite(values[key]) or values[key] < 0:
                raise ValueError(f"{key} must be non-negative and finite")
    elif task == "mpcsc-kuramoto":
        coef = values.get("kuramoto_action_cost_coef", 0.0)
        if type(coef) not in (int, float) or not np.isfinite(coef) or coef < 0:
            raise ValueError("kuramoto_action_cost_coef must be nonnegative and finite")
        if not np.isfinite(values["coupling"]):
            raise ValueError("coupling must be finite")
    if drivers > count:
        raise ValueError("driver count cannot exceed population size")
    mode = values.get("actuator_mode")
    if "actuator_mode" in values and mode != "fixed_v2":
        raise ValueError("actuator_mode must be fixed_v2 or omitted for legacy physics")
    if mode is None and set(values) & CONTROL_KEYS - {"actuator_mode"}:
        raise ValueError(
            "explicit driver_nodes/selection_seed require actuator_mode=fixed_v2"
        )
    if "driver_nodes" in values:
        validate_nodes(values["driver_nodes"], count, drivers)
    if "selection_seed" in values:
        _positive_int(values["selection_seed"], "selection_seed", allow_zero=True)
        raise ValueError("selection_seed is only used by Boids random fixed leaders")
    if mode == "fixed_v2":
        if restoring:
            if not isinstance(fixed, dict) or fixed.get("version") != 2:
                raise ValueError("missing fixed actuator snapshot")
            nodes = validate_nodes(fixed.get("driver_nodes"), count, drivers)
            if (
                fixed.get("driver_sha256") != digest(nodes)
                or fixed.get("physical_sha256") != digest(fixed.get("physical_system"))
                or fixed.get("physical_seed") != values["seed"]
                or ("driver_nodes" in values and values["driver_nodes"] != nodes)
            ):
                raise ValueError("inconsistent fixed actuator / physical snapshot")
        else:
            fixed, _ = prepare(task, values)
    return EnvironmentConfig(task, values, randomized, length, source, fixed)
