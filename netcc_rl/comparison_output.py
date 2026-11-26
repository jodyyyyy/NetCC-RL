"""Common result format for all eight comparison algorithms."""

import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev


def sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def identity(config):
    return {key: config[key] for key in ("algorithm", "environment", "protocol")}


def evaluation(config, episodes):
    values = [row["return"] for row in episodes]
    return dict(**identity(config), episodes=episodes, mean_return=mean(values),
                population_sd=pstdev(values), deterministic=True)


def training_row(config, real_steps, metrics, *, learner_updates=None, episodes=()):
    return dict(**identity(config), kind="train", real_steps=real_steps,
                learner_updates=learner_updates, episodes=list(episodes), metrics=metrics)


def checkpoint_index(config, output, weights, real_steps, seed):
    write_json(output / "checkpoint.json",
               dict(**identity(config), seed=seed, real_steps=real_steps,
                    file=weights.name, format="sb3-zip" if weights.suffix == ".zip" else "torch-pt",
                    sha256=sha256(weights)))


def runtime_record(config, seed, device, torch_version, budget, smoke):
    return dict(**identity(config), seed=seed, device=str(device), torch=torch_version,
                budget=budget, smoke=smoke)


def progress_record(config, phase, real_steps, budget, *, learner_updates=None,
                    gathered=0, wall_seconds=0.0, updated_at=None, error=None):
    return dict(**identity(config), phase=phase, real_steps=real_steps, gathered=gathered,
                learner_updates=learner_updates, budget=budget, wall_seconds=wall_seconds,
                updated_at=updated_at, error=error)
