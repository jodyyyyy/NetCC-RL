"""Shared CLI configuration and runtime selection."""

import argparse
from pathlib import Path

import yaml


def parser(description):
    command = argparse.ArgumentParser(description=description)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--device", help="Runtime override such as cpu or cuda:0")
    command.add_argument("--seed", type=int, help="Override the configuration's training seed")
    return command


def read_config(path):
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Expected a YAML configuration mapping")
    if config.get("kind") == "comparison":
        fields = {"schema_version", "kind", "algorithm", "environment", "protocol", "seed",
                  "device", "steps", "collector_batch_size", "action_bounds", "action_semantics",
                  "drivers", "snapshot_sha256", "profile", "implementation"}
        if set(config) != fields or config["schema_version"] != 1:
            raise ValueError("Comparison config must use schema version 1")
        if config.get("algorithm") not in {"ppo", "sac", "td3", "mappo", "happo", "mat", "jointppo", "cmat"}:
            raise ValueError("Unknown comparison algorithm")
        if config.get("environment") not in {"kuramoto", "sis", "swing"} or config.get("steps") != 1_000_000:
            raise ValueError("Invalid comparison environment or formal budget")
    return config


def comparison_backend(config):
    import os
    os.environ["NETCC_RL_BASELINE_TASK"] = config["environment"]
    if config["algorithm"] == "cmat":
        from .algorithms.baselines.cmat import run
    else:
        from .algorithms.baselines import run
    return run
