"""Enumerate and validate the 60 formal configuration identities."""

import argparse
from itertools import product
import json
from pathlib import Path

from .cli import read_config
from .matrix import ALL_ARMS, MATRIX, resolved
from .algorithms.proposed.common.adaptive_architecture import digest

ALGORITHMS = ("ppo", "sac", "td3", "mappo", "happo", "mat", "jointppo", "cmat")


def entries(root, validate=False):
    root = Path(root)
    for env, arm in product(MATRIX["environments"], ALL_ARMS):
        path = root / "ablation" / env / f"{arm}.yaml"
        if read_config(path) != {"exp017_environment": env, "exp017_arm": arm}:
            raise ValueError(f"Incorrect configuration identity in {path}")
        row = dict(config=str(path), environment=env, algorithm="proposed", identity=arm,
                   factors=ALL_ARMS[arm], protocol=f"comparison-{env}-v1")
        if validate:
            cfg = resolved(env, arm, "cpu")
            row["architecture_sha256"] = digest(cfg.architecture_resolved)
        yield row
    for env, algo in product(MATRIX["environments"], ALGORITHMS):
        path = root / "comparison" / f"{env}-{algo}.yaml"
        cfg = read_config(path)
        if (cfg.get("environment"), cfg.get("algorithm")) != (env, algo):
            raise ValueError(f"Incorrect configuration identity in {path}")
        row = dict(config=str(path), environment=env, algorithm=algo, identity=algo,
                   protocol=cfg["protocol"])
        if validate:
            row["snapshot_sha256"] = cfg["snapshot_sha256"]
            row["profile_sha256"] = digest(cfg["profile"])
        yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    rows = list(entries(args.configs, args.validate))
    if len({(r["environment"], r["identity"]) for r in rows}) != 60:
        raise ValueError("Expected 60 unique configurations")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
