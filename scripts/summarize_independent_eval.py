"""Summarize 100 independent seeded episodes per checkpoint."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t

from analyze_exp017 import write_csv
from netcc_rl.matrix import ALL_ARMS, MATRIX

SEEDS = list(range(203000, 203100))


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set", choices=("historical", "all"), default="all")
    args = parser.parse_args()
    arms = MATRIX["arms"] if args.set == "historical" else ALL_ARMS
    summary, episodes, checks = [], [], {}
    for environment in MATRIX["environments"]:
        for arm in arms:
            run = f"{environment}-{arm}-seed0"
            path = args.evaluations / f"{run}.json"
            row = dict(run=run, environment=environment, arm=arm, training_seed=0, status="pending",
                       evaluation_episodes=0, mean=None, sample_sd=None, ci95_low=None, ci95_high=None)
            if path.is_file():
                result = json.loads(path.read_text())
                ep = result["episodes"]
                load = result["checkpoint_load"]
                if ([p["seed"] for p in ep] != SEEDS or result["eval_policy"] != "stochastic"
                        or (result["exp017_environment"], result["exp017_arm"], result["training_seed"]) != (environment, arm, 0)
                        or not all(load[key] for key in ("policy_restored", "world_model_restored", "critic_restored", "value_support_verified"))
                        or not all(math.isfinite(p["return"]) and p["length"] == 150 and not p["terminated"] for p in ep)):
                    raise ValueError(f"Evaluation protocol differs for {run}")
                values = [p["return"] for p in ep]
                center, sd = mean(values), stdev(values)
                if not math.isclose(center, result["episode_return"], abs_tol=1e-12):
                    raise ValueError(f"Reported mean differs for {run}")
                margin = t.ppf(.975, len(values)-1) * sd / math.sqrt(len(values))
                row.update(status="complete", evaluation_episodes=len(values), mean=center,
                           sample_sd=sd, ci95_low=center-margin, ci95_high=center+margin)
                episodes.extend(dict(run=run, seed=p["seed"], value=p["return"],
                                     steps=p["length"], terminated=int(p["terminated"])) for p in ep)
                checkpoint_hash = digest(args.run_root / run / "final.pt")
                if result["checkpoint_sha256"] != checkpoint_hash:
                    raise ValueError(f"Checkpoint hash differs for {run}")
                checks[run] = dict(checkpoint_sha256=checkpoint_hash,
                                   config_sha256=result["config_sha256"], evaluation_sha256=digest(path),
                                   architecture_sha256=result["exp017_identity_sha256"])
            summary.append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "results.csv", summary)
    write_csv(args.output / "episodes.csv", episodes)
    (args.output / "validation.json").write_text(json.dumps(checks, indent=2)+"\n")


if __name__ == "__main__":
    main()
