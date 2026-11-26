"""Export proposed or comparison JSONL metrics to TensorBoard scalars."""

import argparse
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with args.input.open() as stream, SummaryWriter(args.output) as writer:
        for line in stream:
            row = json.loads(line)
            step = next((row[k] for k in ("step", "env_steps", "real_steps") if k in row), None)
            if step is None:
                raise ValueError("Row lacks a transition count")
            kind = row.get("kind") or ("eval" if "mean_return" in row else "train")
            metrics = dict(row.get("metrics", {}))
            for key in ("episode_return", "mean_return", "updates", "learner_updates"):
                if key in row:
                    metrics[key] = row[key]
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"{kind}/{key}", value, step)


if __name__ == "__main__":
    main()
