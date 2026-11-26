"""Export comparison metrics from a frozen version-zero training run."""

import argparse
import gzip
import json
from pathlib import Path

from netcc_rl.cli import read_config
from netcc_rl.comparison_output import append_json, checkpoint_index, evaluation, sha256, training_row, write_json


def episodes(rows):
    for row in rows:
        yield dict(seed=row["seed"], **{"return":row["return_"]},
                   transitions=row["transitions"], terminated=False,
                   truncated=row["transitions"] == 150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    saved = json.loads((args.run / "config.json").read_text())
    implementation = config["implementation"]
    normalized = dict(saved)
    normalized.update(schema_version=1, action_semantics=config["action_semantics"],
                      implementation={key:normalized.pop(key) for key in implementation},
                      snapshot_sha256=config["snapshot_sha256"])
    if normalized != config:
        raise ValueError("Run and target configuration differ")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/"config.json",config)
    raw_metrics = args.run/"learner-metrics.jsonl.gz"
    if raw_metrics.exists():
        with gzip.open(raw_metrics, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                append_json(args.output/"metrics.jsonl",
                            training_row(config, row["real_steps"], row["metrics"],
                                         learner_updates=row["optimizer_steps"]))
    raw_eval = args.run/"evaluation.jsonl"
    if raw_eval.exists():
        for line in raw_eval.read_text().splitlines():
            row = json.loads(line)
            data = evaluation(config, list(episodes(row["episodes"])))
            append_json(args.output/"evaluation.jsonl",dict(real_steps=row["real_steps"],final=row["final"],**data))
            if row["final"]:
                write_json(args.output/"evaluation.json",data)
    if (args.run/"final.pt").is_file():
        import shutil
        source = args.run/"final.pt"
        destination = args.output/"final.pt"
        shutil.copyfile(source,destination)
        checkpoint_index(config,args.output,destination,config["steps"],config["seed"])
        if sha256(source)!=sha256(destination):
            raise ValueError("Checkpoint copied with different contents")


if __name__ == "__main__":
    main()
