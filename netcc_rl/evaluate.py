"""Evaluate trusted model weights using an explicit NetCC-RL configuration."""

import json
from pathlib import Path

from .cli import comparison_backend, parser, read_config
from .comparison_output import sha256


def main():
    command = parser(__doc__)
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--output", type=Path)
    command.add_argument("--episodes", type=int)
    command.add_argument("--eval-seed", type=int, default=202600)
    args = command.parse_args()
    if args.episodes is not None and args.episodes < 1:
        command.error("episodes must be positive")
    config = read_config(args.config)
    if args.checkpoint.suffix == ".json":
        index = json.loads(args.checkpoint.read_text())
        filename = Path(index["file"])
        if filename.name != str(filename):
            command.error("checkpoint index must reference a file in the same directory")
        args.checkpoint = args.checkpoint.parent / filename
        if sha256(args.checkpoint) != index["sha256"]:
            command.error("checkpoint digest differs from the index")
    if config.get("kind") == "comparison":
        backend = comparison_backend(config)
        episodes = args.episodes or 50
        if config["algorithm"] == "cmat":
            import torch
            torch.set_num_threads(1)
            result = backend.evaluate_checkpoint(config, args.checkpoint, args.device or "cpu",
                                                 episodes, args.eval_seed)
        else:
            import torch
            torch.set_num_threads(1)
            agent = backend.load_runner(config, args.checkpoint, args.device or "cpu",
                                        config["seed"] if args.seed is None else args.seed)
            try:
                result = backend.evaluation(agent, config, episodes=episodes, seed_start=args.eval_seed)
            finally:
                agent.env.close()
    else:
        from .algorithms.proposed import TDMPC2
        from .algorithms.proposed.config import load_config
        from .algorithms.proposed.runner import evaluate, set_seed
        from .environments import make_env

        overrides = [f"device={args.device or 'cpu'}", "compile=false", "learner_cuda_graphs=false",
                     f"eval_seed={args.eval_seed}"]
        if args.episodes is not None:
            overrides.append(f"eval_episodes={args.episodes}")
        if args.seed is not None:
            overrides.append(f"seed={args.seed}")
        cfg = load_config(args.config, overrides)
        set_seed(cfg.seed, cfg.torch_threads)
        env = make_env(cfg)
        try:
            agent = TDMPC2(cfg)
            load_report = agent.load(args.checkpoint)
            result = {**evaluate(agent, cfg), "checkpoint_load": load_report}
        finally:
            env.close()
    result.update(checkpoint_sha256=sha256(args.checkpoint), config_sha256=sha256(args.config))
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
