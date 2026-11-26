"""Train one NetCC-RL configuration."""

import os
from pathlib import Path

from .cli import comparison_backend, parser, read_config


def main():
    command = parser(__doc__)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--smoke", action="store_true", help="Run a bounded real-update smoke test")
    command.add_argument('--eager', action='store_true', help='Disable the proposed learner compiler and graphs')
    command.add_argument('--resume', action='store_true', help='Resume a CMAT run from last-good.pt')
    args = command.parse_args()
    config = read_config(args.config)
    if args.output.exists() and not args.resume:
        command.error("output must be a new directory")
    if args.resume and (config.get('algorithm') != 'cmat' or args.smoke):
        command.error('resume requires a formal CMAT run')
    if config.get("kind") == "comparison":
        device = args.device or ("cpu" if args.smoke else config["device"])
        if config["algorithm"] == "cmat":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        kwargs = {"resume": args.resume} if config["algorithm"] == "cmat" else {}
        comparison_backend(config).train(config, args.output, device,
                                         config["seed"] if args.seed is None else args.seed,
                                         args.smoke, **kwargs)
        return

    from .algorithms.proposed import NetCCAgent
    from .algorithms.proposed.config import load_config
    from .algorithms.proposed.runner import set_seed, train
    from .environments import make_env

    overrides = [f"work_dir={args.output.resolve()}"]
    if args.device is not None or args.smoke:
        overrides.append(f"device={args.device or 'cpu'}")
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    if args.eager or args.smoke:
        overrides += ['compile=false', 'learner_cuda_graphs=false']
    if args.smoke:
        overrides += ["steps=152", "warmup_steps=150",
                      "batch_size=8", "eval_episodes=1", "eval_freq=152", "save_freq=152", "torch_threads=1"]
    cfg = load_config(args.config, overrides)
    set_seed(cfg.seed, cfg.torch_threads)
    env = make_env(cfg)
    try:
        train(NetCCAgent(cfg), env, cfg)
    finally:
        env.close()


if __name__ == "__main__":
    main()
