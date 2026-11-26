"""Train the actor-only agent: python -m netcc_rl.algorithms.proposed.train key=value ..."""

from pathlib import Path

from netcc_rl.environments import make_env

from . import NetCCAgent
from .config import experiment_name, load_config, parse_args, validate_experiment_output
from .runner import set_seed, train


def main():
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.overrides)
    validate_experiment_output(cfg)
    set_seed(cfg.seed, cfg.torch_threads)
    env = make_env(cfg)
    try:
        if cfg.work_dir is None:
            cfg.work_dir = str(Path(__file__).resolve().parents[1] / 'runs' /
                               cfg.task / experiment_name(cfg) / f'seed_{cfg.seed}')
        agent = NetCCAgent(cfg)
        train(agent, env, cfg)
    finally:
        env.close()


if __name__ == '__main__':
    main()
