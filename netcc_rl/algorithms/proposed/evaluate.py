"""Evaluate saved model weights with an explicitly seeded actor policy."""

from copy import deepcopy
import json

from netcc_rl.environments import make_env

from . import TDMPC2
from .config import load_config, parse_args
from .runner import evaluate, set_seed, write_json


def main():
    args = parse_args(__doc__, evaluation=True)
    # A saved resolved config is required, avoiding silently changed architectures.
    config_path = args.config or args.checkpoint.parent / 'config.yaml'
    cfg = load_config(config_path, args.overrides)
    set_seed(cfg.seed, cfg.torch_threads)
    saved_dims = (getattr(cfg, 'action_dim', None), getattr(cfg, 'obs_shape', None))
    env = make_env(cfg)
    try:
        # Confirm task dimensions against the checkpoint's saved metadata.
        if saved_dims[0] is not None and (saved_dims[0] != cfg.action_dim or
                tuple(saved_dims[1]['state']) != env.observation_space.shape):
            raise ValueError('task dimensions differ from the saved training configuration')
        agent = TDMPC2(cfg)
        report = agent.load(args.checkpoint)
        result = {**evaluate(agent, cfg), 'checkpoint_load': report}
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
    finally:
        env.close()


if __name__ == '__main__':
    main()
