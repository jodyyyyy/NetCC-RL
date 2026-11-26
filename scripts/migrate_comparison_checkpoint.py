"""Migrate a trusted historical comparison checkpoint for evaluation."""

import argparse
from pathlib import Path

import torch

from netcc_rl.cli import comparison_backend, read_config
from netcc_rl.comparison_output import checkpoint_index, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    algo = config["algorithm"]
    module = comparison_backend(config)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / ("final.zip" if algo in ("ppo", "sac", "td3") else "final.pt")
    seed = config["seed"]
    if destination.suffix == ".zip":
        from stable_baselines3 import PPO, SAC, TD3
        from netcc_rl.algorithms.baselines.sb3 import PhaseExtractor
        expected_profile = config["profile"]
        policy_kwargs = dict(net_arch=expected_profile["net_arch"], features_extractor_class=PhaseExtractor)
        agent = dict(ppo=PPO,sac=SAC,td3=TD3)[algo].load(
            args.checkpoint, device="cpu", custom_objects={"policy_kwargs":policy_kwargs})
        action_dim = len(config["drivers"])
        if agent.action_space.shape != (action_dim,):
            raise ValueError("historical checkpoint action shape differs")
        if agent.num_timesteps != config["steps"]:
            raise ValueError("historical checkpoint step budget differs")
        if agent.seed != seed:
            raise ValueError("historical checkpoint seed differs")
        agent.save(destination)
        steps = agent.num_timesteps
    else:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if algo == "cmat":
            adapter, learner = module.build(config, "cpu")
            identity = saved["identity"]
            if (identity["algorithm"] != algo or identity["adapter_identity"] != adapter.identity
                    or identity["seed"] != seed):
                raise ValueError("historical CMAT checkpoint protocol differs")
            steps = saved["completed"]
            if steps != config["steps"]:
                raise ValueError("a completed historical CMAT checkpoint is required")
            learner.restore(saved["learner"])
            saved = dict(config=config, learner=learner.training_state(), real_steps=steps)
        else:
            if (saved["algorithm"], saved["case"], saved["profile"]) != (algo, config["environment"], config["profile"]):
                raise ValueError("historical algorithm, environment or profile differs")
            steps = saved["env_steps"]
            if steps != config["steps"]:
                raise ValueError("historical checkpoint step budget differs")
        torch.save(saved, destination)
    write_json(args.output / "config.json", config)
    checkpoint_index(config, args.output, destination, steps, seed)


if __name__ == "__main__":
    main()
