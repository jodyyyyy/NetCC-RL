"""CPU-only, bounded SB3 integration smoke and deterministic checkpoint evaluation."""
import argparse
import time
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from netcc_rl.algorithms.baselines.environment import make_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from netcc_rl.algorithms.baselines.phase_features import phase_features, FEATURE_MULTIPLIER


class PhaseExtractor(BaseFeaturesExtractor):
    """Encode inside policy/Q networks, leaving SB3 replay observations raw."""
    def __init__(self, observation_space):
        super().__init__(observation_space, FEATURE_MULTIPLIER*observation_space.shape[0])

    def forward(self, observations):
        return phase_features(observations)
