"""Network-boundary Kuramoto encoding; replay and simulator states stay raw."""
import numpy as np
import torch

INPUT_ENCODING = 'kuramoto_node_major_sin_cos_v1_self_only_standard_marl'


def phase_features(states):
    """Node-major [..., theta_i] -> [..., sin(theta_i), cos(theta_i)]."""
    if isinstance(states, torch.Tensor):
        return torch.stack((states.sin(), states.cos()), -1).flatten(-2)
    states = np.asarray(states)
    return np.stack((np.sin(states), np.cos(states)), -1).reshape(*states.shape[:-1], 2*states.shape[-1])
