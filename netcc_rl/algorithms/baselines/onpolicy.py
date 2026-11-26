"""Official marlbenchmark MAPPO learner behind an isolated CPU runner."""
import argparse
import json
import time
import numpy as np
from .phase_features import phase_features
import torch
from .environment import make_env, LocalGraph



def build(algorithm, device='cpu', *, formal=False):
    if algorithm!='mappo':raise ValueError('MAPPO adapter only')
    from argparse import Namespace
    from pathlib import Path

    from netcc_rl.algorithms.baselines.vendor.onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
    from netcc_rl.algorithms.baselines.vendor.onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
    from netcc_rl.algorithms.baselines.vendor.onpolicy.utils.separated_buffer import SeparatedReplayBuffer
    from gymnasium.spaces import Box
    if algorithm!='mappo':raise ValueError('MAPPO adapter only')
    a=Namespace(**json.loads((Path(__file__).with_name('vendor')/'onpolicy/defaults.json').read_text()))
    a.algorithm_name='mappo'
    # Explicit integration overrides. Retain upstream optimizer/loss defaults.
    a.use_recurrent_policy=False; a.use_naive_recurrent_policy=False
    a.hidden_size=64; a.episode_length=150; a.n_rollout_threads=1
    a.ppo_epoch=2; a.num_mini_batch=1; a.use_valuenorm=False; a.use_popart=False
    a.use_centralized_V=True; a.gamma=1-1/150
    a.use_proper_time_limits=False
    if formal:a.ppo_epoch=15;a.gamma=.99
    graph=LocalGraph()
    central=Box(-np.inf,np.inf,(len(phase_features(__import__('numpy').zeros(graph.raw_dim))),),dtype=np.float32) if a.use_centralized_V else graph.observation_space
    policies=[R_MAPPOPolicy(a,graph.observation_space,central,graph.action_space,device=torch.device(device)) for _ in graph.drivers]
    trainers=[R_MAPPO(a,p,device=torch.device(device)) for p in policies]
    buffers=[SeparatedReplayBuffer(a,graph.observation_space,central,graph.action_space) for _ in policies]
    return a,graph,policies,trainers,buffers
