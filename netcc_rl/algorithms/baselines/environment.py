"""Native Gym API from the current main's frozen three-environment presets."""
import os
import numpy as np
import gymnasium as gym
from netcc_rl.algorithms.baselines.environment_profile import resolved
from netcc_rl.environments import make_env as factory
CASE=os.environ.get('NETCC_RL_BASELINE_TASK','kuramoto')
CFG=vars(resolved(CASE,device='cpu'))
SNAPSHOT=CFG.get('graph_resolved') or CFG['mpcsc_resolved']
DRIVERS=tuple(SNAPSHOT.get('driver_nodes') or SNAPSHOT['fixed_actuators']['driver_nodes'])
NODE_COUNT=38 if CASE=='swing' else 100
RAW_NODE_DIM=5 if CASE=='swing' else 1
RAW_DIM=NODE_COUNT*RAW_NODE_DIM

def make_env(path=None):
    from copy import deepcopy
    return factory(deepcopy(CFG)).unwrapped

def load_snapshot(path=None):return SNAPSHOT

class LocalGraph:
    def __init__(self,snapshot=None):
        self.drivers=DRIVERS;self.raw_dim=RAW_DIM
        self.nodes=tuple(tuple(range(i*RAW_NODE_DIM,(i+1)*RAW_NODE_DIM)) for i in DRIVERS)
        dim=2 if CASE=='kuramoto' else RAW_NODE_DIM
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(dim,),np.float32)
        self.action_space=gym.spaces.Box(-1,1,(1,),np.float32)
    def observe(self,obs):
        from .phase_features import phase_features
        return phase_features(np.asarray(obs,dtype=np.float32)[...,np.asarray(self.nodes)])
    def joint_action(self,a):
        a=np.asarray(a,dtype=np.float32)
        if a.shape!=(len(DRIVERS),1) or not np.isfinite(a).all():raise ValueError('ordered finite scalar actions required')
        return a[:,0].copy()
    def manifest(self):return dict(driver_nodes=DRIVERS,local_raw_indices=self.nodes,case=CASE,actor_scope='self_only')
