"""Task-specific network boundary transform; no change to simulator/replay."""
import numpy as np
import torch
from .environment import CASE,NODE_COUNT,RAW_NODE_DIM,RAW_DIM
FEATURE_MULTIPLIER=2 if CASE=='kuramoto' else 1
NODE_FEATURES=RAW_NODE_DIM*FEATURE_MULTIPLIER
INPUT_ENCODING=CASE+'_native_v1'
def phase_features(x):
    if CASE!='kuramoto':return x
    if isinstance(x,torch.Tensor):return torch.stack((x.sin(),x.cos()),-1).flatten(-2)
    x=np.asarray(x);return np.stack((np.sin(x),np.cos(x)),-1).reshape(*x.shape[:-1],2*x.shape[-1])
