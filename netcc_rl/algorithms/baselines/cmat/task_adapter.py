"""Continuous task boundary for the three historical CMAT profiles."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import netcc_rl
from netcc_rl.algorithms.proposed.models.architecture import digest as snapshot_digest
from netcc_rl.algorithms.proposed.utils.math import affine_action, squashed_gaussian_density
from netcc_rl.algorithms.proposed.models.node_features import node_features
from netcc_rl.environments.graph.env import GraphEnv
from netcc_rl.algorithms.baselines.environment_profile import resolved
from netcc_rl.environments.mpcsc.config import from_snapshot
from netcc_rl.environments.mpcsc.env import MPCSCEnv


class TaskAdapter:
    def __init__(self, config):
        self.name = config["environment"]
        if config["protocol"] == "comparison-sis-v1":
            self.snapshot = resolved("sis", device="cpu").mpcsc_resolved
        else:
            assets = Path(netcc_rl.__file__).resolve().parent / "environments/graph/data/cmat"
            self.snapshot = json.loads((assets / f"{self.name}.json").read_text())
        identity = config["implementation"]["adapter_identity"]
        if identity.get("version") != "continuous_task_adapter_v1":
            raise ValueError("Unsupported CMAT task adapter identity")
        digest = hashlib.sha256(json.dumps(self.snapshot, sort_keys=True).encode()).hexdigest()
        if digest != identity["snapshot_sha256"] or snapshot_digest(self.snapshot) != config["snapshot_sha256"]:
            raise ValueError("CMAT snapshot differs from the frozen profile")
        self.controls = self.snapshot.get("driver_nodes") or self.snapshot["fixed_actuators"]["driver_nodes"]
        if self.controls != identity["controls"] or self.controls != config["drivers"]:
            raise ValueError("CMAT driver order differs from the frozen profile")
        self.identity = identity
        self.nodes, self.raw_width = identity["raw_shape"]
        self.features = identity["encoded_shape"][1]
        self.action_dim = self.agents = len(self.controls)
        self.low, self.high = config["action_bounds"]
        self.discrete = False
        self.channels = self.group_width = 1
        self.cfg = SimpleNamespace(obs_shape={"state": (self.nodes * self.raw_width,)},
                                   graph_resolved=self.snapshot if self.name == "swing" else None,
                                   mpcsc_resolved=self.snapshot if self.name != "swing" else None)

    def features_tensor(self, raw):
        return node_features(raw, self.cfg)

    def policy_to_canonical(self, action):
        return affine_action(action.tanh(), self.low, self.high)

    def grouped(self, action):
        return action.unsqueeze(-1)

    def flatten(self, grouped):
        return grouped.squeeze(-1).contiguous()

    def mask(self, device):
        return torch.ones((self.agents, 1), dtype=torch.bool, device=device)

    def make_env(self):
        if self.name == "swing":
            return GraphEnv("graph-swing", snapshot=self.snapshot)
        return MPCSCEnv(from_snapshot(self.snapshot))


class GaussianGroup:
    """Tanh-affine Normal using the unique A10 density helper; event is a group."""
    def __init__(self,mean,log_std,low,high,mask=None):
        self.loc=mean;self.log_std=log_std;self.low=low;self.high=high
        self.mask=torch.ones_like(mean,dtype=torch.bool) if mask is None else mask
    @property
    def mean(self):return self.loc
    def rsample(self):
        return self.loc+self.log_std.exp()*torch.randn_like(self.loc)
    def sample(self):
        with torch.no_grad():return self.rsample()
    def log_prob(self,action):
        # Policy replay retains pre-tanh samples, avoiding saturated inverse-tanh.
        eps=(action-self.loc)/self.log_std.exp()
        return squashed_gaussian_density(self.loc.unsqueeze(-1),self.log_std.unsqueeze(-1),eps.unsqueeze(-1),self.low,self.high)[1].squeeze(-1)*self.mask
    def entropy(self):return -self.log_prob(self.rsample())
    def detached(self,index=slice(None)):
        return GaussianGroup(self.loc[index].detach().clone(),self.log_std[index].detach().clone(),self.low,self.high,self.mask.expand_as(self.loc)[index])
    def kl(self,other):
        return torch.distributions.kl_divergence(Normal(self.loc,self.log_std.exp()),Normal(other.loc,other.log_std.exp()))*self.mask
