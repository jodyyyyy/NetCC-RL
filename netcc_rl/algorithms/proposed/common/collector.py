"""Synchronous vector collector with exact partial-tail transition accounting.

The GPU backend batches registered environments on CUDA. The cpu_serial backend
uses the original single-process CPU adapters. Replay receives real contiguous segments, never transitions after a reset.
"""
from copy import deepcopy
import torch
from netcc_rl.environments import make_env
from netcc_rl.environments.graph.env import GraphDynamics,TASKS


class VectorCollector:
    def __init__(self,cfg,batch_size,*,backend='gpu'):
        if type(batch_size) is not int or batch_size<1:raise ValueError('collector batch must be positive')
        self.cfg,self.batch_size=cfg,batch_size
        self.sim=None;self.envs=[]
        if backend not in {'gpu','cpu_serial'}: raise ValueError('unknown collector backend')
        if backend=='gpu' and not str(cfg.device).startswith('cuda'): raise ValueError('GPU collector requires CUDA')
        if backend=='gpu' and cfg.task in TASKS:
            self.sim=GraphDynamics(cfg.task,snapshot=cfg.graph_resolved,device=cfg.device,batch_size=batch_size,seed=cfg.seed)
            self.obs=self.sim.observe()
        elif backend=='gpu':
            from netcc_rl.environments.mpcsc.batched import BatchedMPCSC
            self.sim=BatchedMPCSC(cfg.mpcsc_resolved,batch_size,device=cfg.device,seed=cfg.seed)
            self.obs=self.sim.observe()
        else:
            for slot in range(batch_size):
                c=deepcopy(cfg);c.seed=cfg.seed+slot
                self.envs.append(make_env(c))
            self.obs=torch.stack([e.reset(seed=cfg.seed+i) for i,e in enumerate(self.envs)])
        self.collected=0

    def step(self,actions,count=None):
        count=self.batch_size if count is None else count
        if type(count) is not int or not 1<=count<=self.batch_size:raise ValueError('invalid actual tail count')
        before=self.obs[:count].clone()
        if self.sim:
            mask=torch.arange(self.batch_size,device=self.sim.device)<count
            obs,r,t,tr,info=self.sim.step(actions,active=mask)
            after=obs[:count].clone();reward=r[:count];terminated=t[:count];truncated=tr[:count]
            done=(t|tr)&mask
            if done.any():
                if self.cfg.task in TASKS:
                    self.sim.reset(mask=done)
                else:
                    self.sim.reset(done)
            self.obs=self.sim.observe()
        else:
            after=[];reward=[];terminated=[];truncated=[]
            for i in range(count):
                obs,r,done,info=self.envs[i].step(actions[i])
                after.append(obs);reward.append(r);terminated.append(info['terminated']);truncated.append(done and not info['terminated'])
                self.obs[i]=self.envs[i].reset() if done else obs
            after=torch.stack(after);reward=torch.stack(reward);terminated=torch.stack(terminated);truncated=torch.tensor(truncated)
        self.collected+=count
        return dict(before=before,after=after,action=actions[:count].detach(),reward=reward,terminated=terminated,truncated=truncated,
                    newly_collected_transitions=count)

    def close(self):
        for env in self.envs:env.close()
