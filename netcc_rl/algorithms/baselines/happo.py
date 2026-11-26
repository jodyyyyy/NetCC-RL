"""MARLlib-derived HAPPO, common profile; bounded CPU runner in run.

Copyright (c) 2023 Replicable-MARL. MIT: licenses/MARLlib-HAPPO.txt.
Adapted: clipped surrogate/entropy/KL, shuffled all-policy sweeps, detached
likelihood-factor update, shared value links, max-of-two clipped value loss.
No legacy Ray imports. Task-specific dense state and continuous Gaussian logits.
"""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal,kl_divergence
from .batched import independent_actor_parameters,normal_sample
from .profiling import profiled
from .phase_features import phase_features, FEATURE_MULTIPLIER
from netcc_rl.algorithms.baselines.baseline_config import load_profile
from netcc_rl.algorithms.baselines.baseline_networks import ContinuousActor,network,gradient_step
from .neighborhoods import Neighborhoods,physical_graph
# Historical helper stays importable for the original tests, never used by v2.



def adaptive_kl(coefficient,measured,target):
    if measured>2*target:return coefficient*1.5
    if measured<0.5*target:return coefficient*0.5
    return coefficient


def sequential_advantage(modified,new_logp,old_logp):
    return (modified*(new_logp-old_logp).exp()).detach()


def clipped_value_loss(predicted,old,target,clip):
    clipped=old+(predicted-old).clamp(-clip,clip)
    return torch.maximum((predicted-target).square(),(clipped-target).square()).mean()


def actor_objective(dist,old_dist,actions,old_logp,modified,config,kl_coefficient):
    lp=dist.log_prob(actions).sum(-1)
    ratio=(lp-old_logp).exp()
    score=torch.minimum(modified*ratio,modified*ratio.clamp(1-config['clip_param'],1+config['clip_param'])).mean()
    entropy=dist.entropy().sum(-1).mean();kl=kl_divergence(old_dist,dist).sum(-1).mean()
    return -score-config['entropy_coeff']*entropy+kl_coefficient*kl,kl


class SharedValue(nn.Module):
    def __init__(self,state_dim,agents,config):
        super().__init__();self.agents=agents;self.include_actions=config['opp_action_in_cc']
        self.state_dim=state_dim
        self.net=network(state_dim+(agents-1 if self.include_actions else 0),config['hidden'],1,config['activation'])
        for layer in self.net:
            if isinstance(layer,nn.Linear):
                with torch.no_grad():
                    layer.weight.normal_();layer.weight.mul_(1/layer.weight.square().sum(1,keepdim=True).sqrt());layer.bias.zero_()
        nn.init.orthogonal_(self.net[-1].weight,gain=1.)

    def forward(self,states,actions=None,owner=0):
        if states.shape[-1]!=self.state_dim:raise ValueError('global state shape differs')
        if self.include_actions:
            if actions is None or actions.shape[-1]!=self.agents:raise ValueError('ordered joint actions required')
            others=[j for j in range(self.agents) if j!=owner]
            states=torch.cat([states,actions[...,others]],-1)
        return self.net(states).squeeze(-1)


class HAPPO(nn.Module):
    def __init__(self,config=None,device='cpu',adjacency=None,drivers=None):
        super().__init__();self.config=deepcopy(config or load_profile('happo'));self.device=torch.device(device)
        if adjacency is None:adjacency,drivers=physical_graph()
        self.drivers=tuple(drivers);self.n=len(drivers);self.pi=Neighborhoods(adjacency,drivers,0)
        from .environment import LocalGraph, RAW_DIM
        self.pi.nodes=LocalGraph().nodes
        self.actors=nn.ModuleList([ContinuousActor(FEATURE_MULTIPLIER*len(ids),self.config,'happo') for ids in self.pi.nodes])
        self.central_value=SharedValue(RAW_DIM*FEATURE_MULTIPLIER,self.n,self.config)
        # Identity links, not copies. Equivalent centralized modules across policies.
        self.value_links=[self.central_value for _ in drivers]
        self.to(device)
        self.actor_optimizers=[torch.optim.Adam(m.parameters(),lr=self.config['actor_lr']) for m in self.actors]
        self.value_optimizers=[torch.optim.Adam(self.central_value.parameters(),lr=self.config['critic_lr']) for _ in drivers]
        self.execution_backend='batched'
        self.kl_coefficients=[self.config['kl_coeff'] for _ in drivers];self.env_steps=0

    def distributions(self,states):
        if self.execution_backend=='reference':return [m.distribution(x) for m,x in zip(self.actors,self.pi.gather(states))]
        mean,std=independent_actor_parameters(self.actors,states,self.pi.nodes)
        Normal(mean,std)  # Validate the complete batch once.
        return [Normal(mean[...,i:i+1],std[...,i:i+1],validate_args=False) for i in range(self.n)]

    @torch.no_grad()
    def sample(self,states,deterministic=False):
        states=torch.as_tensor(states,dtype=torch.float32,device=self.device)
        if self.execution_backend!='reference':
            mean,std=independent_actor_parameters(self.actors,states,self.pi.nodes);dist=Normal(mean,std)
            action=mean if deterministic else torch.cat([normal_sample(mean[...,i:i+1],std[...,i:i+1]) for i in range(self.n)],-1)
            return action,dist.log_prob(action)
        ds=self.distributions(states);a=torch.cat([d.mean if deterministic else d.sample() for d in ds],-1)
        lp=torch.stack([d.log_prob(a[...,i:i+1]).sum(-1) for i,d in enumerate(ds)],-1)
        return a,lp

    def predict(self,states):return self.sample(states,True)[0].cpu().numpy()

    def value(self,states,actions=None,owner=0):return self.central_value(phase_features(states),actions,owner)

    def set_scheduled_lr(self,steps):
        c=self.config;fraction=min(max(steps/c['lr_schedule_steps'],0),1)
        lr=c['critic_lr']+fraction*(c['min_lr_schedule']-c['critic_lr'])
        for opt in self.value_optimizers:
            for group in opt.param_groups:group['lr']=lr
        # MARLlib cc_mlp.update_actor accepts computed lr but doesn't apply it.
        # Keep its persistent actor Adam lr, and record that source behavior.
        return lr

    @profiled('happo_sequential_update')
    def update(self,trajectories):
        c=self.config;self.set_scheduled_lr(self.env_steps)
        s=torch.cat([t['states'] for t in trajectories]).float().to(self.device)
        a=torch.cat([t['actions'] for t in trajectories]).float().to(self.device)
        old_lp=torch.cat([t['logp'] for t in trajectories]).float().to(self.device)
        with torch.no_grad():
            old_ds=[Normal(d.mean.clone(),d.stddev.clone()) for d in self.distributions(s)]
            targets=[];advantages=[];old_values=[]
            cached=[]
            if self.execution_backend=='batched' and not self.central_value.include_actions:
                # Shared critic and global rewards do not depend on owner. Cache
                # only frozen pre-update quantities; new sequential logp stays live.
                for t in trajectories:
                    st=t['states'].float().to(self.device);ns=t['next_states'].float().to(self.device)
                    cached.append((self.value(st),self.value(ns[-1])))
            for owner in range(self.n):
                if owner and cached and all(t['rewards'].ndim==1 for t in trajectories):
                    old_values.append(old_values[0]);targets.append(targets[0]);advantages.append(advantages[0]);continue
                vs=[];ads=[];rets=[]
                for ti,t in enumerate(trajectories):
                    states=t['states'].float().to(self.device);ns=t['next_states'].float().to(self.device)
                    acts=t['actions'].float().to(self.device)
                    vals=cached[ti][0] if cached else self.value(states,acts,owner)
                    # Bootstrap opponent actions must be generated at final state,
                    # not taken from the preceding transition or reset episode.
                    next_actions=None if cached else self.sample(ns[-1],True)[0]
                    tail=cached[ti][1] if cached else self.value(ns[-1],next_actions,owner)
                    done=t['terminated'].to(self.device).float()
                    if done.ndim>1:done=done.any(-1).float()
                    rw=t['rewards'].float().to(self.device)
                    if rw.ndim>1:rw=rw[:,owner]
                    # Recursive GAE with masks. No trajectory crosses a reset.
                    nxt=torch.cat([vals[1:],tail[None]])
                    delta=rw+c['gamma']*(1-done)*nxt-vals
                    ad=torch.zeros_like(rw);carry=torch.zeros_like(tail)
                    for j in range(len(rw)-1,-1,-1):
                        carry=delta[j]+c['gamma']*c['gae_lambda']*(1-done[j])*carry;ad[j]=carry
                    vs.append(vals);ads.append(ad);rets.append(ad+vals)
                old_values.append(torch.cat(vs));targets.append(torch.cat(rets));ad=torch.cat(ads)
                if c['normalize_advantages']:ad=(ad-ad.mean())/ad.std(unbiased=False).clamp_min(1e-4)
                advantages.append(ad)
        passes=self.n if c['owner_passes']=='num_agents' else c['owner_passes'];metrics=[]
        for epoch in range(c['num_sgd_iter']):
            # Actual upstream loss is invoked for each owner policy; each call
            # iterates over all linked policies, starting with owner's advantage.
            for owner in range(passes):
                ids=torch.randperm(len(s),device=self.device)
                for subset in ids.split(c['sgd_minibatch_size']):
                    modified=advantages[owner][subset].clone();order=np.random.permutation(self.n).tolist();agent_metrics=[]
                    gathered=self.pi.gather(s[subset]) if self.execution_backend=='batched' else None
                    for i in order:
                        dist=self._loss_distribution(i,gathered[i]) if gathered is not None else self.distributions_for(i,s[subset]);old=Normal(old_ds[i].mean[subset],old_ds[i].stddev[subset],validate_args=gathered is None)
                        loss,kl=actor_objective(dist,old,a[subset,i:i+1],old_lp[subset,i],modified,c,self.kl_coefficients[i])
                        gradient_step(self.actor_optimizers[i],loss,self.actors[i].parameters(),c['grad_clip'])
                        with torch.no_grad():
                            updated=self._loss_distribution(i,gathered[i]) if gathered is not None else self.distributions_for(i,s[subset])
                            new_lp=updated.log_prob(a[subset,i:i+1]).sum(-1)
                            factor=(new_lp-old_lp[subset,i]).exp();modified=sequential_advantage(modified,new_lp,old_lp[subset,i])
                            if not torch.isfinite(modified).all():raise FloatingPointError('invalid sequential advantage')
                        if gathered is None:agent_metrics.append(dict(agent=i,loss=float(loss.detach()),kl=float(kl.detach()),factor_mean=float(factor.mean())))
                        else:agent_metrics.append(torch.stack([loss.detach(),kl.detach(),factor.mean()]))
                    if gathered is not None:
                        packed=torch.stack(agent_metrics).cpu().tolist()
                        agent_metrics=[dict(agent=i,loss=row[0],kl=row[1],factor_mean=row[2]) for i,row in zip(order,packed)]
                    pred=self.value(s[subset],a[subset],owner)
                    vl=c['vf_loss_coeff']*clipped_value_loss(pred,old_values[owner][subset],targets[owner][subset],c['vf_clip_param'])
                    gradient_step(self.value_optimizers[owner],vl,self.central_value.parameters(),c['grad_clip'])
                    metrics.append(dict(epoch=epoch,owner=owner,order=order,actors=agent_metrics,value_loss=float(vl.detach()),
                        batch_samples=len(subset),epochs_configured=c['num_sgd_iter'],owners_configured=passes))
        with torch.no_grad():
            measured=[float(kl_divergence(old,self.distributions_for(i,s)).sum(-1).mean()) for i,old in enumerate(old_ds)]
        previous=list(self.kl_coefficients)
        self.kl_coefficients=[adaptive_kl(k,v,c['kl_target']) for k,v in zip(previous,measured)]
        if metrics:
            metrics[-1]['kl_monitor']=[dict(agent=i,kl=v,target_exceeded=v>c['kl_target'],
                coefficient_before=old,coefficient_after=new)
                for i,(v,old,new) in enumerate(zip(measured,previous,self.kl_coefficients))]
            metrics[-1]['optimizer_early_stopping']=False
        self.env_steps+=len(s)
        return metrics

    def _loss_distribution(self,i,inputs):
        # Recorded actions and old distributions are validated once. Invalid
        # live Gaussian parameters still create a nonfinite loss, rejected by
        # gradient_step before Adam; post-step sequential factors retain guards.
        mean,logstd=self.actors[i].net(inputs).split(1,-1)
        return Normal(mean,logstd.exp(),validate_args=False)

    def distributions_for(self,i,states):return self.actors[i].distribution(phase_features(states[...,list(self.pi.nodes[i])]))

    def training_state(self):
        return dict(weights=self.state_dict(),config=self.config,kl_coefficients=self.kl_coefficients,env_steps=self.env_steps,
            actor_optimizers=[o.state_dict() for o in self.actor_optimizers],value_optimizers=[o.state_dict() for o in self.value_optimizers])

    def restore_training_state(self,state):
        # Optimizer.load_state_dict may retain CPU tensor aliases; own the
        # restored state so a subsequent update cannot mutate its checkpoint.
        state=deepcopy(state)
        if state['config']!=self.config:raise ValueError('config differs')
        self.load_state_dict(state['weights']);self.kl_coefficients=list(state['kl_coefficients']);self.env_steps=state['env_steps']
        for opts,key in [(self.actor_optimizers,'actor_optimizers'),(self.value_optimizers,'value_optimizers')]:
            for opt,data in zip(opts,state[key]):opt.load_state_dict(data)
