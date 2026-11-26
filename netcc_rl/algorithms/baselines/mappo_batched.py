"""Independent MAPPO tensor batching for the frozen feedforward profile.

Original policies, per-agent Adam states, normalization, 15 epochs and every
optimizer call stay separate. Only independent agents' forward/backward graphs
are evaluated together. Their optimizer order commutes across agents; this
specialization must never be used for HAPPO's sequential likelihood factors.
"""
import numpy as np
import torch
from torch.distributions import Normal
from netcc_rl.algorithms.baselines.batched import onpolicy_parameters,upstream_bases,independent_linear


def train_independent(trainers,buffers):
    from netcc_rl.algorithms.baselines.vendor.onpolicy.utils.util import huber_loss,mse_loss
    policies=[t.policy for t in trainers];first=trainers[0];agents=len(policies)
    flags=('_use_recurrent_policy','_use_naive_recurrent','_use_popart','_use_valuenorm')
    if any(any(getattr(t,k) for k in flags) or t.num_mini_batch!=1 for t in trainers):
        raise ValueError('frozen feedforward non-normalized single-minibatch profile required')
    settings=('ppo_epoch','clip_param','entropy_coef','value_loss_coef','max_grad_norm','huber_delta',
        '_use_huber_loss','_use_clipped_value_loss','_use_policy_active_masks','_use_value_active_masks','_use_max_grad_norm')
    if any(any(getattr(t,k)!=getattr(first,k) for k in settings) for t in trainers):raise ValueError('agent profiles differ')
    if not first._use_max_grad_norm:raise ValueError('frozen profile requires gradient clipping')
    if any(b.available_actions is not None or b.factor is not None for b in buffers):raise ValueError('continuous scalar actions without likelihood factors required')
    device=policies[0].device;size=buffers[0].actions.shape[0]*buffers[0].actions.shape[1]
    if any(b.actions.shape[-1]!=1 or b.actions.shape[0]*b.actions.shape[1]!=size for b in buffers):raise ValueError('matching scalar-action batch required')
    advantages=[]
    for b in buffers:
        adv=b.returns[:-1]-b.value_preds[:-1];valid=adv.copy();valid[b.active_masks[:-1]==0]=np.nan
        advantages.append((adv-np.nanmean(valid))/(np.nanstd(valid)+1e-5))
    def field(name,tail=False):
        arrays=[getattr(b,name)[:-1] if tail else getattr(b,name) for b in buffers]
        return torch.as_tensor(np.stack([x.reshape(size,-1) for x in arrays]),device=device)
    fixed=dict(local=field('obs',True),central=field('share_obs',True),actions=field('actions'),
        old=field('action_log_probs'),old_value=field('value_preds',True),returns=field('returns',True),
        active=field('active_masks',True),adv=torch.as_tensor(np.stack([x.reshape(size,1) for x in advantages]),device=device))
    # Source draws permutations agent-major, then epoch-major. No network or
    # optimizer in this profile consumes randomness, so draw ahead in that order.
    perms=torch.stack([torch.stack([torch.randperm(size) for _ in range(t.ppo_epoch)]) for t in trainers]).to(device)
    # The 200-feature global critic exceeds existing 15-epoch parity tolerances
    # when batched (CPU as well as CUDA). Retain the original per-agent critic
    # forward/backward/loss, including optimizer order and all update counts.
    exact_critic=fixed['central'].shape[-1]==200
    floats=torch.zeros(agents,3,dtype=torch.float64,device=device);tensors=torch.zeros(agents,3,device=device)
    for epoch in range(first.ppo_epoch):
        ids=perms[:,epoch,:,None]
        data={k:x.gather(1,ids.expand(-1,-1,x.shape[-1])) for k,x in fixed.items()}
        if exact_critic:
            features=upstream_bases([p.actor.base for p in policies],data['local'])
            mean=independent_linear([p.actor.act.action_out.fc_mean for p in policies],features)
            std=torch.stack([p.actor.act.action_out.logstd._bias.T for p in policies]).exp().expand_as(mean)
        else:mean,std,value=onpolicy_parameters(policies,data['local'],data['central'])
        dist=Normal(mean,std)
        logp=dist.log_prob(data['actions']);ratio=(logp-data['old']).exp()
        score=torch.minimum(ratio*data['adv'],ratio.clamp(1-first.clip_param,1+first.clip_param)*data['adv'])
        active=data['active'];denom=active.sum((1,2))
        policy_loss=-(score*active).sum((1,2))/denom if first._use_policy_active_masks else -score.mean((1,2))
        entropy=(dist.entropy()*active).sum((1,2))/denom if first._use_policy_active_masks else dist.entropy().mean((1,2))
        for p in policies:p.actor_optimizer.zero_grad()
        (policy_loss-first.entropy_coef*entropy).sum().backward()
        actor_norm=[]
        for p in policies:
            actor_norm.append(torch.nn.utils.clip_grad_norm_(p.actor.parameters(),first.max_grad_norm));p.actor_optimizer.step()
        critic_norm=[]
        if exact_critic:
            losses=[]
            for i,(p,t) in enumerate(zip(policies,trainers)):
                # Clone gives the same aligned contiguous input layout as the
                # pinned CPU->CUDA minibatch transfer, rather than a slice.
                prediction=p.critic.v_out(p.critic.base(data['central'][i].clone()))
                loss=t.cal_value_loss(prediction,data['old_value'][i],data['returns'][i],active[i])
                p.critic_optimizer.zero_grad();(first.value_loss_coef*loss).backward()
                critic_norm.append(torch.nn.utils.clip_grad_norm_(p.critic.parameters(),first.max_grad_norm));p.critic_optimizer.step()
                losses.append(loss.detach())
            value_loss=torch.stack(losses)
        else:
            old_value=data['old_value'];clipped=old_value+(value-old_value).clamp(-first.clip_param,first.clip_param)
            loss_fn=(lambda error:huber_loss(error,first.huber_delta)) if first._use_huber_loss else mse_loss
            original=loss_fn(data['returns']-value);clipped_loss=loss_fn(data['returns']-clipped)
            errors=torch.maximum(original,clipped_loss) if first._use_clipped_value_loss else original
            value_loss=(errors*active).sum((1,2))/denom if first._use_value_active_masks else errors.mean((1,2))
            for p in policies:p.critic_optimizer.zero_grad()
            (first.value_loss_coef*value_loss).sum().backward()
            for p in policies:
                critic_norm.append(torch.nn.utils.clip_grad_norm_(p.critic.parameters(),first.max_grad_norm));p.critic_optimizer.step()
        floats+=torch.stack([value_loss.detach(),policy_loss.detach(),entropy.detach()],-1).double()
        tensors+=torch.stack([torch.stack(actor_norm),torch.stack(critic_norm),ratio.detach().mean((1,2))],-1)
    cpu=torch.cat([floats/first.ppo_epoch,(tensors/first.ppo_epoch).double()],-1).cpu().tolist()
    keys=('value_loss','policy_loss','dist_entropy','actor_grad_norm','critic_grad_norm','ratio')
    return [dict(zip(keys,row)) for row in cpu]
