"""Differentiable independent-parameter MLP batching; original Parameters stay owned.

Stacking is a differentiable view operation, not parameter sharing or a new
optimizer. Leading dimension is module identity, second is arbitrary samples.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .phase_features import phase_features, FEATURE_MULTIPLIER


def independent_linear(modules, inputs):
    width=max(m.in_features for m in modules)
    weights=torch.stack([(m.weight if m.in_features==width else F.pad(m.weight,(0,width-m.in_features))) for m in modules])
    biases=torch.stack([m.bias for m in modules])
    return torch.bmm(inputs,weights.transpose(1,2))+biases[:,None,:]


def independent_mlp(modules, inputs):
    for index,layer in enumerate(modules[0]):
        if isinstance(layer,nn.Linear):inputs=independent_linear([m[index] for m in modules],inputs)
        elif isinstance(layer,nn.ReLU):inputs=inputs.relu()
        elif isinstance(layer,nn.Tanh):inputs=inputs.tanh()
        else:raise TypeError(f'unsupported independent layer {type(layer)}')
    return inputs


def gather_padded(states, nodes):
    """Pack encoded ragged inputs for bmm; padding has zero matching weights.

    Each original network still owns exactly 2*k input columns, without mask
    features. Internal rectangular packing does not extend its observation.
    """
    width=max(map(len,nodes));ids=torch.tensor([list(x)+[0]*(width-len(x)) for x in nodes],device=states.device)
    mask=torch.arange(width,device=states.device)[None,:]<torch.tensor(list(map(len,nodes)),device=states.device)[:,None]
    flat=states.reshape(-1,states.shape[-1])
    features=phase_features(flat[:,ids])
    return features.permute(1,0,2)*mask.repeat_interleave(FEATURE_MULTIPLIER,-1)[:,None,:]


def independent_actor_parameters(actors, states, nodes):
    inputs=gather_padded(states,nodes)
    logits=independent_mlp([a.net for a in actors],inputs)
    if actors[0].kind=='happo':mean,logstd=logits.split(1,-1)
    else:
        mean=independent_linear([a.mean_head for a in actors],logits)
        logstd=torch.stack([a.log_std for a in actors])[:,None,:].expand_as(mean)
    shape=(*states.shape[:-1],len(actors))
    return mean.squeeze(-1).transpose(0,1).reshape(shape),logstd.exp().squeeze(-1).transpose(0,1).reshape(shape)


def independent_norm(modules, inputs):
    weights=torch.stack([m.weight for m in modules]);biases=torch.stack([m.bias for m in modules])
    return torch.vmap(lambda x,w,b:F.layer_norm(x,modules[0].normalized_shape,w,b,modules[0].eps))(inputs,weights,biases)


def independent_sequence(modules,inputs):
    for i,layer in enumerate(modules[0]):
        same=[m[i] for m in modules]
        if isinstance(layer,nn.Linear):inputs=independent_linear(same,inputs)
        elif isinstance(layer,nn.LayerNorm):inputs=independent_norm(same,inputs)
        elif isinstance(layer,nn.ReLU):inputs=inputs.relu()
        elif isinstance(layer,nn.Tanh):inputs=inputs.tanh()
        else:raise TypeError(type(layer))
    return inputs


def upstream_bases(modules,inputs):
    if modules[0]._use_feature_normalization:inputs=independent_norm([m.feature_norm for m in modules],inputs)
    inputs=independent_sequence([m.mlp.fc1 for m in modules],inputs)
    for j in range(modules[0]._layer_N):inputs=independent_sequence([m.mlp.fc2[j] for m in modules],inputs)
    return inputs




def onpolicy_parameters(policies,local,central):
    if any(p.actor._use_recurrent_policy or p.actor._use_naive_recurrent_policy for p in policies):raise ValueError('feedforward only')
    x=upstream_bases([p.actor.base for p in policies],local)
    mean=independent_linear([p.actor.act.action_out.fc_mean for p in policies],x)
    std=torch.stack([p.actor.act.action_out.logstd._bias.T for p in policies]).exp().expand_as(mean)
    value=upstream_bases([p.critic.base for p in policies],central)
    value=independent_linear([p.critic.v_out for p in policies],value)
    return mean,std,value


def normal_sample(mean,std):
    """Normal draw after caller validates the distribution once for the batch.

    torch.normal(Tensor, CUDA Tensor) synchronizes to validate every scalar
    standard deviation. Keep one randn call per original actor, preserving draw
    order; affine arithmetic may differ by final-bit rounding on CUDA.
    """
    return mean+std*torch.randn_like(mean,memory_format=torch.contiguous_format)
