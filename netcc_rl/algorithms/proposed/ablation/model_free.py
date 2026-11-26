"""One-step model-free Bellman objective and learner update."""

import torch
from torch import nn


def model_free_update(agent, obs, action, reward, terminated, task, model_valid_mask):
    """A9 real-transition critic update, then unchanged detached-latent actor."""
    agent._validate_batch(obs, action, reward, terminated, task)
    valid = (
        torch.ones(action.shape[:2], device=action.device, dtype=torch.bool)
        if model_valid_mask is None
        else model_valid_mask
    )
    if (
        valid.dtype != torch.bool
        or valid.shape != action.shape[:2]
        or not bool(valid.any())
    ):
        raise ValueError("model-free update requires boolean valid real transitions")
    obs_valid = torch.cat((torch.ones_like(valid[:1]), valid), 0)
    obs = torch.where(obs_valid[..., None], obs, 0)
    action, reward, terminated = (
        torch.where(valid[..., None], x, 0) for x in (action, reward, terminated)
    )
    model = agent.model
    agent.optim.zero_grad(set_to_none=True)
    with torch.no_grad():
        next_nodes, next_z = model.encode_actor_features(obs[1:])
        targets = agent._td_target(
            next_z, reward, terminated, task, next_nodes=next_nodes
        )
    model.train()
    z = model.encode(obs[:-1], task)
    value = agent._value_loss(z, action, targets, valid, task)
    loss = agent.cfg.value_coef * value
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model._clip_parameters, agent.cfg.grad_clip_norm)
    agent._check_finite_step(loss, norm, "model-free critic")
    agent.optim.step()
    agent.optim.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        nodes, z = model.encode_actor_features(obs[:-1])
    info = agent.update_pi(z, action, actor_nodes=nodes, model_valid_mask=valid)
    model.soft_update_target_Q()
    info.update(
        value_loss=value,
        value_loss_encoder=value,
        value_loss_total=value,
        total_loss=loss,
        grad_norm=norm,
        valid_transition_count=valid.sum(),
        valid_transition_fraction=valid.float().mean(),
    )
    return {k: v.detach().float().mean() for k, v in info.items()}


def model_free_backward(agent, obs, action, reward, terminated, valid):
    """Identical D0 loss, with backward captured by the segmented runtime."""
    agent.optim.zero_grad(set_to_none=False)
    model = agent.model
    model.eval()
    with torch.no_grad():
        nodes, next_z = model.encode_actor_features(obs[1:])
        target = agent._td_target(next_z, reward, terminated, None, next_nodes=nodes)
    model.train()
    z = model.encode(obs[:-1], None)
    value = agent._value_loss(z, action, target, valid, None)
    loss = agent.cfg.value_coef * value
    loss.backward()
    return dict(
        value_loss=value.detach(),
        value_loss_encoder=value.detach(),
        value_loss_total=value.detach(),
        total_loss=loss.detach(),
    )
