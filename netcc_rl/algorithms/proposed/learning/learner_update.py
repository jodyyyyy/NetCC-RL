"""EXP-011 phase/padding machinery adapted to EXP-012 H/z and node losses.

Source runtime/update: 025d023d055eecbf8ed6770376e8097ec5b7355e.
Model phases retain their approved objectives; Gaussian uses the canonical density.
"""

import torch
import torch.nn.functional as F
from netcc_rl.algorithms.proposed.utils import math


def model_backward(self, obs, action, reward, terminated, model_valid, full_model):
    if getattr(self.cfg, "ablation_arm", None) and self.cfg.dynamics_type == "disabled":
        from netcc_rl.algorithms.proposed.ablation.model_free import model_free_backward
        return model_free_backward(self, obs, action, reward, terminated, model_valid)
    task = None
    self.optim.zero_grad(set_to_none=False)
    self.model.eval()
    with torch.no_grad():
        graph_dynamics = "global_mlp" == "graph_mpnn"
        next_nodes = None
        next_nodes, next_z = self.model.encode_actor_features(obs[1:])
        td_targets = self._td_target(
            next_z, reward, terminated, task, next_nodes=next_nodes
        )
    self.model.train()
    zs = torch.empty(
        self.cfg.horizon + 1, action.shape[1], self.cfg.latent_dim, device=self.device
    )
    predicted_nodes = []
    z = self.model.encode(obs[0], task)
    zs[0] = z
    for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
        z = self.model.next(z, _action, task)
        zs[t + 1] = z
    _zs = zs[:-1]
    critic_actions = action
    valid_mask = model_valid
    value_loss_rollout = self._value_loss(
        _zs, critic_actions, td_targets, valid_mask, task
    )
    critic_encoder_z = self.model.encode(obs[:-1], task)
    value_loss_encoder = self._value_loss(
        critic_encoder_z, critic_actions, td_targets, valid_mask, task
    )
    value_loss = (
        value_loss_rollout + self.cfg.critic_encoder_aux_coef * value_loss_encoder
    )
    reward_preds = self.model.reward(_zs, action, task)
    if self.cfg.episodic:
        termination_pred = self.model.termination(zs[1:], task, unnormalized=True)
    rho = self.cfg.rho ** torch.arange(self.cfg.horizon, device=z.device)
    errors = (zs[1:] - next_z).square().mean(-1)
    reward_errors = math.soft_ce(
        reward_preds.reshape(-1, reward_preds.shape[-1]),
        reward.reshape(-1, 1),
        self.cfg,
    ).reshape(model_valid.shape)
    if full_model:
        consistency_loss = (errors.mean(-1) * rho).sum()
        reward_loss = (reward_errors.mean(-1) * rho).sum()
    else:
        consistency_loss = (torch.where(model_valid, errors, 0).sum(-1) * rho).sum()
        reward_loss = (torch.where(model_valid, reward_errors, 0).sum(-1) * rho).sum()
    denominator = self.cfg.horizon if full_model else model_valid.sum()
    consistency_loss = consistency_loss / denominator
    reward_loss = reward_loss / denominator
    node_loss = zs.new_zeros(())
    gradient_info = {}
    if self.cfg.episodic:
        termination_errors = F.binary_cross_entropy_with_logits(
            termination_pred, terminated, reduction="none"
        ).squeeze(-1)
        termination_loss = (
            torch.where(model_valid, termination_errors, 0).sum() / model_valid.sum()
        )
    else:
        termination_loss = zs.new_zeros(())
    total_loss = (
        self.cfg.consistency_coef * consistency_loss
        + self.cfg.reward_coef * reward_loss
        + self.cfg.termination_coef * termination_loss
        + self.cfg.value_coef * value_loss
    )
    self._learner_runtime.backward(total_loss, "model")
    result = dict(
        consistency_loss=consistency_loss.detach(),
        reward_loss=reward_loss.detach(),
        value_loss=value_loss.detach(),
        value_loss_rollout=value_loss_rollout.detach(),
        value_loss_encoder=value_loss_encoder.detach(),
        value_loss_total=value_loss.detach(),
        termination_loss=termination_loss.detach(),
        total_loss=total_loss.detach(),
    )
    return result


def row_mean(value, padding):
    if padding is None:
        return value.mean()
    mask = padding["mask"].reshape(-1, *[1] * (value.ndim - 1))
    return torch.where(mask, value, 0).sum() / (
        padding["mask"].sum() * (value.numel() // len(value))
    )


def update_scale(self, values, padding):
    if padding is None:
        self.scale.update(values)
        return
    with torch.no_grad():
        count = padding["mask"].sum()
        ordered = (
            torch.where(padding["mask"][:, None], values.detach(), float("inf"))
            .sort(0)
            .values
        )
        positions = self.scale._percentiles * (count - 1) / 100
        low = positions.floor().long()
        high = torch.minimum(low + 1, count - 1)
        weight = (positions - low)[:, None]
        percentiles = ordered[low] * (1 - weight) + ordered[high] * weight
        self.scale.value.lerp_(
            (percentiles[1] - percentiles[0]).clamp_min(1.0), self.cfg.tau
        )


def actor_backward(self, zs, action, actor_obs, weights, full, padding=None):
    """Keep EXP-007 Gaussian objective unchanged in an FP32 graph phase."""
    self.pi_optim.zero_grad(set_to_none=False)
    self.model.eval()
    if padding is not None:
        logits = self.model._pi(zs, node_states=actor_obs)
        mean, log_std = logits.chunk(2, -1)
        log_std = self.model.policy_log_std(log_std)
        sampled_action, policy_info = self.model._squashed_gaussian(
            mean, log_std, padding["noise"]
        )
    else:
        sampled_action, policy_info = self.model.pi(zs, node_states=actor_obs)
    qs = self._actor_Q(zs, sampled_action)
    update_scale(self, qs[0] if full else qs[:, 0], padding)
    scaled_q = self.scale(qs)
    pi_loss = row_mean(
        -(self.cfg.entropy_coef * policy_info["scaled_entropy"] + scaled_q).mean((1, 2))
        * weights,
        padding,
    )
    info = {
        "pi_entropy": row_mean(policy_info["entropy"], padding).detach(),
        "pi_scaled_entropy": row_mean(policy_info["scaled_entropy"], padding).detach(),
    }
    self._learner_runtime.backward(pi_loss, "actor")
    info.update(pi_loss=pi_loss.detach(), pi_scale=self.scale.value.detach())
    return info


def update(self, obs, action, reward, terminated, task=None, model_valid_mask=None):
    runtime = self._learner_runtime
    self._validate_batch(obs, action, reward, terminated, task)
    valid = (
        model_valid_mask
        if model_valid_mask is not None
        else torch.ones(action.shape[:2], device=action.device, dtype=torch.bool)
    )
    full, any_valid = torch.stack([valid.all(), valid.any()]).tolist()
    if not any_valid:
        raise ValueError("batch must contain a real model transition")
    if not full:
        obs_valid = torch.cat([torch.ones_like(valid[:1]), valid], 0)
        obs = torch.where(obs_valid[..., None], obs, 0)
        action, reward, terminated = (
            torch.where(valid[..., None], v, 0) for v in (action, reward, terminated)
        )
    info = runtime.call(
        "model_backward",
        lambda *a: model_backward(self, *a),
        (obs, action, reward, terminated, valid, full),
    )
    world_ids = {id(p) for g in self.optim.param_groups for p in g["params"]}
    world_params = [
        p for p in self.model.parameters() if id(p) in world_ids and p.grad is not None
    ]
    model_norm = runtime.finish("model", info["total_loss"], world_params)
    self.model.eval()

    @torch.no_grad()
    def encode(x):
        return self.model.encode_actor_features(x)

    features = runtime.call("post_encode", encode, (obs[:-1],))
    actor_nodes = None
    actor_nodes, zs = (x.detach() for x in features)
    actor_valid = valid
    actor_full, actor_any = torch.stack([actor_valid.all(), actor_valid.any()]).tolist()
    if actor_any:
        rho = self.cfg.rho ** torch.arange(len(zs), device=self.device)
        weights = (
            rho if actor_full else rho[:, None].expand_as(actor_valid)[actor_valid]
        )
        actor_z = zs if actor_full else zs[actor_valid].unsqueeze(1)
        actor_action = action if actor_full else action[actor_valid].unsqueeze(1)
        actor_obs = actor_nodes
        if actor_obs is not None and (not actor_full):
            actor_obs = actor_obs[actor_valid].unsqueeze(1)
        padding = None
        if runtime.graphs and (not actor_full):
            count = actor_z.shape[0]
            capacity = valid.numel()
            noise = torch.randn_like(actor_action)
            time = None

            def pad(x):
                return torch.cat([x, x.new_zeros(capacity - count, *x.shape[1:])])

            padding = dict(
                mask=torch.arange(capacity, device=self.device) < count,
                noise=pad(noise),
                time=pad(time) if time is not None else None,
            )
            actor_z, actor_action, actor_obs, weights = (
                pad(x) if x is not None else None
                for x in (actor_z, actor_action, actor_obs, weights)
            )
        pi_info = runtime.call(
            "actor_backward",
            lambda *a: actor_backward(self, *a),
            (actor_z, actor_action, actor_obs, weights, actor_full, padding),
        )
        pi_norm = runtime.finish(
            "actor", pi_info["pi_loss"], self.model._pi.parameters()
        )
    else:
        pi_info = self.update_pi(
            zs,
            action,
            None,
            actor_obs=None,
            actor_nodes=actor_nodes,
            model_valid_mask=valid,
        )
        pi_norm = pi_info["pi_grad_norm"]
    combined = dict(
        info,
        **pi_info,
        grad_norm=model_norm,
        pi_grad_norm=pi_norm,
        actor_valid_count=actor_valid.sum(),
        actor_update_skipped=zs.new_tensor(float(not actor_any))
    )
    runtime.call("target_polyak", self.model.soft_update_target_Q, ())
    self.model.eval()
    runtime.updates += 1
    combined.update(
        model_valid_count=valid.sum(),
        valid_transition_count=valid.sum(),
        valid_transition_fraction=valid.float().mean(),
    )
    packed = torch.stack([v.detach().float().mean() for v in combined.values()])
    return dict(zip(combined, packed.unbind()))
