"""TD-MPC2 value/model updates with a one-step Gaussian actor.

World-model losses and TD targets derive from upstream commit
e9f59321933cbc8e11a002b842adc7d4ffae8ff1 (see LICENSE.TDMPC2).
"""

from copy import deepcopy
from pathlib import Path
from time import perf_counter
import torch
import torch.nn.functional as F
from .config import GAUSSIAN_POLICY_CONTRACT, actor_metadata, episode_discount, validate
from .common import math
from .common.scale import RunningScale
from .common.world_model import WorldModel


class TDMPC2(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = validate(cfg)
        self.device = torch.device(cfg.device)
        model_type = WorldModel
        if getattr(cfg, "model_width_profile", "custom") == "exp016_adaptive_v1":
            from .common.adaptive_architecture import validate_identity
            from .common.adaptive_world_model import AdaptiveWorldModel

            validate_identity(cfg)
            model_type = AdaptiveWorldModel
        self.model = model_type(cfg).to(self.device)
        for module in self.model.modules():
            if hasattr(module, "prepare_static_graphs"):
                module.prepare_static_graphs()
        from .common.learner_runtime import GraphAdam, graph_support

        status = graph_support(cfg, self.device)
        if cfg.learner_cuda_graphs and status != "enabled":
            raise ValueError(f"requested learner CUDA Graphs unavailable: {status}")
        adam = GraphAdam if status == "enabled" else torch.optim.Adam
        self.optim = adam(
            [
                {
                    "params": self.model._encoder.parameters(),
                    "lr": cfg.lr * cfg.enc_lr_scale,
                },
                {
                    "params": (
                        self.model._dynamics.parameters()
                        if self.model._dynamics is not None
                        else []
                    )
                },
                {
                    "params": (
                        self.model._reward.parameters()
                        if self.model._reward is not None
                        else []
                    )
                },
                {
                    "params": (
                        self.model._termination.parameters()
                        if self.model._termination is not None
                        else []
                    )
                },
                {"params": self.model._Qs.parameters()},
            ],
            lr=cfg.lr,
            capturable=self.device.type == "cuda",
        )
        self.pi_optim = adam(
            self.model._pi.parameters(),
            lr=cfg.lr,
            eps=1e-05,
            capturable=self.device.type == "cuda",
        )
        self.scale = RunningScale(cfg, self.device)
        self.discount = self._get_discount(cfg.episode_length)
        self.model.eval()
        self.eval_generator = torch.Generator(device=self.device)
        self.reset_eval_seed(cfg.eval_seed)
        self.last_act_info = {}
        if status == "enabled":
            from .common.learner_runtime import LearnerRuntime

            self.__dict__["_learner_runtime"] = LearnerRuntime(self)
        if cfg.compile and cfg.learner_cuda_graphs and getattr(cfg, "exp017_arm", None):
            from .common.exp017 import compile_modules

            self.__dict__["_compiled_module_names"] = compile_modules(self.model)
        elif cfg.compile:
            if torch._dynamo.config.suppress_errors:
                raise ValueError(
                    "compiled learner requires torch._dynamo.config.suppress_errors=false"
                )
            self._update = torch.compile(self._update, mode="reduce-overhead")

    def _get_discount(self, episode_length):
        return episode_discount(self.cfg, episode_length)

    def reset_eval_seed(self, seed=None):
        self.eval_generator.manual_seed(self.cfg.eval_seed if seed is None else seed)

    def save(self, fp):
        """Save model weights and resolved settings; this is not exact resumption."""
        fp = Path(fp)
        fp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 4,
                "kind": "model_weights",
                "training_resume_supported": False,
                "metadata": actor_metadata(self.cfg),
                "config": deepcopy(vars(self.cfg)),
                "model": self.model.state_dict(),
            },
            fp,
        )

    def load(self, fp):
        """Restore matching weights after checking policy, environment and architecture."""
        from types import SimpleNamespace
        from .common.adaptive_architecture import validate_identity

        checkpoint = (
            fp
            if isinstance(fp, dict)
            else torch.load(fp, map_location=self.device, weights_only=False)
        )
        metadata, saved = (
            checkpoint.get("metadata") or {},
            checkpoint.get("config") or {},
        )
        expected = actor_metadata(self.cfg)
        for key in (
            "gaussian_policy_contract",
            "gaussian_action_low",
            "gaussian_action_high",
            "mainline_contract",
        ):
            if metadata.get(key) != expected[key]:
                raise ValueError(
                    "checkpoint policy contract or bounds differ; train from scratch"
                )
        for key in (
            "actor_type",
            "actor_architecture",
            "gaussian_action_low",
            "gaussian_action_high",
            "log_std_min",
            "log_std_max",
            "num_bins",
            "vmin",
            "vmax",
            "horizon",
            "critic_encoder_aux_coef",
            "graph_attention_dropout",
        ):
            if saved.get(key) != getattr(self.cfg, key):
                raise ValueError(f"checkpoint {key} differs; train from scratch")
        if validate_identity(SimpleNamespace(**saved)) != validate_identity(self.cfg):
            raise ValueError(
                "checkpoint architecture/environment identity differs; train from scratch"
            )
        for key in (
            "state_encoder_type",
            "exp017_arm",
            "exp017_environment",
            "exp017_factors",
            "exp017_identity_sha256",
            "dynamics_enabled",
            "model_horizon",
        ):
            if metadata.get(key) != expected.get(key):
                raise ValueError(
                    f"checkpoint {key} identity differs; train from scratch"
                )
        source, target = (checkpoint["model"], self.model.state_dict())
        if source.keys() != target.keys() or any(
            (
                (not torch.is_tensor(source[k]) or source[k].shape != target[k].shape)
                if torch.is_tensor(target[k])
                else source[k] != target[k]
            )
            for k in target
        ):
            raise ValueError("checkpoint model keys/shapes differ; train from scratch")
        self.model.load_state_dict(source, strict=True)
        self.optim.state.clear()
        self.pi_optim.state.clear()
        self.scale.value.fill_(1)
        if self.__dict__.get("_learner_runtime") is not None:
            from .common.learner_runtime import LearnerRuntime

            self.optim.zero_grad(set_to_none=True)
            self.pi_optim.zero_grad(set_to_none=True)
            self.__dict__["_learner_runtime"] = LearnerRuntime(self)
        self.model.eval()
        return dict(
            policy_restored=True,
            world_model_restored=True,
            critic_restored=True,
            training_state_restored=False,
            value_support_verified=True,
        )

    @torch.no_grad()
    def act(
        self, obs, t0=False, eval_mode=False, task=None, *, noise=None, return_cpu=True
    ):
        """Sample one action from the current real observation."""
        if self.cfg.mpc:
            raise ValueError("NetCC-RL control requires mpc=false")
        if task is not None:
            raise ValueError("only single-task control is supported")
        start = perf_counter()
        obs = obs.to(self.device, non_blocking=True)
        single = obs.ndim == 1
        if single:
            obs = obs.unsqueeze(0)
            if noise is not None:
                noise = noise.unsqueeze(0)
        runtime = self.__dict__.get("_learner_runtime")
        if runtime is not None:
            shape = (*obs.shape[:-1], self.cfg.action_dim)
            if noise is None:
                if eval_mode and self.cfg.eval_policy == "zero_noise":
                    noise = torch.zeros(shape, device=self.device)
                else:
                    noise = torch.randn(
                        shape,
                        device=self.device,
                        generator=self.eval_generator if eval_mode else None,
                    )
            elif noise.shape != shape:
                raise ValueError("Gaussian initial noise must match the action shape")
            if not bool(torch.isfinite(noise).all()):
                raise ValueError("Gaussian initial noise must be finite")
            noise = noise.to(device=self.device, dtype=obs.dtype).reshape(shape)

            def deployment(observation, eps):
                h, z = self.model.encode_actor_features(observation)
                logits = self.model._pi(z, node_states=h)
                mean, raw = logits.chunk(2, -1)
                log_std = self.model.policy_log_std(raw)
                action, info = self.model._squashed_gaussian(mean, log_std, eps)
                return (action, dict(info.items()))

            action, info = runtime.call("deployment", deployment, (obs, noise))
        else:
            nodes = None
            nodes, z = self.model.encode_actor_features(obs)
            if eval_mode and noise is None and (self.cfg.eval_policy == "zero_noise"):
                noise = torch.zeros(
                    *z.shape[:-1], self.cfg.action_dim, device=self.device
                )
            action, info = self.model.pi(
                z,
                noise=noise,
                node_states=nodes,
                generator=self.eval_generator if eval_mode else None,
            )
        if eval_mode and self.cfg.eval_policy == "gaussian_mean":
            action = info["mean"]
        if return_cpu:
            action = action.cpu()
        self.last_act_info = {"act_latency_ms": 1000 * (perf_counter() - start)}
        return action[0] if single else action

    def plan(self, *args, **kwargs):
        raise RuntimeError("MPC is disabled in NetCC-RL")

    def _plan(self, *args, **kwargs):
        raise RuntimeError("MPC is disabled in NetCC-RL")

    def _estimate_value(self, *args, **kwargs):
        raise RuntimeError("MPC trajectory values are disabled in NetCC-RL")

    def update_pi(
        self,
        zs,
        action,
        task=None,
        *,
        actor_obs=None,
        actor_nodes=None,
        model_valid_mask=None,
    ):
        if task is not None:
            raise ValueError("only single-task actor updates are supported")
        action_shape = (*zs.shape[:-1], self.cfg.action_dim)
        if zs.ndim != 3 or action.shape != action_shape:
            raise ValueError("actor requires aligned [H,B,Z] and replay [H,B,D]")
        if len(zs) != self.cfg.horizon:
            raise ValueError(
                "actor requires H latents and H replay actions, never H+1 latents"
            )
        from .common.node_features import node_layout

        expected_nodes = (
            *zs.shape[:-1],
            node_layout(self.cfg)[0],
            self.model._pi.width,
        )
        if actor_nodes is None or actor_nodes.shape != expected_nodes:
            raise ValueError(
                f"node embedding actor requires current actor_nodes {expected_nodes}"
            )
        actor_nodes = actor_nodes.detach()
        zs, action = (zs.detach(), action.detach())
        if model_valid_mask is None:
            model_valid_mask = torch.ones(
                zs.shape[:-1], device=zs.device, dtype=torch.bool
            )
        if (
            model_valid_mask.dtype != torch.bool
            or model_valid_mask.shape != zs.shape[:-1]
        ):
            raise ValueError("model_valid_mask must be boolean [H,B]")
        valid = model_valid_mask
        count = valid.sum()
        if not bool(count):
            zero = zs.new_zeros(())
            return dict(
                pi_loss=zero,
                pi_grad_norm=zero,
                pi_scale=self.scale.value.detach().clone(),
                actor_valid_count=count.detach().clone(),
                actor_update_skipped=zero + 1,
            )
        full = bool(valid.all())
        rho = self.cfg.rho ** torch.arange(len(zs), device=self.device)
        weights = rho
        if not full:
            weights = rho[:, None].expand_as(valid)[valid]
            zs, action = (zs[valid].unsqueeze(1), action[valid].unsqueeze(1))
            actor_nodes = actor_nodes[valid].unsqueeze(1)
        self.pi_optim.zero_grad(set_to_none=True)
        training = self.model.training
        self.model.eval()
        sampled_action, policy_info = self.model.pi(zs, node_states=actor_nodes)
        qs = self._actor_Q(zs, sampled_action)
        self.scale.update(qs[0] if full else qs[:, 0])
        scaled_q = self.scale(qs)
        pi_loss = (
            -(self.cfg.entropy_coef * policy_info["scaled_entropy"] + scaled_q).mean(
                (1, 2)
            )
            * weights
        ).mean()
        info = {
            "pi_entropy": policy_info["entropy"].mean(),
            "pi_scaled_entropy": policy_info["scaled_entropy"].mean(),
        }
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(), self.cfg.grad_clip_norm
        )
        self._check_finite_step(pi_loss, pi_grad_norm, "actor")
        self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.train(training)
        info.update(
            pi_loss=pi_loss,
            pi_grad_norm=pi_grad_norm,
            pi_scale=self.scale.value,
            actor_valid_count=count,
            actor_update_skipped=zs.new_zeros(()),
        )
        return {key: value.detach().clone() for key, value in info.items()}

    def _actor_Q(self, zs, action):
        return self.model.Q(zs, action, None, return_type="avg", detach=True)

    def _validate_batch(self, obs, action, reward, terminated, task):
        if task is not None:
            raise ValueError("only single-task updates are supported")
        if action.ndim != 3 or action.shape[0] != self.cfg.horizon:
            raise ValueError("replay action must be [H,B,action_dim]")
        h, b, d = action.shape
        if d != self.cfg.action_dim or obs.shape != (
            h + 1,
            b,
            self.cfg.obs_shape["state"][0],
        ):
            raise ValueError("replay observation/action alignment is invalid")
        if reward.shape != (h, b, 1) or terminated.shape != reward.shape:
            raise ValueError("reward and terminated must be [H,B,1]")

    @torch.no_grad()
    def _td_target(
        self, next_z, reward, terminated, task, *, next_obs=None, next_nodes=None
    ):
        """
        Compute the TD-target from a reward and the observation at the following time step.

        Args:
            next_z (torch.Tensor): Latent state at the following time step.
            reward (torch.Tensor): Reward at the current time step.
            terminated (torch.Tensor): Termination signal at the current time step.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: TD-target.
        """
        action, _ = self.model.pi(next_z, task, node_states=next_nodes)
        discount = self.discount
        return reward + discount * (1 - terminated) * self.model.Q(
            next_z, action, task, return_type="min", target=True
        )

    def _value_loss(self, z, actions, targets, valid_mask=None, task=None):
        """Average the five single-step critics over real transitions only."""
        valid = (
            torch.ones(z.shape[:-1], device=z.device, dtype=torch.bool)
            if valid_mask is None
            else valid_mask
        )
        safe_z = torch.where(valid[..., None], z, 0)
        safe_a = torch.where(valid[..., None], actions, 0)
        qs = self.model.Q(safe_z, safe_a, task, return_type="all")
        mask = valid.unsqueeze(0).expand(qs.shape[:-1])
        labels = targets.unsqueeze(0).expand(*qs.shape[:-1], 1)
        losses = math.soft_ce(
            torch.where(mask[..., None], qs, 0).reshape(-1, qs.shape[-1]),
            torch.where(mask[..., None], labels, 0).reshape(-1, 1),
            self.cfg,
        ).reshape(mask.shape)
        rho = self.cfg.rho ** torch.arange(len(z), device=z.device)
        return (torch.where(mask, losses, 0) * rho[None, :, None]).sum() / (
            self.cfg.num_q * valid.sum().clamp_min(1)
        )

    def _update(
        self, obs, action, reward, terminated, task=None, model_valid_mask=None
    ):
        if (
            getattr(self.cfg, "exp017_arm", None)
            and self.cfg.dynamics_type == "disabled"
            and self.__dict__.get("_learner_runtime") is None
        ):
            from .common.exp017 import model_free_update

            return model_free_update(
                self, obs, action, reward, terminated, task, model_valid_mask
            )
        if self.__dict__.get("_learner_runtime") is not None:
            from .common.learner_update import update

            return update(self, obs, action, reward, terminated, task, model_valid_mask)
        self._validate_batch(obs, action, reward, terminated, task)
        model_valid = (
            model_valid_mask
            if model_valid_mask is not None
            else torch.ones(action.shape[:2], dtype=torch.bool, device=action.device)
        )
        full_model, any_model = torch.stack(
            [model_valid.all(), model_valid.any()]
        ).tolist()
        if not any_model:
            raise ValueError("batch must contain a real model transition")
        if not full_model:
            obs_valid = torch.cat([torch.ones_like(model_valid[:1]), model_valid], 0)
            obs = torch.where(obs_valid[..., None], obs, 0)
            action, reward, terminated = (
                torch.where(model_valid[..., None], x, 0)
                for x in (action, reward, terminated)
            )
        self.optim.zero_grad(set_to_none=True)
        with torch.no_grad():
            graph_dynamics = "global_mlp" == "graph_mpnn"
            next_nodes = None
            next_nodes, next_z = self.model.encode_actor_features(obs[1:])
            td_targets = self._td_target(
                next_z, reward, terminated, task, next_nodes=next_nodes
            )
        self.model.train()
        zs = torch.empty(
            self.cfg.horizon + 1,
            action.shape[1],
            self.cfg.latent_dim,
            device=self.device,
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
            reward_loss = (
                torch.where(model_valid, reward_errors, 0).sum(-1) * rho
            ).sum()
        denominator = self.cfg.horizon if full_model else model_valid.sum()
        consistency_loss = consistency_loss / denominator
        reward_loss = reward_loss / denominator
        node_loss = zs.new_zeros(())
        gradient_info = {}
        if self.cfg.episodic:
            termination_loss = F.binary_cross_entropy_with_logits(
                termination_pred[model_valid], terminated[model_valid]
            )
        else:
            termination_loss = 0.0
        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.termination_coef * termination_loss
            + self.cfg.value_coef * value_loss
        )
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model._clip_parameters, self.cfg.grad_clip_norm
        )
        self._check_finite_step(total_loss, grad_norm, "model")
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)
        self.model.eval()
        with torch.no_grad():
            actor_nodes, z_actor = self.model.encode_actor_features(obs[:-1])
        actor_actions = action
        pi_kwargs = dict(model_valid_mask=model_valid)
        pi_info = self.update_pi(
            z_actor.detach(),
            actor_actions.detach(),
            task,
            actor_nodes=actor_nodes.detach(),
            **pi_kwargs,
        )
        self.model.soft_update_target_Q()
        self.model.eval()
        info = {
            "model_valid_count": model_valid.sum(),
            "consistency_loss": consistency_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "value_loss_rollout": value_loss_rollout,
            "value_loss_encoder": value_loss_encoder,
            "value_loss_total": value_loss,
            "termination_loss": termination_loss,
            "total_loss": total_loss,
            "grad_norm": grad_norm,
        }
        info.update(
            valid_transition_count=valid_mask.sum(),
            valid_transition_fraction=valid_mask.float().mean(),
        )
        if self.cfg.episodic:
            info.update(
                math.termination_statistics(
                    torch.sigmoid(termination_pred[model_valid]),
                    terminated[model_valid],
                )
            )
        info.update(pi_info)
        return {
            k: (
                v.detach().float().mean()
                if isinstance(v, torch.Tensor)
                else zs.new_tensor(v)
            )
            for k, v in info.items()
        }

    def update(self, buffer, *, timer=None):
        """
        Main update function. Corresponds to one iteration of model learning.

        Args:
            buffer (common.buffer.Buffer): Replay buffer.

        Returns:
            dict: Dictionary of training statistics.
        """
        kwargs = {}
        if timer is None:
            batch = buffer.sample()
        else:
            with timer.measure("sample"):
                batch = buffer.sample()
        obs, action, reward, terminated, task = batch["model_batch"]
        kwargs["model_valid_mask"] = batch["model_valid_mask"]
        if task is not None:
            kwargs["task"] = task
        if timer is None:
            return self._update(obs, action, reward, terminated, **kwargs)
        with timer.measure("learner_update"):
            return self._update(obs, action, reward, terminated, **kwargs)

    @staticmethod
    def _check_finite_step(loss, grad_norm, name):
        if not bool(
            torch.stack(
                [torch.isfinite(loss).all(), torch.isfinite(grad_norm).all()]
            ).all()
        ):
            raise FloatingPointError(
                f"nonfinite {name} loss or gradient before optimizer step"
            )
