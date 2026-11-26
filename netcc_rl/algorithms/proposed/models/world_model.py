from copy import deepcopy
import torch
import torch.nn as nn
from netcc_rl.algorithms.proposed.models import layers
from netcc_rl.algorithms.proposed.utils import math
from netcc_rl.algorithms.proposed.utils import init
from tensordict.nn import TensorDictParams


class WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    NetCC-RL exposes single-task vector observations in this implementation.
    """

    def __init__(self, cfg):
        super().__init__()
        if cfg.task_dim != 0:
            raise ValueError("NetCC-RL world model supports single-task inputs only")
        self.cfg = cfg
        self._encoder = layers.enc(cfg)
        model_free = (
            getattr(cfg, "ablation_arm", None) and cfg.dynamics_type == "disabled"
        )
        self._dynamics = (
            None
            if model_free
            else layers.mlp(
                cfg.latent_dim + cfg.action_dim + cfg.task_dim,
                2 * [cfg.mlp_dim],
                cfg.latent_dim,
                act=layers.SimNorm(cfg),
            )
        )
        self._reward = (
            None
            if model_free
            else layers.mlp(
                cfg.latent_dim + cfg.action_dim + cfg.task_dim,
                2 * [cfg.mlp_dim],
                max(cfg.num_bins, 1),
            )
        )
        self._termination = (
            layers.mlp(cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], 1)
            if cfg.episodic and not model_free
            else None
        )
        from netcc_rl.algorithms.proposed.models.graph_networks import SharedNativeGaussianActor

        if (
            getattr(cfg, "ablation_arm", None)
            and cfg.actor_architecture == "global_latent"
        ):
            from netcc_rl.algorithms.proposed.ablation.variants import GlobalGaussianActor

            self._pi = GlobalGaussianActor(cfg)
        else:
            self._pi = SharedNativeGaussianActor(cfg)
        q_action_dim = cfg.action_dim
        self._Qs = layers.Ensemble(
            [
                layers.mlp(
                    cfg.latent_dim + q_action_dim + cfg.task_dim,
                    2 * [cfg.mlp_dim],
                    max(cfg.num_bins, 1),
                    dropout=cfg.dropout,
                ).apply(init.weight_init)
                for _ in range(cfg.num_q)
            ]
        )
        self.apply(init.weight_init)
        init.zero_([self._Qs.params["2", "weight"]])
        if self._reward is not None:
            init.zero_([self._reward[-1].weight])
        self.register_buffer("log_std_min", torch.tensor(float(cfg.log_std_min)))
        self.register_buffer(
            "log_std_dif", torch.tensor(float(cfg.log_std_max)) - self.log_std_min
        )
        self.init()

    def init(self):
        self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
        self._target_Qs_params = TensorDictParams(
            self._Qs.params.data.clone(), no_convert=True
        )
        with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
            self._detach_Qs = deepcopy(self._Qs)
            self._target_Qs = deepcopy(self._Qs)
        delattr(self._detach_Qs, "params")
        self._detach_Qs.__dict__["params"] = self._detach_Qs_params
        delattr(self._target_Qs, "params")
        self._target_Qs.__dict__["params"] = self._target_Qs_params
        self._detach_Qs.refresh_functional_params()
        self._target_Qs.refresh_functional_params()
        self.__dict__["_mode_modules"] = tuple(
            (m for m in self.modules() if not isinstance(m, TensorDictParams))
        )
        self.__dict__["_target_mode_modules"] = tuple(self._target_Qs.modules())
        self.__dict__["_clip_parameters"] = tuple(self.parameters())
        self.__dict__["_polyak_pairs"] = tuple(
            zip(
                self._target_Qs._functional_params.values(),
                self._detach_Qs._functional_params.values(),
            )
        )

    def __repr__(self):
        repr = "TD-MPC2 World Model\n"
        modules = [
            "Encoder",
            "Dynamics",
            "Reward",
            "Termination",
            "Policy prior",
            "Q-functions",
        ]
        for i, m in enumerate(
            [
                self._encoder,
                self._dynamics,
                self._reward,
                self._termination,
                self._pi,
                self._Qs,
            ]
        ):
            if m == self._termination and (not self.cfg.episodic):
                continue
            repr += f"{modules[i]}: {m}\n"
        repr += "Learnable parameters: {:,}".format(self.total_params)
        return repr

    @property
    def total_params(self):
        return sum((p.numel() for p in self.parameters() if p.requires_grad))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        target = self._target_Qs_params.data.clone()
        self.init()
        self._target_Qs_params.data.copy_(target)
        self._target_Qs.train(False)
        return self

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode must be boolean")
        for module in self._mode_modules:
            module.training = mode
        for module in self._target_mode_modules:
            module.training = False
        return self

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        with torch.no_grad():
            for target, online in self._polyak_pairs:
                target.lerp_(online, self.cfg.tau)

    def encode(self, obs, task):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if self.cfg.obs == "rgb" and obs.ndim == 5:
            return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
        return self._encoder[self.cfg.obs](obs)

    def encode_features(self, obs):
        """Return node embeddings (if present) and the global latent."""
        return self._encoder["state"].forward_features(obs)

    def encode_actor_features(self, obs):
        return self.encode_features(obs)

    def next(self, z, a, task):
        """
        Predicts the next latent state given the current latent state and action.
        """
        if self._dynamics is None:
            raise RuntimeError("Dynamics disabled: no imagined rollout")
        z = torch.cat([z, a], dim=-1)
        return self._dynamics(z)

    def reward(self, z, a, task):
        """
        Predicts instantaneous (single-step) reward.
        """
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def termination(self, z, task, unnormalized=False):
        """
        Predicts termination signal.
        """
        assert task is None
        if unnormalized:
            return self._termination(z)
        return torch.sigmoid(self._termination(z))

    def pi(self, z, task=None, *, noise=None, generator=None, node_states=None):
        """Reparameterized one-step action, joint entropy and per-dimension entropy."""
        if task is not None:
            raise ValueError("single-task policy only")
        mean, raw = self._pi(z, node_states=node_states).chunk(2, dim=-1)
        log_std = self.policy_log_std(raw)
        if noise is None:
            noise = (
                torch.randn_like(mean)
                if generator is None
                else torch.randn(
                    mean.shape,
                    device=mean.device,
                    dtype=mean.dtype,
                    generator=generator,
                )
            )
        elif noise.shape != mean.shape or not torch.isfinite(noise).all():
            raise ValueError("Gaussian noise must be finite and match [...,action_dim]")
        return self._squashed_gaussian(mean, log_std, noise.to(mean))

    def policy_log_std(self, raw):
        """Bounded log std with algebraic tail gradients (no clamp or STE)."""
        return self.log_std_min + self.log_std_dif * (
            0.5 + torch.atan(raw) / 3.141592653589793
        )

    def _squashed_gaussian(self, mean, log_std, eps):
        low, high = (self.cfg.gaussian_action_low, self.cfg.gaussian_action_high)
        action, log_prob = math.squashed_gaussian_density(mean, log_std, eps, low, high)
        entropy = -log_prob
        return (
            action,
            dict(
                mean=math.affine_action(mean.tanh(), low, high),
                log_std=log_std,
                entropy=entropy,
                scaled_entropy=entropy / self.cfg.action_dim,
            ),
        )

    def Q(self, z, a, task, return_type="min", target=False, detach=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        if a.shape != (*z.shape[:-1], self.cfg.action_dim):
            raise ValueError("Q requires aligned single-step actions [...,D]")
        assert return_type in {"min", "avg", "all"}
        z = torch.cat([z, a], dim=-1)
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        out = qnet(z)
        if return_type == "all":
            return out
        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = math.two_hot_inv(out[qidx], self.cfg)
        if return_type == "min":
            return Q.min(0).values
        return Q.sum(0) / 2
