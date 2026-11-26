"""MAT/CMAT/JointPPO task extension retaining AR / consensus and PPO objectives."""
import torch
from torch import nn
from .joint_policy import Compression, incremental_consensus
from .joint_train import Learner,load_profile
from .task_adapter import GaussianGroup

class TaskJointPolicy(nn.Module):
    def __init__(self,algorithm,adapter,profile):
        super().__init__();self.algorithm=algorithm;self.adapter=adapter;self.width=profile['width']
        self.consensus_backend = "incremental"
        self.consensus_steps = profile["consensus_steps"]
        w, h, blocks = self.width, profile["heads"], profile["blocks"]
        self.embedding = nn.Linear(adapter.features, w)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(w, h, 2*w, dropout=0., activation="gelu", batch_first=True),
            blocks, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(w, h, 2*w, dropout=0., activation="gelu", batch_first=True), blocks)
        self.action_head = nn.Linear(w, adapter.group_width)
        self.log_std = nn.Parameter(torch.ones(adapter.agents, adapter.group_width))
        self.critic = nn.Sequential(nn.Linear(w,w),nn.GELU(),nn.Linear(w,1))
        self.register_buffer("drivers", torch.tensor(adapter.controls))
        self.critic_compressor = Compression(w)
        self.actor_compressor = Compression(w)
        self.latent_projection = nn.Linear(w,w)
        self.actor_context = nn.Sequential(nn.Linear(2*w,w),nn.GELU())

    def encode(self, raw):
        return self.encoder(self.embedding(self.adapter.features_tensor(raw)))

    def components(self, memory):
        initial = self.critic_compressor(memory)
        trajectory = incremental_consensus(
            self.decoder, memory, initial, self.latent_projection, self.consensus_steps)
        consensus = self.actor_compressor(trajectory)
        own = memory[:, self.drivers]
        context = self.actor_context(torch.cat([own, consensus[:, None].expand_as(own)], -1))
        return context, self.critic(initial)

    def distribution(self, context):
        mean = self.action_head(context)
        std = self.log_std.sigmoid() * .5
        return GaussianGroup(mean, std.log().expand_as(mean), self.adapter.low, self.adapter.high,
                             self.adapter.mask(mean.device))

    def forward(self, raw, action):
        context, values = self.components(self.encode(raw))
        groups = self.adapter.grouped(action)
        dist = self.distribution(context)
        return dist.log_prob(groups).sum(-1), values, dist.entropy().sum(-1)

    @torch.no_grad()
    def act(self, raw, deterministic=False):
        context, values = self.components(self.encode(raw))
        dist = self.distribution(context)
        grouped = dist.mean if deterministic else dist.sample()
        action = self.adapter.flatten(grouped)
        return action, action, dist.log_prob(grouped).sum(-1), values

    @torch.no_grad()
    def predict(self, obs):
        device = next(self.parameters()).device
        raw = torch.as_tensor(obs, device=device, dtype=torch.float32).reshape(1, -1)
        return self.adapter.policy_to_canonical(self.act(raw, True)[0][0]).cpu().numpy()


class TaskJointLearner(Learner):
    def __init__(self,algorithm,adapter,device='cpu', *, profile=None):
        self.algorithm=algorithm;self.adapter=adapter;self.device=torch.device(device)
        self.profile=load_profile(algorithm) if profile is None else dict(profile)
        self.policy=TaskJointPolicy(algorithm,adapter,self.profile).to(device)
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=self.profile['learning_rate'],eps=1e-5)
        self.updates=0
    def training_state(self):
        state=dict(super().training_state(),task_identity=self.adapter.identity)
        if self.algorithm=='cmat':state['consensus_backend']=self.policy.consensus_backend
        return state
    def restore(self,state):
        if state['task_identity']!=self.adapter.identity:raise ValueError('task/checkpoint identity differs')
        if self.algorithm=='cmat' and 'consensus_backend' in state and state['consensus_backend']!=self.policy.consensus_backend:
            raise ValueError('CMAT consensus backend differs; exact continuation is not cross-backend')
        super().restore(state)
