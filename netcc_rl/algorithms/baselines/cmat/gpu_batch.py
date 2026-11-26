"""Real device-native collectors for the three continuous EXP-016 tasks.

The simulator consumes canonical environment actions, never Gaussian latents.
SB3's NumPy VecEnv boundary is explicit; its physics still executes on CUDA.
"""
import torch


class ContinuousBatch:
    def __init__(self, adapter, batch_size=32, device='cuda:0', seed=0):
        if adapter.name not in ('kuramoto', 'sis', 'swing'):
            raise ValueError('only the three approved continuous tasks')
        self.adapter, self.batch_size = adapter, batch_size
        self.device = torch.device(device)
        self.max_episode_steps = adapter.snapshot['max_episode_steps'] if adapter.name != 'swing' else adapter.snapshot['params']['episode_length']
        if adapter.name == 'swing':
            from netcc_rl.environments.graph.env import GraphDynamics
            self.sim = GraphDynamics('graph-swing', snapshot=adapter.snapshot, batch_size=batch_size, device=device, seed=seed)
        else:
            from netcc_rl.environments.mpcsc.batched import BatchedMPCSC
            self.sim = BatchedMPCSC(adapter.snapshot, batch_size, device=device, seed=seed)
        self.transitions = 0

    def reset(self):
        return self.sim.reset()

    def step(self, actions):
        if actions.shape != (self.batch_size, self.adapter.action_dim):
            raise ValueError('canonical batch action shape')
        if not torch.isfinite(actions).all():
            raise ValueError('nonfinite environment action')
        # Batched MPCSC receives normalized canonical actions; SIS applies its
        # native (action+1)/2 mapping exactly once inside the simulator.
        result = self.sim.step(actions.to(self.device))
        self.transitions += self.batch_size
        return result

    def reset_done(self, mask):
        return self.sim.reset(mask=mask) if self.adapter.name == 'swing' else self.sim.reset(mask)

    def close(self):
        pass
