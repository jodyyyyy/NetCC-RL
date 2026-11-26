"""Opt-in EXP016 real G2 runner: GPU B32 physics/policy and per-slot H3 Replay.

The frozen warmup boundary follows the historical runner: transitions 1..5000
use random actions, and transition 5000 earns the first update. Earlier warmup
transitions never accrue update credit. Thus 1M collects exactly 995001 updates.
"""
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import torch
from tensordict import TensorDict
from .math import affine_action
import yaml

PROTOCOL = 'exp016_three_environments'
TASKS = {'mpcsc-kuramoto-control-cost', 'mpcsc-sis', 'mpcsc-sis-control-cost',
         'graph-swing'}


def validate_collector(cfg):
    backend = getattr(cfg, 'collector_backend', 'cpu_serial')
    batch = getattr(cfg, 'collector_batch_size', 1)
    protocol = getattr(cfg, 'collector_protocol', None)
    if backend == 'cpu_serial':
        if batch != 1 or protocol is not None:
            raise ValueError('serial runner requires collector_batch_size=1, protocol=null')
        return
    if backend != 'gpu' or protocol != PROTOCOL:
        raise ValueError('unknown GPU collector protocol')
    if type(batch) is not int or batch != 32 or not str(cfg.device).startswith('cuda'):
        raise ValueError('formal GPU collector requires CUDA and 32 parallel environment slots')
    required = dict(exp016_arm='G2', graph_global_readout='mean_max', actor_type='gaussian',
                    model_width_profile='exp016_adaptive_v1',
                    graph_backend='turbo', batch_size=512, horizon=3, utd_ratio=1,
                    initial_updates=0, warmup_steps=5000, graph_num_layers=2, graph_num_heads=4)
    if cfg.task not in TASKS or any(getattr(cfg, k, None) != v for k, v in required.items()):
        raise ValueError('formal GPU collector requires the three-environment G2 B512/H3/UTD1 protocol')
    limit = getattr(cfg, 'collector_pilot_steps', None)
    if limit is not None and (type(limit) is not int or not 5000 < limit <= cfg.steps):
        raise ValueError('pilot limit must exceed warmup and not exceed requested steps')
    if not (cfg.learner_cuda_graphs or cfg.compile):
        raise ValueError('formal GPU collector requires explicit CUDA Graph or compiled learner')


def next_count(step, total, batch, boundaries, warmup):
    """Stop before update enablement, eval/save/log and final boundaries exactly."""
    limits = [batch-step % batch, total-step]
    limits += [b-step % b for b in boundaries if b > 0]
    limits += [b-step for b in (warmup-1, warmup) if b > step]
    return min(limits)


class EpisodeStaging:
    """GPU rows only; no observation/action/reward copies during collection.

    CPU integer lengths mirror the simulator's fixed 150-step truncation. Only
    complete episodes and final live tails enter the unchanged Buffer API.
    """
    def __init__(self, collector, cfg):
        self.cfg, self.collector = cfg, collector
        b, t, d = cfg.collector_batch_size, cfg.episode_length, cfg.obs_shape['state'][0]
        dev = torch.device(cfg.device)
        self.lengths = [0]*b
        self.data = TensorDict({
            'obs': torch.zeros(b, t+1, d, device=dev),
            'action': torch.zeros(b, t+1, cfg.action_dim, device=dev),
            'reward': torch.zeros(b, t+1, device=dev),
            'terminated': torch.zeros(b, t+1, device=dev),
            'truncated': torch.zeros(b, t+1, device=dev, dtype=torch.bool),
        }, batch_size=(b, t+1), device=dev)
        self.data['obs'][:, 0] = collector.obs
        self.completed = 0

    def append(self, result, step, buffer):
        """Return diagnostics for complete episodes, never unfinished tails."""
        count = result['newly_collected_transitions']
        completed, returns = [], []
        slots = torch.arange(count, device=self.cfg.device)
        rows = torch.tensor([self.lengths[i]+1 for i in range(count)], device=self.cfg.device)
        for key, value in [('obs', result['after']), ('action', result['action']),
                           ('reward', result['reward']), ('terminated', result['terminated']),
                           ('truncated', result['truncated'])]:
            self.data[key][slots, rows] = value.to(self.data[key].dtype)
        for slot in range(count):
            self.lengths[slot] += 1
            if self.lengths[slot] == self.cfg.episode_length:
                buffer.add(self.data[slot].clone())
                self.completed += 1
                returns.append(self.data['reward'][slot, 1:].sum(dtype=torch.float64))
                completed.append(dict(episode=self.completed, slot=slot,
                                      episode_length=self.lengths[slot]))
                self.lengths[slot] = 0
                self.data['obs'][slot, 0] = self.collector.obs[slot]
        # One transfer per completed cohort; no return synchronization on steps
        # without completed episodes. Replay and learner tensors are unchanged.
        if completed:
            for record, value in zip(completed, torch.stack(returns).cpu().tolist()):
                record['episode_return'] = value
        return completed

    def flush(self, buffer):
        for slot, length in enumerate(self.lengths):
            if length:
                buffer.add(self.data[slot, :length+1].clone())


@torch.no_grad()
def evaluate_gpu(agent, cfg):
    """20 fresh GPU episodes; dedicated seed/noise streams, no training RNG use."""
    from .collector import VectorCollector
    other = deepcopy(cfg)
    other.seed = cfg.eval_seed
    collector = VectorCollector(other, cfg.eval_episodes)
    sim = collector.sim
    seeds = torch.arange(cfg.eval_episodes, device=cfg.device)+cfg.eval_seed
    # Each slot has the same key as an independent one-slot simulator at its seed.
    if hasattr(sim, 'keys'):
        sim.keys = seeds*104729
        sim.counter.zero_()
        sim.reset()
    else:
        sim.reset_keys = seeds*104729
        sim.reset_counter.zero_()
        sim.reset()
    collector.obs = sim.observe()
    generators = [torch.Generator(device=cfg.device).manual_seed(cfg.eval_seed+i)
                  for i in range(cfg.eval_episodes)]
    was_training = agent.model.training
    totals = torch.zeros(cfg.eval_episodes, device=cfg.device)
    previous_rng = agent.eval_generator.get_state()
    agent.model.eval()
    try:
        for _ in range(cfg.episode_length):
            noise = torch.stack([torch.randn(cfg.action_dim, device=cfg.device, generator=g)
                                 for g in generators])
            action = agent.act(collector.obs, eval_mode=True, noise=noise, return_cpu=False)
            result = collector.step(action)
            totals += result['reward']
        values = totals.cpu().tolist()
    finally:
        collector.close()
        agent.eval_generator.set_state(previous_rng)
        agent.model.train(was_training)
    return dict(task=cfg.task, eval_policy=cfg.eval_policy, eval_seed=cfg.eval_seed,
                evaluation_backend='gpu_vector_independent_seed_v1', episode_return=sum(values)/len(values),
                episodes=[dict(seed=cfg.eval_seed+i, length=cfg.episode_length, terminated=0.,
                               **{'return': value}) for i, value in enumerate(values)])


def train_vector(agent, cfg, *, load_report=None, writer):
    from .buffer import Buffer
    from .collector import VectorCollector
    from .timing import TrainingTimer, runtime_metadata
    from .utd import UpdateBudget
    from ..config import actor_metadata, validate, experiment_name
    from ..runner import write_json, _pack_metrics
    validate(cfg)
    out = Path(cfg.work_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any((out/name).exists() for name in ('config.yaml', 'final.pt', 'metrics.jsonl')):
        raise FileExistsError(f'training output already exists: {out}')
    (out/'config.yaml').write_text(yaml.safe_dump(vars(cfg), sort_keys=False))
    write_json(out/'runtime.json', runtime_metadata(cfg.device))
    write_json(out/'run-manifest.json', dict(experiment_name=experiment_name(cfg), **actor_metadata(cfg)))
    if load_report:
        write_json(out/'warm_start.json', load_report)
    torch.cuda.reset_peak_memory_stats(cfg.device)
    collector = VectorCollector(cfg, cfg.collector_batch_size)
    if not collector.obs.is_cuda or not collector.sim.state.is_cuda or collector.envs:
        raise RuntimeError('GPU simulation required; no CPU fallback')
    staging = EpisodeStaging(collector, cfg)
    buffer, budget = Buffer(cfg), UpdateBudget(cfg.utd_ratio)
    timer = TrainingTimer(cfg, out/'timing.jsonl', writer=writer)
    random = torch.Generator(device=cfg.device).manual_seed(cfg.seed+90173)
    target = getattr(cfg, 'collector_pilot_steps', None) or cfg.steps
    latest, step = {}, 0
    window_start, window_updates = None, 0
    boundaries = []

    def log(kind, values):
        writer.submit('metrics.jsonl', dict(kind=kind, step=step, updates=budget.updates,
            collector_batch_size=cfg.collector_batch_size, **budget.diagnostics(), **values))

    def evaluation():
        with timer.measure('evaluation', force=True):
            result = evaluate_gpu(agent, cfg)
        log('eval', result)
        write_json(out/'evaluation.json', result)

    try:
        evaluation()
        while step < target:
            writer.check()
            count = next_count(step, target, cfg.collector_batch_size,
                               [cfg.log_freq, cfg.eval_freq, cfg.save_freq], cfg.warmup_steps)
            timer.select(step+count, buffer)
            with timer.measure('main_loop'):
                with timer.measure('policy'):
                    if step < cfg.warmup_steps:
                        action = torch.rand(cfg.collector_batch_size, cfg.action_dim,
                                            device=cfg.device, generator=random)*2-1
                        action = affine_action(action, cfg.gaussian_action_low, cfg.gaussian_action_high)
                    else:
                        # Partial tails generate actions only for the active slots.
                        active_action = agent.act(collector.obs[:count], return_cpu=False)
                        action = torch.zeros(cfg.collector_batch_size, cfg.action_dim, device=cfg.device)
                        action[:count] = active_action
                with timer.measure('environment_interaction'):
                    result = collector.step(action, count)
                with timer.measure('replay_add'):
                    completed_episodes = staging.append(result, step, buffer)
                # Transition 5000 is random but earns the first post-warmup update,
                # exactly as in the serial runner; transitions <5000 earn nothing.
                budget.collect(count, enabled=step+count >= cfg.warmup_steps)
                step += count
                for episode in completed_episodes:
                    log('episode', episode)
                if step == target:
                    with timer.measure('replay_add', force=True):
                        staging.flush(buffer)
                while buffer.num_eps and budget.credit >= 1:
                    latest = agent.update(buffer, timer=timer) if timer.active else agent.update(buffer)
                    if not bool(torch.isfinite(_pack_metrics(latest)).all()):
                        raise FloatingPointError('nonfinite training metric')
                    budget.updated()
                if step == cfg.warmup_steps:
                    torch.cuda.synchronize(cfg.device)
                    window_start, window_updates = perf_counter(), budget.updates
                if step % cfg.log_freq == 0 or step == target:
                    log('train', {**dict(zip(latest, _pack_metrics(latest).cpu().tolist())),
                                  **buffer.diagnostics()})
                if step % cfg.eval_freq == 0 or step == target:
                    boundaries.append(dict(step=step, kind='evaluation', partial_count=count))
                    evaluation()
                if step % cfg.save_freq == 0:
                    boundaries.append(dict(step=step, kind='checkpoint', partial_count=count))
                    with timer.measure('checkpoint', force=True):
                        agent.save(out/f'step_{step}.pt')
        torch.cuda.synchronize(cfg.device)
        measured_seconds = perf_counter()-window_start if window_start else None
        with timer.measure('checkpoint', force=True):
            agent.save(out/'final.pt')
        expected_updates = max(0, target-max(1, cfg.warmup_steps)+1)
        if budget.collected != target or budget.updates != expected_updates or budget.credit:
            raise RuntimeError('GPU runner exact transition/update accounting failed')
        runtime = agent.__dict__.get('_learner_runtime')
        if cfg.compile:
            from torch._dynamo.utils import counters
            stats = dict(counters['stats'])
            if runtime is not None or not hasattr(agent._update, '_torchdynamo_orig_callable') or not stats.get('unique_graphs', 0):
                raise RuntimeError('explicit compiled learner did not execute compiled graphs')
            acceleration = dict(mode='torch_compile', precision='fp32', graphs=False,
                                compiler_stats=stats)
        else:
            if runtime is None or runtime.counts['eager_shape_fallbacks']:
                raise RuntimeError('real G2 CUDA graph learner required without eager fallback')
            acceleration = runtime.metadata()
        return dict(steps=step, requested_steps=cfg.steps, pilot=target!=cfg.steps, updates=budget.updates, episodes=staging.completed,
                    collector_backend='gpu', collector_batch_size=cfg.collector_batch_size,
                    collector_protocol=PROTOCOL, boundaries=boundaries,
                    environment_device=str(collector.sim.state.device),
                    model_device=str(next(agent.model.parameters()).device),
                    replay_transfer_policy='GPU episode staging; Buffer.add metadata scan to CPU at episode boundaries only; storage auto-selected',
                    **budget.diagnostics(), **buffer.diagnostics(), **timer.summary(step),
                    measured_window=dict(transitions=max(0, step-cfg.warmup_steps),
                        updates=budget.updates-window_updates, seconds=measured_seconds),
                    learner_acceleration=acceleration, checkpoint=str(out/'final.pt'),
                    training_resume_supported=False)
    finally:
        collector.close()
