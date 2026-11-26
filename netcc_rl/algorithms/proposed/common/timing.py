"""Low-frequency wall-clock profiling; CUDA barriers only in selected windows.

Main-loop windows include logging and scheduled evaluation/checkpoint work.
All episode adds, evaluations and saves are timed separately, even outside a
window. These barriers perturb throughput; no timing is an unperturbed claim.
"""
from contextlib import contextmanager
import json
from time import perf_counter

import numpy as np
import torch


class TrainingTimer:
    def __init__(self, cfg, path, *, writer=None):
        self.device = torch.device(cfg.device)
        self.interval, self.window = cfg.profile_interval, cfg.profile_window
        self.path = path
        self.writer = writer
        self.started = perf_counter()
        self.active = False
        self.step = 0
        self.stage = 'initial'
        self.occupancy = 0.0
        self.stage_counts = {}
        self.rows = []
        self.sync_count = 0
        self.sync_seconds = 0.

    def select(self, step, buffer):
        self.step = step
        self.occupancy = buffer.num_transitions / buffer.capacity
        self.stage = ('near_full' if self.occupancy >= .9 else
                      'middle' if self.occupancy >= .5 else 'initial')
        ready = buffer.num_transitions > 0 and step >= buffer.cfg.warmup_steps
        count = self.stage_counts.get(self.stage, 0)
        self.active = ready and (count < self.window or (step % self.interval) < self.window)
        if self.active:
            self.stage_counts[self.stage] = count + 1
        return self.active

    def sync(self):
        if self.device.type == 'cuda':
            started = perf_counter()
            torch.cuda.synchronize(self.device)
            self.sync_count += 1
            self.sync_seconds += perf_counter()-started

    @contextmanager
    def measure(self, operation, *, force=False):
        if not (self.active or force):
            yield
            return
        self.sync()
        start = perf_counter()
        try:
            yield
        finally:
            self.sync()
            row = dict(step=self.step, stage=self.stage, occupancy=self.occupancy,
                       operation=operation, seconds=perf_counter()-start)
            self.rows.append(row)
            if self.writer is not None:
                self.writer.submit(self.path.name, row)
            else:
                with self.path.open('a') as handle:
                    handle.write(json.dumps(row, allow_nan=False)+'\n')

    def summary(self, steps):
        elapsed = perf_counter()-self.started
        groups = {}
        for row in self.rows:
            groups.setdefault((row['stage'], row['operation']), []).append(row['seconds'])
        stats = [dict(stage=stage, operation=op, count=len(values),
                      mean_seconds=float(np.mean(values)), p50_seconds=float(np.percentile(values, 50)),
                      p95_seconds=float(np.percentile(values, 95)), total_seconds=sum(values))
                 for (stage, op), values in sorted(groups.items())]
        ratios = {}
        for stage in self.stage_counts:
            total = lambda op: sum(groups.get((stage, op), []))
            sample, learner, loop = total('sample'), total('learner_update'), total('main_loop')
            ratios[stage] = dict(sample_over_sample_plus_learner=sample/(sample+learner) if sample+learner else None,
                                 sample_over_profiled_main_loop=sample/loop if loop else None)
        return dict(wall_seconds=elapsed, steps_per_second=steps/elapsed, timing_statistics=stats,
                    timing_ratios=ratios, timing_method='synchronized_wall_clock_windows',
                    profile_interval=self.interval, profile_window=self.window,
                    timing_perturbs_throughput=True,
                    profiler_cuda_barrier_count=self.sync_count, profiler_cuda_barrier_seconds=self.sync_seconds,
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None,
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if self.device.type == 'cuda' else None)


def runtime_metadata(device):
    """Bind a run to the actual checkout, dependency versions and hardware."""
    from importlib.metadata import version
    import os
    import platform
    from pathlib import Path
    import subprocess
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    device = torch.device(device)
    return dict(git_sha=git('rev-parse', 'HEAD'), git_status=git('status', '--porcelain'),
                python=platform.python_version(), platform=platform.platform(), cpu=platform.processor(),
                dependencies={name: version(name) for name in ('torch','torchrl','tensordict','numpy','gymnasium','PyYAML')},
                cuda_runtime=torch.version.cuda, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                device=str(device), gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                gpu_total_bytes=torch.cuda.get_device_properties(device).total_memory if device.type == 'cuda' else None,
                torch_threads=torch.get_num_threads())
