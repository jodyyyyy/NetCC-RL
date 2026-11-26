"""Historical B32 CMAT trainer and deterministic policy evaluation."""

from pathlib import Path
import time

import numpy as np
import torch

from ..runtime import rng, restore_rng, seed_everything
from netcc_rl.comparison_output import append_json, checkpoint_index, evaluation, progress_record, runtime_record, training_row, write_json

from .gpu_batch import ContinuousBatch
from .joint_train import Learner
from .task_adapter import TaskAdapter
from .task_joint import TaskJointLearner


def build(config, device):
    adapter = TaskAdapter(config)
    if adapter.name == "kuramoto":
        learner = Learner("cmat", config["profile"], device, controls=adapter.controls)
    else:
        learner = TaskJointLearner("cmat", adapter, device, profile=config["profile"])
    return adapter, learner


@torch.no_grad()
def collect(adapter, learner, collector, length):
    state = collector.reset()
    rows = {key: [] for key in ("states", "latent", "logp", "values", "next_values",
                                "rewards", "terminated", "truncated")}
    for step in range(length):
        action, latent, logp, values = learner.policy.act(state)
        if adapter.name != "kuramoto":
            action = adapter.policy_to_canonical(action)
        if not torch.isfinite(action).all() or (action < adapter.low - 1e-6).any() or (action > adapter.high + 1e-6).any():
            raise ValueError("CMAT action outside its frozen protocol")
        nxt, reward, terminal, truncated, _ = collector.step(action)
        memory = learner.policy.encode(nxt)
        next_values = learner.policy.critic(learner.policy.critic_compressor(memory))
        fields = (state, latent, logp, values, next_values, reward, terminal, truncated)
        for key, value in zip(rows, fields, strict=True):
            rows[key].append(value.detach())
        if terminal.any() or (truncated.any() and step != collector.max_episode_steps - 1):
            raise ValueError("unexpected boundary in synchronous CMAT collector")
        state = nxt
    if bool(truncated.all()) != (length == collector.max_episode_steps):
        raise ValueError("CMAT episode boundary disagrees with horizon")
    packed = {key: list(torch.stack(value).cpu().transpose(0, 1).flatten(0, 1))
              for key, value in rows.items()}
    packed["episode_lengths"] = [length] * collector.batch_size
    return packed


def evaluate(adapter, learner, config, episodes, seed_start):
    saved_rng = rng()
    env = adapter.make_env()
    rows = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            obs, _ = env.reset(seed=seed)
            total = 0.0
            for step in range(env.max_episode_steps):
                obs, reward, terminal, truncated, _ = env.step(learner.policy.predict(obs))
                total += reward
                if terminal or truncated:
                    break
            if not np.isfinite(total):
                raise FloatingPointError("nonfinite CMAT return")
            rows.append({'seed':seed, 'return':total, 'transitions':step+1,
                         'terminated':bool(terminal), 'truncated':bool(truncated)})
    finally:
        env.close()
        restore_rng(saved_rng)
    return evaluation(config, rows)


def episode_summaries(data):
    offset = 0
    for count in data["episode_lengths"]:
        end = offset + count
        yield {"return":float(torch.stack(data["rewards"][offset:end]).sum()),
               "transitions":count, "terminated":bool(data["terminated"][end-1]),
               "truncated":bool(data["truncated"][end-1])}
        offset = end


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(config, output, device, seed, smoke, *, resume=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(seed)
    adapter, learner = build(config, device)
    output = Path(output)
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", config)
    budget = 4 if smoke else config["steps"]
    batch = config["collector_batch_size"]
    collector = ContinuousBatch(adapter, batch, device, seed)
    horizon = collector.max_episode_steps
    completed, next_eval, next_checkpoint = 0, 50_000, 100_000
    runtime = runtime_record(config,seed,device,str(torch.__version__),budget,smoke)
    last_good = output / "last-good.pt"
    pending = None
    gathered = 0
    stage = "initializing"
    start = time.time()
    last_status = 0.0

    def progress(force=False):
        nonlocal last_status
        now = time.time()
        if force or now - last_status >= 15:
            last_status = now
            write_json(output / "progress.json", progress_record(config,stage,completed,budget,
                       gathered=gathered if pending is not None else 0, learner_updates=learner.updates,
                       updated_at=now, wall_seconds=now-start))

    def checkpoint():
        sim = {k: v.detach().cpu() for k, v in vars(collector.sim).items() if torch.is_tensor(v)}
        save_state(last_good, dict(runtime=runtime, config=config, learner=learner.training_state(), rng=rng(),
                                  sim=sim, transitions=collector.transitions, pending=pending,
                                  gathered=gathered, completed=completed, next_eval=next_eval,
                                  next_checkpoint=next_checkpoint))

    if resume:
        saved = torch.load(last_good, map_location="cpu", weights_only=False)
        if saved["runtime"] != runtime or saved["config"] != config:
            raise ValueError("CMAT runtime identity differs from the saved training state")
        learner.restore(saved["learner"])
        for key, value in saved["sim"].items():
            setattr(collector.sim, key, value.to(collector.device).clone())
        collector.transitions = saved["transitions"]
        pending, gathered, completed = (saved[k] for k in ("pending", "gathered", "completed"))
        next_eval, next_checkpoint = saved["next_eval"], saved["next_checkpoint"]
        restore_rng(saved["rng"])
        for name in ("metrics.jsonl", "evaluation.jsonl"):
            path = output / name
            if path.exists():
                import json
                rows = path.read_text().splitlines(keepends=True)
                committed = [line for line in rows if json.loads(line)["real_steps"] <= completed]
                temporary = path.with_suffix(".tmp")
                temporary.write_text("".join(committed))
                temporary.replace(path)
    else:
        write_json(output / "runtime.json", runtime)
        checkpoint()
    progress(True)

    # A pending checkpoint precedes each update wave. If interrupted inside an
    # update, resume repeats the entire wave from its saved RNG and Adam state.
    heartbeat = learner.optimizer.register_step_post_hook(lambda *_: progress())
    while completed < budget:
        if pending is None:
            stage = "collecting"
            progress(True)
            remaining = budget - completed
            if remaining >= batch * horizon:
                pending = collect(adapter, learner, collector, horizon)
                gathered = batch * horizon
            else:
                full, tail = divmod(remaining, horizon)
                pieces = []
                gathered = 0
                for size, length in ((full, horizon), (int(tail > 0), tail)):
                    if size:
                        segment = ContinuousBatch(adapter, size, device, 1_000_003 + completed + gathered + seed)
                        pieces.append(collect(adapter, learner, segment, length))
                        gathered += size * length
                pending = {key: sum((piece[key] for piece in pieces), []) for key in pieces[0]}
            checkpoint()
        stage = "learning"
        progress(True)
        before = [p.detach().clone() for p in learner.policy.parameters()] if smoke else None
        metrics = learner.update(pending)
        if before is not None and not any(not torch.equal(p, b) for p, b in zip(learner.policy.parameters(), before)):
            raise RuntimeError("CMAT learner parameters did not change")
        completed += gathered
        append_json(output / "metrics.jsonl", training_row(config, completed, metrics,
                    learner_updates=learner.updates, episodes=episode_summaries(pending)))
        pending = None
        if completed >= next_eval or completed == budget:
            stage = "evaluating"
            progress(True)
            final = completed == budget
            result = evaluate(adapter, learner, config, 1 if smoke else (50 if final else 20), 202600)
            append_json(output / 'evaluation.jsonl', dict(real_steps=completed, final=final, **result))
            if final:
                write_json(output / 'evaluation.json', result)
            next_eval += 50_000
        if completed >= next_checkpoint or completed == budget:
            path = output / ("final.pt" if completed == budget else f"checkpoint-{completed:09d}.pt")
            save_state(path, dict(config=config, learner=learner.training_state(), real_steps=completed))
            if completed == budget:
                checkpoint_index(config, output, path, completed, seed)
            next_checkpoint += 100_000
        checkpoint()
    heartbeat.remove()
    if not smoke and learner.updates != config["implementation"]["expected_optimizer_steps"]:
        raise RuntimeError("CMAT optimizer budget differs from the frozen run")
    stage = "completed"
    progress(True)


def evaluate_checkpoint(config, checkpoint, device, episodes, seed_start):
    adapter, learner = build(config, device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    original = saved["config"]
    if "schema_version" not in original:
        implementation = config["implementation"]
        legacy = dict(config)
        legacy.pop("schema_version")
        legacy.pop("action_semantics")
        legacy.pop("implementation")
        legacy.update(implementation, snapshot_sha256=implementation["adapter_identity"]["snapshot_sha256"])
        matches = original == legacy
    else:
        matches = original == config
    if not matches:
        raise ValueError("CMAT checkpoint configuration identity differs")
    learner.restore(saved["learner"])
    return evaluate(adapter, learner, config, episodes, seed_start)
