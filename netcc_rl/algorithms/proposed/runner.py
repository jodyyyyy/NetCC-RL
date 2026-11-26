"""Single-task collection and reproducible actor-only evaluation on MPCSC."""

from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import torch
from tensordict import TensorDict
import yaml

from netcc_rl.environments import make_env

from .common.buffer import Buffer
from .common.math import affine_action
from .common.writer import JsonlWriter
from .common.timing import TrainingTimer, runtime_metadata
from .config import actor_metadata, experiment_name, validate_experiment_output


ACTION_METRICS = ("act_latency_ms",)
REWARD_METRICS = (
    "infection_reward",
    "synchronization_reward",
    "control_cost",
    "total_reward",
    "mean_control_intensity",
    "mean_squared_action",
    "sum_squared_action",
)


def set_seed(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)


def record(obs, action=None, reward=None, terminated=None, *, action_dim):
    """Store incoming single-step transition on its resulting observation."""
    if action is None:
        action = torch.full((action_dim,), float("nan"))
        reward = terminated = torch.tensor(float("nan"))
    return TensorDict(
        dict(
            obs=obs.detach().cpu().unsqueeze(0),
            action=action.detach().cpu().unsqueeze(0),
            reward=torch.as_tensor(reward).cpu().reshape(1),
            terminated=torch.as_tensor(terminated).cpu().reshape(1),
        ),
        batch_size=(1,),
    )


def evaluate(agent, cfg):
    """Fresh environment and dedicated action-noise stream for every evaluation."""
    env = make_env(deepcopy(cfg))
    episodes = []
    was_training = agent.model.training
    previous_rng = agent.eval_generator.get_state()
    agent.model.eval()
    try:
        for index in range(cfg.eval_episodes):
            seed = cfg.eval_seed + index
            agent.reset_eval_seed(seed)
            obs = env.reset(seed=seed)
            total, action_metrics, reward_metrics = 0.0, [], []
            for step in range(env.max_episode_steps):
                action = agent.act(obs, eval_mode=True)
                obs, reward, done, info = env.step(action)
                total += float(reward)
                reward_metrics.append(
                    {key: float(info[key]) for key in REWARD_METRICS if key in info}
                )
                action_metrics.append(agent.last_act_info.copy())
                if done:
                    break
            episode = {
                "seed": seed,
                "return": total,
                "length": step + 1,
                "terminated": float(info["terminated"]),
            }
            episode.update(
                {
                    key: float(
                        np.mean([item[key] for item in action_metrics if key in item])
                    )
                    for key in ACTION_METRICS
                    if any(key in item for item in action_metrics)
                }
            )
            for key in REWARD_METRICS:
                values = [item[key] for item in reward_metrics if key in item]
                if values:
                    episode[key] = float(
                        np.mean(values)
                        if key in ("mean_control_intensity", "mean_squared_action")
                        else np.sum(values)
                    )
            episodes.append(episode)
    finally:
        env.close()
        agent.eval_generator.set_state(previous_rng)
        agent.model.train(was_training)
    return {
        "task": cfg.task,
        "actor_type": cfg.actor_type,
        "mpc": False,
        **actor_metadata(cfg),
        "eval_policy": cfg.eval_policy,
        "eval_seed": cfg.eval_seed,
        "episode_return": float(np.mean([ep["return"] for ep in episodes])),
        "episodes": episodes,
    }


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _pack_metrics(metrics):
    if not metrics:
        return torch.empty(0)
    device = next(
        (v.device for v in metrics.values() if torch.is_tensor(v)), torch.device("cpu")
    )
    return torch.stack(
        [
            torch.as_tensor(v, device=device).float().reshape(())
            for v in metrics.values()
        ]
    )


def retain_best(agent, cfg, evaluation, step, best):
    """Keep earliest maximum evaluation-return weights; opt-in, including step0."""
    if not getattr(cfg, "retain_best_checkpoint", False):
        return best
    score = float(evaluation["episode_return"])
    if not np.isfinite(score):
        raise FloatingPointError("nonfinite best-checkpoint selection metric")
    if best is None or score > best:
        path = Path(cfg.work_dir) / "best.pt"
        agent.save(path)
        write_json(
            Path(cfg.work_dir) / "best.json",
            dict(
                step=step,
                metric="mean undiscounted evaluation episode return",
                score=score,
                checkpoint=str(path),
                tie_break="earliest",
                includes_untrained_step0=True,
                training_resume_supported=False,
            ),
        )
        return score
    return best


def _train(agent, env, cfg, *, load_report=None, writer=None):
    validate_experiment_output(cfg)
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    # Refuse to mix metrics/checkpoints from separate runs.
    if any(
        (work_dir / name).exists()
        for name in ("metrics.jsonl", "final.pt", "config.yaml")
    ):
        raise FileExistsError(f"training output already exists: {work_dir}")
    with (work_dir / "config.yaml").open("w") as handle:
        yaml.safe_dump(vars(cfg), handle, sort_keys=False)
    write_json(
        work_dir / "runtime.json",
        {**runtime_metadata(cfg.device), **actor_metadata(cfg)},
    )
    if getattr(cfg, "exp016_arm", None):
        write_json(
            work_dir / "run-manifest.json",
            {
                "experiment_name": experiment_name(cfg),
                "work_dir": str(work_dir.resolve()),
                **actor_metadata(cfg),
                "runtime": runtime_metadata(cfg.device),
                "config": "config.yaml",
            },
        )
    if load_report:
        write_json(work_dir / "warm_start.json", load_report)
    if torch.device(cfg.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(cfg.device))
    buffer = Buffer(cfg)
    timer = TrainingTimer(cfg, work_dir / "timing.jsonl", writer=writer)
    pretraining_done, update_count, episode_idx = False, 0, 0
    latest, action_metrics = {}, []
    from .common.utd import UpdateBudget

    budget = (
        UpdateBudget(cfg.utd_ratio)
        if getattr(cfg, "utd_ratio", None) is not None
        else None
    )

    def log(kind, step, values):
        data = {
            "kind": kind,
            "step": step,
            "updates": update_count,
            **actor_metadata(cfg),
            **buffer.diagnostics(),
            **(budget.diagnostics() if budget else {}),
            **values,
        }
        writer.submit("metrics.jsonl", data)
        print(json.dumps(data, allow_nan=False), flush=True)

    with timer.measure("evaluation", force=True):
        evaluation = evaluate(agent, cfg)
    log("eval", 0, evaluation)
    best_return = retain_best(agent, cfg, evaluation, 0, None)
    obs = env.reset(seed=cfg.seed)
    episode = [record(obs, action_dim=cfg.action_dim)]
    episode_return = 0.0
    episode_reward_metrics = []
    for step in range(1, cfg.steps + 1):
        writer.check()
        timer.select(step, buffer)
        with timer.measure("main_loop"):
            with timer.measure("environment_interaction"):
                action = env.rand_act() if step <= cfg.warmup_steps else agent.act(obs)
                obs, reward, done, info = env.step(action)
            if step > cfg.warmup_steps:
                action_metrics.append(agent.last_act_info.copy())
            if info["terminated"] and not cfg.episodic:
                raise ValueError("termination detected; episodic=true is required")
            episode.append(
                record(
                    obs, action, reward, info["terminated"], action_dim=cfg.action_dim
                )
            )
            episode_return += float(reward)
            episode_reward_metrics.append(
                {key: float(info[key]) for key in REWARD_METRICS if key in info}
            )
            # Count the just-completed interaction before any episode log. A
            # completed episode will enter Replay below before updates execute.
            enabled = step >= cfg.warmup_steps and bool(buffer.num_eps or done)
            scheduled = budget.collect(1, enabled=enabled) if budget else 0
            if done:
                with timer.measure("replay_add", force=True):
                    buffer.add(torch.cat(episode))
                episode_idx += 1
                reward_summary = {}
                for key in REWARD_METRICS:
                    values = [
                        item[key] for item in episode_reward_metrics if key in item
                    ]
                    if values:
                        reward_summary[key] = float(
                            np.mean(values)
                            if key in ("mean_control_intensity", "mean_squared_action")
                            else np.sum(values)
                        )
                log(
                    "episode",
                    step,
                    {
                        "episode": episode_idx,
                        "episode_return": episode_return,
                        "episode_length": len(episode) - 1,
                        **reward_summary,
                    },
                )
                with timer.measure("environment_reset"):
                    obs = env.reset()
                episode = [record(obs, action_dim=cfg.action_dim)]
                episode_return = 0.0
                episode_reward_metrics = []
            if enabled:
                num_updates = scheduled if budget else cfg.updates_per_step
                if not pretraining_done:
                    num_updates += cfg.initial_updates
                    pretraining_done = True
                for _ in range(num_updates):
                    metrics = (
                        agent.update(buffer, timer=timer)
                        if timer.active
                        else agent.update(buffer)
                    )
                    latest = metrics
                    with timer.measure("metric_check"):
                        if not bool(torch.isfinite(_pack_metrics(latest)).all()):
                            raise FloatingPointError("nonfinite training metric")
                    update_count += 1
                    if budget:
                        budget.updated()
            if step % cfg.log_freq == 0 or step == cfg.steps:
                action_info = {
                    key: float(
                        np.mean([item[key] for item in action_metrics if key in item])
                    )
                    for key in ACTION_METRICS
                    if any(key in item for item in action_metrics)
                }
                with timer.measure("metrics_to_cpu"):
                    latest_cpu = dict(zip(latest, _pack_metrics(latest).cpu().tolist()))
                log("train", step, {**latest_cpu, **action_info})
                action_metrics.clear()
            if step % cfg.eval_freq == 0 or step == cfg.steps:
                with timer.measure("evaluation", force=True):
                    evaluation = evaluate(agent, cfg)
                log("eval", step, evaluation)
                write_json(work_dir / "evaluation.json", evaluation)
                best_return = retain_best(agent, cfg, evaluation, step, best_return)
            if step % cfg.save_freq == 0:
                with timer.measure("checkpoint", force=True):
                    agent.save(work_dir / f"step_{step}.pt")
    with timer.measure("checkpoint", force=True):
        agent.save(work_dir / "final.pt")
    result = {
        "steps": cfg.steps,
        "updates": update_count,
        "episodes": episode_idx,
        **(budget.diagnostics() if budget else {}),
        "actor_type": cfg.actor_type,
        "mpc": False,
        **actor_metadata(cfg),
        **buffer.diagnostics(),
        **timer.summary(cfg.steps),
        "checkpoint": str(work_dir / "final.pt"),
        "training_resume_supported": False,
    }
    runtime = agent.__dict__.get("_learner_runtime")
    if runtime is not None:
        result["learner_acceleration"] = runtime.metadata()
    return result


def train(agent, env, cfg, *, load_report=None):
    """Drain accepted events on success or failure; publish summary after join."""
    validate_experiment_output(cfg)
    work_dir = Path(cfg.work_dir)
    if any(
        (work_dir / name).exists()
        for name in ("metrics.jsonl", "final.pt", "config.yaml")
    ):
        raise FileExistsError(f"training output already exists: {work_dir}")
    from time import perf_counter

    started = perf_counter()
    writer = JsonlWriter(work_dir)
    try:
        with writer:
            if getattr(cfg, "collector_backend", "cpu_serial") == "gpu":
                from .common.vector_runner import train_vector

                result = train_vector(
                    agent, cfg, load_report=load_report, writer=writer
                )
            else:
                result = _train(agent, env, cfg, load_report=load_report, writer=writer)
    finally:
        write_json(work_dir / "writer.json", writer.diagnostics())
    result["writer"] = writer.diagnostics()
    # Include queue drain/close in end-to-end wall throughput.
    result["wall_seconds"] = perf_counter() - started
    result["steps_per_second"] = result["steps"] / result["wall_seconds"]
    if hasattr(agent, "model"):
        result["model_parameters"] = agent.model.total_params
        result["actor_parameters"] = sum(
            p.numel() for p in agent.model._pi.parameters() if p.requires_grad
        )
    write_json(work_dir / "summary.json", result)
    return result
