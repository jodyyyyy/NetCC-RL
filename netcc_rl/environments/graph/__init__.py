"""Public graph-dynamics factories."""
from .env import TASKS, GraphEnv, GraphDynamics
from netcc_rl.environments.mpcsc.adapter import NumpyEnvAdapter, TensorEnvAdapter, _get, _set


def make_gym_env(task,*,seed=None,params=None):
    return GraphEnv(task,params=params,seed=seed)


def make_env(cfg):
    if _get(cfg,'obs','state')!='state' or _get(cfg,'multitask',False):
        raise ValueError('graph dynamics requires single-task state observations')
    if _get(cfg,'mpcsc_resolved') or _get(cfg,'mpcsc_params'):
        raise ValueError('MPCSC options conflict with graph task')
    raw=GraphEnv(_get(cfg,'task'),params=_get(cfg,'graph_params'),snapshot=_get(cfg,'graph_resolved'),seed=_get(cfg,'seed',0))
    _set(cfg,'graph_resolved',raw.snapshot())
    env=TensorEnvAdapter(NumpyEnvAdapter(raw))
    _set(cfg,'obs_shape',{'state':env.observation_space.shape})
    _set(cfg,'action_dim',env.action_space.shape[0])
    _set(cfg,'episode_length',env.max_episode_steps)
    _set(cfg,'seed_steps',max(1000,5*env.max_episode_steps))
    from netcc_rl.algorithms.proposed.config import resolve_gaussian_bounds, resolve_model_widths
    resolve_gaussian_bounds(cfg)
    resolve_model_widths(cfg,env.observation_space.shape)
    return env
