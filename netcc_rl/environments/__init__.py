"""Environment public API; no policy, optimizer or removed simulator imports."""

from .mpcsc import TASKS as MPCSC_TASKS
from .mpcsc import make_env as make_mpcsc_env, make_gym_env as make_mpcsc_gym_env
from .graph import TASKS as GRAPH_TASKS
from .graph import make_env as make_graph_env, make_gym_env as make_graph_gym_env

TASKS = ("mpcsc-kuramoto", "mpcsc-sis", "graph-swing")


def supported_task(task):
    from .mpcsc.config import task_identity

    return task == "graph-swing" or task_identity(task) in {
        "mpcsc-kuramoto",
        "mpcsc-kuramoto-control-cost",
        "mpcsc-sis",
        "mpcsc-sis-control-cost",
    }


def make_env(cfg):
    from .mpcsc.tdmpc2 import _get

    task = str(_get(cfg, "task", ""))
    if not supported_task(task):
        raise ValueError("main registry supports only Kuramoto, SIS and Swing")
    if _get(cfg, "collector_protocol") == "exp016_three_environments":
        from .graph.frozen import for_config

        for_config(cfg)
    if task in GRAPH_TASKS:
        return make_graph_env(cfg)
    if _get(cfg, "graph_resolved") or _get(cfg, "graph_params"):
        raise ValueError("graph options conflict with non-graph task")
    return make_mpcsc_env(cfg)


def make_gym_env(task="mpcsc-kuramoto", **kwargs):
    if not supported_task(task):
        raise ValueError("main registry supports only Kuramoto, SIS and Swing")
    return (make_graph_gym_env if task in GRAPH_TASKS else make_mpcsc_gym_env)(
        task, **kwargs
    )


__all__ = ["TASKS", "make_env", "make_gym_env"]
