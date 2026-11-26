"""Versioned fixed actuators, selected without reset trajectories or rewards.

Export an environment only (no learner):
python -m netcc_rl.environments.mpcsc.fixed_actuators --task kuramoto --output nodes.json
"""

import hashlib
import json
import time
import networkx as nx
import numpy as np
import torch

CONTROL_KEYS = {"actuator_mode", "driver_nodes", "selection_seed"}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def population_keys(task):
    return ("node_num", "driver_node_num")


def construct_backend(task, params, *, full=False, physics=None):
    """Full construction consumes the same RNG draws for every actuator budget.

    v2 deliberately anchors physics to the historical all-node construction.
    Actuator installation below consumes neither physical RNG stream.
    """
    p = {
        k: v
        for k, v in params.items()
        if k not in CONTROL_KEYS
        and k not in {"kuramoto_action_cost_coef", "sis_control_cost_coef"}
    }
    if full:
        nkey, dkey = population_keys(task)
        p[dkey] = p[nkey]
    if physics is not None and True:
        p["frozen_adjacency"] = physics["adjacency"]
        from ..graph.frozen import physical_layout

        layout = physical_layout(physics)
        if layout is not None:
            p["frozen_neighbors"] = layout["neighbors"]
    if task == "mpcsc-kuramoto":
        from .dynamics.kuramoto import KuramotoDynamics

        return KuramotoDynamics(**p)
    if task == "mpcsc-sis":
        from .dynamics.sis import SISDynamics

        return SISDynamics(**p)
    raise ValueError("unsupported mainline MPCSC task")


def physical_state(task, backend):
    state = {
        "schema": "all_node_physics_v1",
        "python_rng": backend.py_random.getstate(),
        "torch_rng": backend.torch_generator.get_state().tolist(),
    }
    state["adjacency"] = backend.adj_matrix.tolist()
    if task == "mpcsc-kuramoto":
        state["omegas"] = backend.init_omegas.tolist()
    else:
        state["initial_infected"] = list(backend.initial_I_node_list)
        state["default_infected"] = list(backend.default_initial_I_node_list)
    return json.loads(json.dumps(state))


def restore_physics(task, backend, state):
    """Restore stored physics, not graph generation under a newer dependency."""

    def tuples(x):
        return tuple((tuples(v) for v in x)) if isinstance(x, list) else x

    backend.py_random.setstate(tuples(state["python_rng"]))
    backend.torch_generator.set_state(
        torch.tensor(state["torch_rng"], dtype=torch.uint8)
    )
    backend.adj_matrix = torch.tensor(state["adjacency"], dtype=torch.float32)
    from ..graph.frozen import physical_layout

    layout = physical_layout(state)
    backend.G = None if layout else nx.from_numpy_array(backend.adj_matrix.numpy())
    if task == "mpcsc-kuramoto":
        backend.init_omegas = torch.tensor(state["omegas"], dtype=torch.float32)
    else:
        backend.initial_I_node_list = list(state["initial_infected"])
        backend.default_initial_I_node_list = list(state["default_infected"])
        backend.node_edge_dict = (
            {i: list(v) for i, v in enumerate(layout["neighbors"])}
            if layout
            else {
                i: torch.nonzero(row).flatten().tolist()
                for i, row in enumerate(backend.adj_matrix)
            }
        )


def validate_nodes(nodes, count, budget):
    if (
        not isinstance(nodes, list)
        or len(nodes) != budget
        or any((type(i) is not int or not 0 <= i < count for i in nodes))
        or (len(set(nodes)) != budget)
    ):
        raise ValueError(
            "driver_nodes must contain exactly D unique integer IDs in [0,N)"
        )
    return list(nodes)


def install(task, backend, nodes):
    count = backend.node_num
    matrix = np.eye(count, dtype=np.float32)[nodes]
    backend.driver_node_num, backend.driver_node_list = (len(nodes), list(nodes))
    backend.action_dim = len(nodes)
    backend.control_matrix = torch.from_numpy(matrix)


def greedy_spectral(adjacency, budget, *, grounded):
    """Exact symmetric eigenvalue scoring; near ties (1e-10) use smallest ID."""
    from scipy.linalg import eigh

    a = np.asarray(adjacency, dtype=np.float64)
    degrees = a.sum(axis=1)
    matrix = (
        np.diag(degrees) - a
        if grounded
        else a / np.sqrt(degrees[:, None] * degrees[None, :])
    )
    remaining, selected, trace = (list(range(len(a))), [], [])
    for _ in range(budget):
        scores = {}
        for node in remaining:
            indices = [i for i in remaining if i != node]
            if not indices:
                score = 0.0
            else:
                sub = matrix[np.ix_(indices, indices)]
                index = 0 if grounded else len(indices) - 1
                score = float(
                    eigh(sub, subset_by_index=[index, index], eigvals_only=True)[0]
                )
            scores[node] = score
        optimum = max(scores.values()) if grounded else min(scores.values())
        best_node = min(
            (node for node, score in scores.items() if abs(score - optimum) <= 1e-10)
        )
        best_score = scores[best_node]
        selected.append(best_node)
        remaining.remove(best_node)
        trace.append({"node": best_node, "score": best_score})
    return (sorted(selected), trace)


def sis_selection(adjacency, budget, beta, gamma, *, time_limit=30.0):
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    a = np.asarray(adjacency, dtype=np.float64)
    n = len(a)
    u, v = np.where(np.triu(a, 1) != 0)
    incidence = coo_matrix(
        (np.ones(2 * len(u)), (np.tile(np.arange(len(u)), 2), np.concatenate([u, v]))),
        shape=(len(u), n),
    ).tocsc()
    result = milp(
        np.ones(n),
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(incidence, 1, np.inf),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    solver = {
        "name": "scipy.optimize.milp/HiGHS",
        "status": int(result.status),
        "message": str(result.message),
        "time_limit": time_limit,
        "optimal": bool(result.success),
    }
    for key in ("fun", "mip_gap", "mip_dual_bound"):
        value = getattr(result, key, None)
        solver[key] = float(value) if value is not None and np.isfinite(value) else None
    cover = None
    if result.x is not None and np.all(np.abs(result.x - np.rint(result.x)) <= 1e-06):
        proposed = np.flatnonzero(result.x > 0.5).tolist()
        if all((i in proposed or j in proposed for i, j in zip(u, v))):
            solver["verified_cover_size"] = len(proposed)
            if len(proposed) <= budget:
                cover = proposed
    if cover is not None:
        nodes = sorted(
            cover + [i for i in range(n) if i not in cover][: budget - len(cover)]
        )
        trace, method = ([], "minimum_vertex_cover_milp")
    else:
        nodes, trace = greedy_spectral(a, budget, grounded=False)
        method = "original_degree_propagation_spectral_greedy"
    unselected = [i for i in range(n) if i not in nodes]
    normalized = a / np.sqrt(a.sum(1)[:, None] * a.sum(1)[None, :])
    rho = (
        float(np.linalg.eigvalsh(normalized[np.ix_(unselected, unselected)])[-1])
        if unselected
        else 0.0
    )
    threshold = gamma / beta if beta > 0 else None
    certified = cover is not None or beta == 0 or rho < threshold
    return (
        nodes,
        {
            "selector": method,
            "solver": solver,
            "trace": trace,
            "rho_original_degree_P_UU": rho,
            "gamma_over_beta": threshold,
            "certificate": (
                "VERTEX_COVER"
                if cover is not None
                else "SPECTRAL_RADIUS" if certified else "NOT_CERTIFIED"
            ),
            "certificate_assumptions": "noise_std=0; sustained native control >= beta at all drivers",
        },
    )


def prepare(task, params):
    start = time.perf_counter()
    backend = construct_backend(task, params, full=True)
    physics = physical_state(task, backend)
    nkey, dkey = population_keys(task)
    count, budget = (params[nkey], params[dkey])
    if "driver_nodes" in params:
        nodes = validate_nodes(params["driver_nodes"], count, budget)
        selection = {"selector": "explicit_driver_ids", "certificate": "NOT_EVALUATED"}
    elif budget == count:
        nodes, selection = (list(range(count)), {"selector": "all_nodes_reference"})
    elif task == "mpcsc-kuramoto":
        nodes, trace = greedy_spectral(physics["adjacency"], budget, grounded=True)
        selection = {
            "selector": "grounded_laplacian_greedy",
            "trace": trace,
            "lambda_min": trace[-1]["score"],
        }
    elif task == "mpcsc-sis":
        nodes, selection = sis_selection(
            physics["adjacency"], budget, params["beta"], params["gamma"]
        )
        if params["noise_std"] != 0 or params["beta"] > 1:
            selection["certificate"] = "NOT_CERTIFIED"
            selection["reason"] = "nonzero_noise_or_unreachable_native_suppression"
    else:
        seed = params.get("selection_seed", 20261021)
        nodes = sorted(
            np.random.default_rng(seed).choice(count, budget, replace=False).tolist()
        )
        selection = {"selector": "uniform_random_fixed_leaders", "selection_seed": seed}
    selection.update(
        version=1, tie_tolerance=1e-10, elapsed_seconds=time.perf_counter() - start
    )
    elapsed = selection.pop("elapsed_seconds")
    result = {
        "version": 2,
        "driver_nodes": nodes,
        "driver_sha256": digest(nodes),
        "physical_seed": params["seed"],
        "physical_system": physics,
        "physical_sha256": digest(physics),
        "selection": selection,
    }
    return (result, elapsed)


def build_backend(task, params, fixed):
    backend = construct_backend(
        task, params, full=True, physics=fixed["physical_system"]
    )
    restore_physics(task, backend, fixed["physical_system"])
    install(task, backend, fixed["driver_nodes"])
    return backend


def main():
    import argparse
    from pathlib import Path
    import subprocess
    import scipy
    from .config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["kuramoto", "sis"], required=True)
    parser.add_argument("--nodes", type=int, default=200)
    parser.add_argument("--drivers", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; refuse overwrite")
    task = "mpcsc-" + args.task
    nkey, dkey = population_keys(task)
    params = {
        nkey: args.nodes,
        dkey: args.drivers if args.drivers is not None else args.nodes // 2,
        "actuator_mode": "fixed_v2",
    }
    if args.task == "kuramoto":
        params["kuramoto_action_cost_coef"] = 0.1
    elif args.task == "sis":
        task += "-control-cost"
    start = time.perf_counter()
    cfg = load_config(task, params=params)
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "environments").rglob("*.py")) + [
        root / "requirements-environments.txt"
    ]
    report = {
        "snapshot": cfg.snapshot(),
        "elapsed_seconds": time.perf_counter() - start,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            )
        ),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "networkx": nx.__version__,
            "torch": torch.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "elapsed_seconds": report["elapsed_seconds"],
                "driver_sha256": cfg.fixed_actuators["driver_sha256"],
                "selection": cfg.fixed_actuators["selection"]["selector"],
            }
        )
    )


if __name__ == "__main__":
    main()
