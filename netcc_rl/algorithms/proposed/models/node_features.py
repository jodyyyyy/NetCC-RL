"""Node-wise feature/layout helpers for MPCSC graph and shared-policy experiments."""

import torch


def node_layout(cfg):
    """Return (node_count, raw_features_per_node, encoded_features_per_node, task)."""
    graph = getattr(cfg, "graph_resolved", None)
    if graph:
        n, f = (graph["params"]["n"], graph["raw_features"])
        if tuple(cfg.obs_shape["state"]) != (n * f,):
            raise ValueError("graph observation shape mismatch")
        return (n, f, f, graph["task"])
    snapshot = getattr(cfg, "mpcsc_resolved", None)
    if not isinstance(snapshot, dict):
        raise ValueError(
            "graph/shared actors require a resolved MPCSC environment snapshot"
        )
    task = snapshot.get("task")
    params = snapshot.get("params", {})
    if task in {"mpcsc-kuramoto", "mpcsc-sis"}:
        node_count, raw_features = (params.get("node_num"), 1)
    else:
        raise ValueError(f"unsupported node-wise observation task: {task!r}")
    if type(node_count) is not int or node_count <= 0:
        raise ValueError("environment snapshot has no valid node count")
    obs_shape = getattr(cfg, "obs_shape", None)
    if not isinstance(obs_shape, dict) or "state" not in obs_shape:
        raise ValueError("node-wise actors require a resolved state observation shape")
    obs_dim = int(obs_shape["state"][0])
    if obs_dim != node_count * raw_features:
        raise ValueError(
            f"state observation dimension {obs_dim} does not match {node_count} nodes x {raw_features} features"
        )
    encoded_features = 2 if task == "mpcsc-kuramoto" else raw_features
    return (node_count, raw_features, encoded_features, task)


def node_features(obs, cfg):
    """Reshape flattened MPCSC observations and encode circular Kuramoto phases."""
    node_count, raw_features, _, task = node_layout(cfg)
    if obs.shape[-1] != cfg.obs_shape["state"][0]:
        raise ValueError("observation dimension differs from the resolved state shape")
    nodes = obs.reshape(*obs.shape[:-1], node_count, raw_features)
    if task == "mpcsc-kuramoto":
        theta = nodes[..., 0]
        return torch.stack((theta.sin(), theta.cos()), dim=-1)
    return nodes


def fixed_driver_nodes(cfg):
    """Read the exact ordered controlled-node list from the frozen environment snapshot."""
    graph = getattr(cfg, "graph_resolved", None)
    if graph:
        nodes = graph["driver_nodes"]
        if len(nodes) != cfg.action_dim:
            raise ValueError("graph driver dimension mismatch")
        return nodes
    snapshot = getattr(cfg, "mpcsc_resolved", None)
    fixed = snapshot.get("fixed_actuators") if isinstance(snapshot, dict) else None
    if not isinstance(fixed, dict) or fixed.get("version") != 2:
        raise ValueError(
            "shared actors require actuator_mode=fixed_v2 with frozen driver IDs"
        )
    nodes = fixed.get("driver_nodes")
    node_count, _, _, _ = node_layout(cfg)
    if (
        not isinstance(nodes, list)
        or len(nodes) != cfg.action_dim
        or any((type(i) is not int or not 0 <= i < node_count for i in nodes))
        or (len(set(nodes)) != len(nodes))
    ):
        raise ValueError("frozen driver IDs do not match the action dimension")
    return nodes


def prior_adjacency(cfg):
    """Load the physical adjacency from the frozen snapshot, never from a live environment."""
    graph = getattr(cfg, "graph_resolved", None)
    if graph:
        return torch.tensor(graph["adjacency"], dtype=torch.float32)
    snapshot = getattr(cfg, "mpcsc_resolved", None)
    fixed = snapshot.get("fixed_actuators") if isinstance(snapshot, dict) else None
    physical = fixed.get("physical_system") if isinstance(fixed, dict) else None
    adjacency = physical.get("adjacency") if isinstance(physical, dict) else None
    if adjacency is None:
        raise ValueError(
            "this graph encoder requires adjacency in the frozen physical snapshot"
        )
    node_count, _, _, _ = node_layout(cfg)
    adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
    if adjacency.shape != (node_count, node_count):
        raise ValueError(
            "frozen adjacency shape does not match the physical node count"
        )
    if not torch.isfinite(adjacency).all() or (adjacency < 0).any():
        raise ValueError("frozen adjacency must be finite and nonnegative")
    if not torch.allclose(adjacency, adjacency.T, atol=1e-06, rtol=0):
        raise ValueError(
            "current graph encoders require an undirected symmetric prior graph"
        )
    return adjacency
