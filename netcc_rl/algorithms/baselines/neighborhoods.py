"""Frozen physical graph powers, with independent observation/value radii."""
import numpy as np
import torch
from .environment import load_snapshot
from .phase_features import phase_features


class Neighborhoods:
    def __init__(self, adjacency, centers, radius):
        if type(radius) is not int or radius < 0:
            raise ValueError('radius must be a nonnegative integer')
        graph=np.asarray(adjacency)
        if graph.ndim!=2 or graph.shape[0]!=graph.shape[1]: raise ValueError('square graph required')
        # Exact source matrix-power reachability then self union. Do not replace
        # with a clique or node-index distance; at r=2 this differs from BFS <=r.
        reach=np.linalg.matrix_power((graph>0).astype(np.int64),radius)>0
        reach |= np.eye(len(graph),dtype=bool)
        self.centers=tuple(centers)
        self.nodes=tuple(tuple(int(j) for j in np.flatnonzero(reach[i])) for i in centers)

    def gather(self, states):
        return [phase_features(states[...,list(nodes)]) for nodes in self.nodes]

    def controlled_neighbors(self, drivers):
        mapping={int(node):i for i,node in enumerate(drivers)}
        return [[mapping[j] for j in nodes if j in mapping] for nodes in self.nodes]

    def reduce(self, values, drivers):
        return torch.stack([values[...,idx].sum(-1) for idx in self.controlled_neighbors(drivers)],-1)


def physical_graph():
    s=load_snapshot(); f=s.get('fixed_actuators',s)
    return np.asarray(s.get('adjacency') or f['physical_system']['adjacency']),tuple(f['driver_nodes'])
