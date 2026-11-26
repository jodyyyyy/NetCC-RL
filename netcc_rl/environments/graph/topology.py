"""Frozen MATPOWER branch endpoints, without electrical parameter reuse."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import shortest_path


def ieee39_core38():
    data=json.loads((Path(__file__).parent/'data/ieee39_core38_v1.json').read_text())
    a=np.zeros((38,38),dtype=np.float32)
    for u,v in data['branches_bus_ids']:
        a[u-1,v-1]=a[v-1,u-1]=1
    assert a.sum()==88 and not np.diag(a).any()
    assert np.isfinite(shortest_path(a,directed=False)).all()
    return a,data


def grounded_distance_greedy(a,budget):
    """Maximize grounded lambda_min; spectral ties spread drivers, then ID."""
    a=np.asarray(a,dtype=np.float64)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not 0<budget<len(a):
        raise ValueError("expected square adjacency and budget between 1 and n-1")
    distances=shortest_path(a,directed=False)
    matrix=np.diag(a.sum(1))-a
    remaining=list(range(len(a)));ordered=[];trace=[]
    for _ in range(budget):
        scores={}
        for node in remaining:
            indices=[i for i in remaining if i!=node]
            scores[node]=float(eigh(matrix[np.ix_(indices,indices)],subset_by_index=[0,0],eigvals_only=True)[0])
        best=max(scores.values())
        tied=[i for i in remaining if abs(scores[i]-best)<=1e-10]
        separation={i:float(min(distances[i,j] for j in ordered)) if ordered else 0. for i in tied}
        node=min(tied,key=lambda i:(-separation[i],i))
        ordered.append(node);remaining.remove(node)
        trace.append(dict(node=node,score=scores[node],minimum_driver_distance=separation[node]))
    return sorted(ordered),dict(selector='grounded_laplacian_distance_greedy_v1',
        tie_tolerance=1e-10,tie_break='greatest minimum shortest-path distance, then smallest ID',
        ordered_driver_nodes=ordered,trace=trace)
