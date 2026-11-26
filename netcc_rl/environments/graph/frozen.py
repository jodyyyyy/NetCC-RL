"""Read-only runtime topology registry. No graph generation or numerical solvers."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).parent/'data/frozen_runtime_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=16)
def _load(key):
    index=json.loads((ROOT/'index.json').read_text())
    if index.get('schema')!='frozen_runtime_v1':raise ValueError('unsupported frozen topology index schema')
    record=index['snapshots'].get(key)
    if record is None:return None
    raw=(ROOT/record['file']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('frozen topology artifact hash mismatch')
    data=json.loads(raw)
    if data['schema']!='frozen_runtime_v1' or digest(data['snapshot'])!=key:
        raise ValueError('frozen topology artifact identity mismatch')
    return data


def artifact(snapshot,*,required=False):
    data=_load(digest(snapshot))
    if required and data is None:raise ValueError('unknown frozen topology/snapshot identity; preprocess explicitly')
    return data


def for_config(cfg):
    get=cfg.get if isinstance(cfg,dict) else lambda key:getattr(cfg,key,None)
    required=get('collector_protocol')=='exp016_three_environments'
    snapshot=get('graph_resolved') or get('mpcsc_resolved')
    if not snapshot:
        if required:raise ValueError('formal collector protocol requires a nonempty frozen environment snapshot; preprocess explicitly')
        return None
    data=artifact(snapshot,required=required)
    if data is not None:
        if data['task']!=get('task') or data['drivers']!=list((snapshot.get('driver_nodes') or snapshot.get('fixed_actuators',{}).get('driver_nodes',[]))):
            raise ValueError('frozen topology task/action identity mismatch')
    return data


def validate_snapshot(task,snapshot,params=None):
    from .env import DEFAULTS
    if not isinstance(snapshot,dict) or snapshot.get('task')!=task or snapshot.get('version') not in (1,2,3):
        raise ValueError('invalid graph snapshot schema/task')
    body={k:v for k,v in snapshot.items() if k!='sha256'}
    if snapshot.get('sha256')!=digest(body):raise ValueError('graph snapshot integrity hash mismatch')
    p=snapshot['params'];n=p['n'];drivers=snapshot['driver_nodes'];a=snapshot['adjacency']
    if set(p)!=set(DEFAULTS) or type(n) is not int or n<3 or snapshot['node_order']!=list(range(n)):
        raise ValueError('invalid graph snapshot layout')
    if len(a)!=n or any(len(row)!=n for row in a) or any(v not in (0,1) for row in a for v in row):
        raise ValueError('invalid graph adjacency shape/range')
    if len(set(drivers))!=len(drivers) or any(type(i) is not int or not 0<=i<n for i in drivers):
        raise ValueError('invalid graph driver nodes')
    if snapshot['raw_features']!=5:raise ValueError('invalid feature layout')
    if params and (set(params)-set(p) or any(p[k]!=v for k,v in params.items())):
        raise ValueError('graph environment snapshot/config mismatch')
    # Versioned research identities are registered, including their ORIGINAL
    # diagnostic bytes. Never recompute spectra or replace scores during restore.
    data=artifact(snapshot,required=bool(snapshot.get('selection_profile')))
    return deepcopy(snapshot),data


def physical_layout(physics):
    index=json.loads((ROOT/'index.json').read_text())
    key=index['physical'].get(digest(physics))
    return _load(key) if key else None
