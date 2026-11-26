"""Real/synthetic trajectory storage with owned tensors and resumable RNG."""
from collections import deque
from copy import deepcopy
import numpy as np
import torch

FIELDS=('states','actions','rewards','next_states','terminated','truncated','logp')


def trajectory(**items):
    result={k:torch.as_tensor(items[k]).detach().cpu().clone() for k in FIELDS}
    sizes={len(v) for v in result.values()}
    if len(sizes)!=1 or not sizes or next(iter(sizes))<1: raise ValueError('aligned nonempty trajectory required')
    result['kind']=items.get('kind','real')
    if result['kind'] not in ('real','synthetic'): raise ValueError('invalid trajectory kind')
    if result['states'].shape!=result['next_states'].shape: raise ValueError('state shape mismatch')
    # A trajectory ends at a boundary, never contains the next episode.
    boundary=result['terminated']|result['truncated']
    if boundary[:-1].any(): raise ValueError('trajectory crosses an episode boundary')
    for k in ('states','actions','rewards','next_states','logp'):
        if not torch.isfinite(result[k]).all(): raise ValueError('nonfinite trajectory')
    return result


class TrajectoryReplay:
    def __init__(self, capacity, seed=0):
        if capacity<1: raise ValueError('positive trajectory capacity required')
        self.capacity=int(capacity);self.items=deque(maxlen=capacity);self.rng=np.random.default_rng(seed)

    def add(self,item):
        self.items.append(trajectory(**item))

    def sample(self,count,length=None):
        if not self.items: raise RuntimeError('empty trajectory replay')
        out=[]
        for i in self.rng.integers(0,len(self.items),size=count):
            item=self.items[int(i)]; size=len(item['states'])
            n=size if length is None else min(size,length)
            start=int(self.rng.integers(0,size-n+1))
            out.append(trajectory(**{k:item[k][start:start+n] for k in FIELDS},kind=item['kind']))
        return out

    def state_dict(self):
        return dict(capacity=self.capacity,items=deepcopy(list(self.items)),rng=deepcopy(self.rng.bit_generator.state))

    def load_state_dict(self,state):
        if state['capacity']!=self.capacity: raise ValueError('replay capacity differs')
        self.items=deque((trajectory(**x) for x in state['items']),maxlen=self.capacity)
        self.rng.bit_generator.state=deepcopy(state['rng'])
