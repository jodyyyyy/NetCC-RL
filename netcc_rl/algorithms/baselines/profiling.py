"""Low frequency stage profiling and counter hooks, independent of algorithm budgets."""
from contextlib import contextmanager
from collections import defaultdict
import time
import numpy as np
import torch

class Stages:
    def __init__(self,device='cpu',synchronize=False):
        self.device=torch.device(device);self.synchronize=synchronize;self.samples=defaultdict(list)
    def sync(self):
        if self.synchronize and self.device.type=='cuda':torch.cuda.synchronize(self.device)
    @contextmanager
    def measure(self,name):
        self.sync();start=time.perf_counter()
        with torch.profiler.record_function(name):yield
        self.sync();self.samples[name].append(time.perf_counter()-start)
    def summary(self):
        return {k:dict(calls=len(v),total_seconds=sum(v),p50_seconds=float(np.median(v)),p95_seconds=float(np.percentile(v,95))) for k,v in self.samples.items()}

class OptimizerCounts:
    def __init__(self):self.counts=defaultdict(int);self.handles=[]
    def add(self,name,optimizer):
        def increment(*_):self.counts[name]+=1
        self.handles.append(optimizer.register_step_post_hook(increment))
    def close(self):
        for h in self.handles:h.remove()
        self.handles=[]


def profiled(name):
    """Optional stage timing on an owning object; no synchronization in hot loop."""
    from functools import wraps
    def decorate(method):
        @wraps(method)
        def wrapped(self,*args,**kwargs):
            stages=getattr(self,'stages',None)
            if stages is None:return method(self,*args,**kwargs)
            with stages.measure(name):return method(self,*args,**kwargs)
        return wrapped
    return decorate
