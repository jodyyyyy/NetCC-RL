"""EXP-011 segmented runtime (025d023d), reused for EXP-012 node models."""
from collections import Counter
from time import perf_counter
import torch


def tensors(tree):
    if isinstance(tree, torch.Tensor): yield tree
    elif isinstance(tree, dict):
        for v in tree.values(): yield from tensors(v)
    elif isinstance(tree,(tuple,list)):
        for v in tree: yield from tensors(v)


def clone(tree):
    if isinstance(tree,torch.Tensor):return tree.detach().clone()
    if isinstance(tree,dict):return {k:clone(v) for k,v in tree.items()}
    if isinstance(tree,(tuple,list)):return type(tree)(clone(v) for v in tree)
    return tree


def signature(tree):
    if isinstance(tree,torch.Tensor):return (tuple(tree.shape),tree.dtype,tree.device)
    if isinstance(tree,dict):return tuple((k,signature(v)) for k,v in tree.items())
    if isinstance(tree,(tuple,list)):return tuple(signature(v) for v in tree)
    return tree


def copy_into(dest,src):
    if isinstance(src,torch.Tensor):dest.copy_(src.detach())
    elif isinstance(src,dict):
        for k in src:copy_into(dest[k],src[k])
    elif isinstance(src,(tuple,list)):
        for d,s in zip(dest,src):copy_into(d,s)


class Segment:
    def __init__(self,runtime,name,fn,args):
        self.args=clone(args);self.name=name
        agent=runtime.agent;device=agent.device
        mutable=list(agent.model.parameters())+list(agent.model.buffers())+[agent.scale.value]
        for opt in runtime.optimizers.values():
            mutable += list(tensors(opt.state))
        # Preserve real gradients while warming/capturing a clip/Adam segment.
        mutable += [p.grad for p in agent.model.parameters() if p.grad is not None]
        unique={id(t):t for t in mutable};saved=[(t,t.detach().clone()) for t in unique.values()]
        cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state(device)
        generators=[]
        states=[g.get_state() for g in generators]
        def restore():
            with torch.no_grad():
                for dst,src in saved:
                    # Do not bump static topology versions when values are unchanged.
                    if not torch.equal(dst,src):dst.copy_(src)
            torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda,device)
            for g,s in zip(generators,states):g.set_state(s)
        torch.cuda.synchronize(device);begin=perf_counter()
        allocated=torch.cuda.memory_allocated(device);reserved=torch.cuda.memory_reserved(device)
        stream=torch.cuda.Stream(device=device);stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):fn(*self.args)
        torch.cuda.current_stream(device).wait_stream(stream);torch.cuda.synchronize(device)
        restore();stream.wait_stream(torch.cuda.current_stream(device))
        self.graph=torch.cuda.CUDAGraph()
        for g in generators:self.graph.register_generator_state(g)
        try:
            with torch.cuda.graph(self.graph,stream=stream):self.output=fn(*self.args)
        finally:
            torch.cuda.synchronize(device);restore()
        # Frozen profiles retain all prepared CSRs; legacy profiles may use an
        # LRU. CUDA graphs hold raw addresses, not the Python adjacency owner.
        # Pin owners for the full segment lifetime in BOTH preparation modes.
        self.graph_owners = tuple(graph for module in agent.model.modules()
                                  for graph in getattr(module, '_turbo_graphs', {}).values())
        self.capture_ms=(perf_counter()-begin)*1000
        self.allocated_bytes=torch.cuda.memory_allocated(device)-allocated
        self.reserved_bytes=torch.cuda.memory_reserved(device)-reserved

    def __call__(self,args):
        copy_into(self.args,args);self.graph.replay();return self.output


class GraphAdam(torch.optim.Adam):
    """The unchanged torch Adam calculation, captured after state initialization."""
    def step(self,closure=None):
        runtime=getattr(self,'runtime',None)
        if runtime is None or not runtime.graphs:return super().step(closure)
        if closure is not None:raise ValueError('learner graph Adam does not accept closures')
        self.ensure_state()
        return runtime.call('adam_'+self.phase,lambda:super(GraphAdam,self).step(),())

    @torch.no_grad()
    def ensure_state(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or self.state[p]:continue
                self.state[p].update(step=torch.zeros((),device=p.device),
                                     exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
                if group['amsgrad']:self.state[p]['max_exp_avg_sq']=torch.zeros_like(p)


class LearnerRuntime:
    max_shapes=32
    def __init__(self,agent):
        self.agent=agent
        self.graphs=bool(getattr(agent.cfg,'learner_cuda_graphs',False) and agent.device.type=='cuda')
        self.optimizers={'model':agent.optim,'actor':agent.pi_optim}
        for name,opt in self.optimizers.items():
            if isinstance(opt,GraphAdam):opt.runtime=self;opt.phase=name
        self.cache={};self.counts=Counter();self.updates=0

    def call(self,name,fn,args):
        if not self.graphs:return fn(*args)
        key=(name,signature(args))
        if key not in self.cache:
            self.counts['graph_cache_misses']+=1
            if sum(k[0]==name for k in self.cache)>=self.max_shapes:
                raise RuntimeError(f'learner CUDA Graph shape limit exceeded for {name}; eager fallback is disabled')
            # CUDAGraph binds its RNG/device at construction. The caller may
            # leave cuda:0 current while this learner lives on another GPU.
            with torch.cuda.device(self.agent.device):
                self.cache[key]=Segment(self,name,fn,args)
        else:self.counts['graph_cache_hits']+=1
        with torch.cuda.device(self.agent.device):
            return self.cache[key](args)

    def backward(self,loss,name):loss.backward()

    def finish(self,name,loss,params):
        opt=self.optimizers[name]
        params=tuple(params)
        norm=self.call('clip_'+name,lambda:torch.nn.utils.clip_grad_norm_(params,self.agent.cfg.grad_clip_norm),())
        if not bool(torch.stack([loss.isfinite().all(),norm.isfinite().all()]).all()):
            self.counts[name+'_nonfinite']+=1
            raise FloatingPointError(f'nonfinite {name} loss or gradient before optimizer step')
        opt.step();self.counts[name+'_steps']+=1
        # Retain graph-owned gradient pointers; zeros are never stale gradients.
        opt.zero_grad(set_to_none=False)
        return norm

    def metadata(self):
        return dict(precision='fp32',graphs=self.graphs,updates=self.updates,
                    compiled_modules=getattr(self.agent, '_compiled_module_names', []),counts=dict(self.counts),
                    segments=[dict(name=k[0],shapes=str(k[1]),capture_ms=v.capture_ms,
                                   allocated_bytes=v.allocated_bytes,reserved_bytes=v.reserved_bytes,
                                   pinned_graph_owners=len(v.graph_owners)) for k,v in self.cache.items()])


def graph_support(cfg, device):
    """Return explicit support status; EXP-012 rejects unsupported requests."""
    if not getattr(cfg, 'learner_cuda_graphs', False):
        return 'disabled'
    if device.type != 'cuda':
        return 'eager: CPU'
    if getattr(cfg, 'exp017_arm', None):
        if cfg.graph_checkpoint or cfg.graph_backend != 'pyg' or cfg.multitask:
            return 'unsupported: EXP017 requires PyG/checkpoint=false/single-task'
        return 'enabled'
    # Episodic K1 uses a fixed-shape masked BCE in model_backward. Its
    # termination parameters share the model clip/GraphAdam segments, and
    # terminated remains a replay input to the captured TD-target mask.
    if (cfg.actor_type != 'gaussian' or cfg.actor_architecture != 'shared_native_state'
            or cfg.multitask or cfg.compile):
        return 'unsupported: requires K1 Gaussian shared native-state actor, single-task'
    if cfg.state_encoder_type != 'mlp' and not (
            cfg.state_encoder_type == 'gatv2_prior' and cfg.graph_backend == 'turbo'
            and cfg.graph_batch_size == 512 and not cfg.graph_checkpoint):
        return 'unsupported: graph learner requires Turbo prior, microbatch512/checkpoint=false'
    return 'enabled'
