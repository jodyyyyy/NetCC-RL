"""Compile tensor modules while preserving native FP32 kernels."""

import torch


def _native_norm_inductor_backend(graph, inputs):
    """Retain ATen normalization/nonlinearity numerics inside Inductor graphs.

    B512 repeated-state probes expose amplified action drift when these FP32
    operations are decomposed. Keep their native kernels (also in backward),
    while Inductor compiles the remaining tensor operations. The outer runtime
    still captures all native and generated CUDA kernels together.
    """
    from torch._inductor.compile_fx import compile_fx
    from torch._inductor.decomposition import decompositions, select_decomp_table
    from torch._inductor.lowering import lowerings, make_fallback

    native = (
        torch.ops.aten.native_layer_norm.default,
        torch.ops.aten.native_layer_norm_backward.default,
        torch.ops.aten.mish.default,
        torch.ops.aten.mish_backward.default,
    )
    table = dict(select_decomp_table())
    for op in native:
        table.pop(op, None)
        if op not in lowerings:
            # Register the native lowering without changing the default global
            # decomposition policy used by other compile entry points.
            previous = decompositions.pop(op, None)
            try:
                make_fallback(op, warn=False)
            finally:
                if previous is not None:
                    decompositions[op] = previous
    return compile_fx(
        graph,
        inputs,
        config_patches={"triton.cudagraphs": False},
        decompositions=table,
    )


def compile_modules(model):
    """Inductor tensor kernels inside the existing segmented CUDA Graph runtime.

    Compile bound forward methods, preserving state_dict keys and parameter
    ownership. Inductor's own CUDA Graph wrapper is disabled to avoid nesting
    graph pools: LearnerRuntime captures forward/backward/optimizer segments.
    fullgraph=True and suppress_errors=False make unsupported kernels fail.
    """
    if torch._dynamo.config.suppress_errors:
        raise ValueError("Ablation compilation requires suppress_errors=false")
    names = []
    modules = [
        ("encoder", model._encoder["state"]),
        ("dynamics", model._dynamics),
        ("reward", model._reward),
        ("actor", model._pi),
    ]
    for name, module in modules:
        if module is not None:
            module.forward = torch.compile(
                module.forward, fullgraph=True, backend=_native_norm_inductor_backend
            )
            names.append(name)
    return names
