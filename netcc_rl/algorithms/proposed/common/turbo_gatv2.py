"""Turbo-GNN aggregation with exactly the PyG GATv2Conv parameterization.

PyG lin_l projects source nodes, lin_r projects destination nodes. Turbo's
`x` is the destination projection; `x_neighbors` is the source projection.
The original PyG modules own all parameters and checkpoint keys.
"""
import torch


def turbo_aggregate(conv, x, graph, *, fused_projection=False, kernel_options=None):
    if not x.is_cuda or x.dtype != torch.float32:
        raise ValueError('Turbo GATv2 currently requires CUDA float32 inputs')
    if torch.is_autocast_enabled():
        raise ValueError('Turbo GATv2 currently requires autocast disabled')
    if conv.out_channels not in {32, 64, 128, 256}:
        raise ValueError('unsupported Turbo GATv2 head width')
    if conv.dropout != 0:
        raise ValueError('Turbo GATv2 does not support attention dropout; use PyG')
    if conv.add_self_loops or conv.edge_dim is not None or not conv.concat or conv.res is not None:
        raise ValueError('unsupported PyG GATv2 options for Turbo aggregation')
    from .verified_turbo import verified_turbo
    gatv2_aggr = verified_turbo().gatv2_aggr
    heads, width = conv.heads, conv.out_channels
    if fused_projection:
        # Keep the original Parameters and optimizer identities. Concatenation is
        # differentiable, including both independent biases; its cost is measured.
        from torch.nn.functional import linear
        weight = torch.cat((conv.lin_l.weight, conv.lin_r.weight))
        if (conv.lin_l.bias is None) != (conv.lin_r.bias is None):
            raise ValueError('fused projection requires matching bias presence')
        bias = None if conv.lin_l.bias is None else torch.cat((conv.lin_l.bias, conv.lin_r.bias))
        source, target = linear(x, weight, bias).split(heads * width, dim=-1)
        source = source.reshape(-1, heads, width).contiguous()
        target = target.reshape(-1, heads, width).contiguous()
    else:
        source = conv.lin_l(x).view(-1, heads, width).contiguous()
        target = conv.lin_r(x).view(-1, heads, width).contiguous()
    result = gatv2_aggr(graph, target, x_neighbors=source,
                      attention_weights=conv.att.squeeze(0).contiguous(),
                      negative_slope=conv.negative_slope, **(kernel_options or {}))
    result = result.reshape(-1, heads * width)
    return result if conv.bias is None else result + conv.bias
