"""Explicit source-profile feedforward continuous policy building blocks."""
import torch
from torch import nn
from torch.distributions import Normal


def network(dim, hidden, out, activation='relu'):
    act={'relu':nn.ReLU,'tanh':nn.Tanh}[activation]; layers=[]
    for width in hidden:
        layers.extend([nn.Linear(dim,width),act()]);dim=width
    layers.append(nn.Linear(dim,out))
    return nn.Sequential(*layers)


class ContinuousActor(nn.Module):
    def __init__(self, dim, config, kind="happo"):
        super().__init__()
        self.kind = kind
        self.net = network(dim, config["hidden"], 2, config["activation"])
        with torch.no_grad():
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    layer.weight.normal_()
                    layer.weight.mul_(1 / layer.weight.square().sum(1, keepdim=True).sqrt())
                    layer.bias.zero_()
            self.net[-1].weight.mul_(config["gain"])

    def distribution(self, x):
        mean, log_std = self.net(x).split(1, -1)
        return Normal(mean, log_std.exp())


def gradient_step(optimizer,loss,params,clip=None):
    if not torch.isfinite(loss): raise FloatingPointError('nonfinite objective')
    params=list(params);optimizer.zero_grad();loss.backward()
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(params,clip,error_if_nonfinite=True)
    elif not torch.isfinite(torch.stack(torch._foreach_norm([p.grad for p in params if p.grad is not None],float('inf')))).all():
        raise FloatingPointError('nonfinite gradient')
    optimizer.step()
