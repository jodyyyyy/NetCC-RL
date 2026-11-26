import torch
import torch.nn.functional as F


def soft_ce(pred, target, cfg):
    """Computes the cross entropy loss between predictions and soft targets."""
    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, cfg)
    return -(target * pred).sum(-1, keepdim=True)


def affine_action(unit_action, low, high):
    """Map [-1, 1] to the policy bounds, preserving FP32 endpoints."""
    weight = (unit_action + 1) / 2
    return low * (1-weight) + high * weight


def squashed_gaussian_density(mean, log_std, eps, low=-1., high=1.):
    """Sample and evaluate joint density in policy/environment coordinates.

    Independent action dimensions are summed, leaving batch axes.
    The Jacobian is evaluated before tanh, including at FP32 boundary actions.
    """
    import math as scalar_math
    x = mean + log_std.exp() * eps
    scale = (high - low) / 2
    log_jac = 2 * (scalar_math.log(2.) - x - F.softplus(-2*x))
    base = -0.5 * eps.square() - log_std - 0.5 * scalar_math.log(2*scalar_math.pi)
    log_prob = (base - log_jac - scalar_math.log(scale)).sum(-1, keepdim=True)
    action = affine_action(x.tanh(), low, high)
    return action, log_prob


def int_to_one_hot(x, num_classes):
    """
    Converts an integer tensor to a one-hot tensor.
    Supports batched inputs.
    """
    one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    one_hot.scatter_(-1, x.unsqueeze(-1), 1)
    return one_hot


def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
    bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size)
    bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device=x.device, dtype=x.dtype)
    bin_idx = bin_idx.long()
    soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot = soft_two_hot.scatter(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
    return soft_two_hot


def two_hot_inv(x, cfg):
    """Converts a batch of soft two-hot encoded vectors to scalars."""
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symexp(x)
    dreg_bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device, dtype=x.dtype)
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
    return symexp(x)


def gumbel_softmax_sample(p, temperature=1.0, dim=0):
    """Sample from the Gumbel-Softmax distribution."""
    logits = p.log()
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    return y_soft.argmax(-1)


def termination_statistics(pred, target, eps=1e-9):
    """Compute episode termination statistics."""
    pred = pred.squeeze(-1)
    target = target.squeeze(-1)
    rate = target.sum() / len(target)
    tp = ((pred > 0.5) & (target == 1)).sum()
    fn = ((pred <= 0.5) & (target == 1)).sum()
    fp = ((pred > 0.5) & (target == 0)).sum()
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    return {'termination_rate': rate,
            'termination_f1': f1}
