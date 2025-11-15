import torch


def ste_quantize(x: torch.Tensor, num_bits: int = 16) -> torch.Tensor:
    """
    Bit precision control of Gaussian parameters using a straight-through estimator.
    Reference: https://arxiv.org/abs/1308.3432
    """
    qmin, qmax = 0, 2**num_bits - 1
    min_val, max_val = x.min().item(), x.max().item()
    scale = max((max_val - min_val) / (qmax - qmin), 1e-8)
    # Quantize in forward pass (non-differentiable)
    q_x = torch.round((x - min_val) / scale).clamp(qmin, qmax)
    dq_x = q_x * scale + min_val
    # Restore gradients in backward pass
    dq_x = x + (dq_x - x).detach()
    return dq_x
