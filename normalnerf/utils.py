import torch
from jaxtyping import Float
from torch import Tensor


def linear_to_srgb(linear: Float[Tensor, '*batch 3']) -> Float[Tensor, '*batch 3']:
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    eps = torch.tensor(torch.finfo(linear.dtype).eps)
    srgb0 = 323 / 25 * linear
    srgb1 = (211 * torch.fmax(eps, linear) ** (5 / 12) - 11) / 200
    return torch.where(linear <= 0.0031308, srgb0, srgb1)


def srgb_to_linear(srgb: Float[Tensor, '*batch 3']) -> Float[Tensor, '*batch 3']:
    """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    eps = torch.tensor(torch.finfo(srgb.dtype).eps)
    linear0 = 25 / 323 * srgb
    linear1 = torch.fmax(eps, ((200 * srgb + 11) / (211))) ** (12 / 5)
    return torch.where(srgb <= 0.04045, linear0, linear1)
