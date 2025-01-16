from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def pred_normal_loss(
    weights: Tensor,
    normals: Tensor,
    pred_normals: Tensor,
    num_rays: Optional[int] = None,
):
    """Loss between normals calculated from density and normals from prediction network."""
    w = weights
    nc = normals
    np = pred_normals

    return (w[..., 0] * (1.0 - torch.sum(nc * np, dim=-1))).sum() / num_rays


def compute_weighted_mae(weights, normals, normals_gt):
    """Compute weighted mean angular error, assuming normals are unit length."""
    one_eps = 1 - torch.finfo(normals.dtype).eps
    normals = F.normalize(normals, dim=-1)
    normals_gt = F.normalize(normals_gt, dim=-1)
    return (weights * torch.arccos(torch.clip((normals * normals_gt).sum(-1, keepdim=True), -one_eps, one_eps))).sum() / weights.sum() * 180.0 / torch.pi
