from typing import Optional

import nerfacc
import torch
from jaxtyping import Float, Int
from nerfstudio.utils.math import safe_normalize
from torch import Tensor, nn


class NormalsRenderer(nn.Module):
    """Calculate normals along the ray."""

    @classmethod
    def forward(
        cls,
        normals: Float[Tensor, '*bs num_samples 3'],
        weights: Float[Tensor, '*bs num_samples 1'],
        ray_indices: Optional[Int[Tensor, 'num_samples']] = None,
        num_rays: Optional[int] = None,
        normalize: bool = True,
    ) -> Float[Tensor, '*bs 3']:
        """Calculate normals along the ray.

        Args:
            normals: Normals for each sample.
            weights: Weights of each sample.
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.
            normalize: Normalize normals.
        """
        if ray_indices is not None and num_rays is not None:
            n = nerfacc.accumulate_along_rays(weights[..., 0], ray_indices=ray_indices, values=normals, n_rays=num_rays)
        else:
            n = torch.sum(weights * normals, dim=-2)
        if normalize:
            n = safe_normalize(n)
        return n
