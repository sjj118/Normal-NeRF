from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn


class NormalsShader(nn.Module):
    """Calculate shading for normals."""

    def __init__(self, bg: Literal['white', 'black', 'alpha'] = 'white'):
        super().__init__()
        self.bg = bg

    def forward(
        self,
        normals: Float[Tensor, '*bs 3'],
        accumulation: Float[Tensor, '*bs 1'],
    ):
        """Applies a rainbow colormap to the normals.

        Args:
            normals: Normalized 3D vectors.
            weights: Optional weights to scale to the normal colors. (Can be used for masking)

        Returns:
            Colored normals
        """
        normals = (normals + 1) / 2
        if self.bg == 'white':
            normals = normals + 0.5 * (1 - accumulation)
        elif self.bg == 'black':
            pass
        elif self.bg == 'alpha':
            normals /= accumulation.clamp_min(1e-12)
            normals = torch.cat((normals, accumulation), -1)
        return normals

    def reverse(
        self,
        normals: Float[Tensor, '*bs 3'],
        accumulation: Float[Tensor, '*bs 1'],
    ):
        if self.bg == 'white':
            normals = normals - 0.5 * (1 - accumulation)
        elif self.bg == 'black':
            pass
        elif self.bg == 'alpha':
            normals = normals[..., :3] * accumulation

        normals = normals * 2 - 1
        return normals
