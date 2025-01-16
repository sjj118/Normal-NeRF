from typing import Union

import tinycudann as tcnn
import torch
from torch import nn

from . import _C

grid_type_map = {
    'Dense': 0,
    'Tiled': 1,
    'Hash': 2,
}


def _grid_offset(grid: Union[tcnn.Encoding, tcnn.NetworkWithInputEncoding]):
    cfg = grid.native_tcnn_module.hyperparams()
    if isinstance(grid, tcnn.NetworkWithInputEncoding):
        cfg = cfg['encoding']
    grid_type = grid_type_map[cfg['type']]
    n_levels = cfg['n_levels']
    log2_hashmap_size = cfg['log2_hashmap_size']
    base_resolution = cfg['base_resolution']
    per_level_scale = cfg['per_level_scale']
    n_features_per_level = cfg['n_features_per_level']
    offsets = _C.grid_offset(
        grid_type,
        n_levels,
        log2_hashmap_size,
        base_resolution,
        per_level_scale,
        n_features_per_level,
    )
    return offsets


class GridDecayLoss(nn.Module):
    def __init__(self, grid: Union[tcnn.Encoding, tcnn.NetworkWithInputEncoding]):
        super().__init__()
        self.grid = grid
        offsets = _grid_offset(grid)
        self.scale = nn.Parameter(torch.ones(offsets[-1], dtype=grid.params.dtype), requires_grad=False)
        for i in range(len(offsets) - 1):
            self.scale[offsets[i] : offsets[i + 1]] /= offsets[i + 1] - offsets[i]

    def forward(self):
        return (self.scale * (self.grid.params[: len(self.scale)] ** 2)).sum()
