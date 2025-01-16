from typing import Literal

import numpy as np
from nerfstudio.field_components import MLP
from torch import Tensor, nn


class DensityMLP(MLP):
    """MLP for density
    The tcnn MLP does not support second-order gradients.
    Use PyTorch MLP instead and initializing its parameters by copying them from the tcnn MLP.
    """

    def __init__(self, implementation: Literal['tcnn', 'torch'] = 'torch', *args, **kwargs) -> None:
        super().__init__(implementation='torch', *args, **kwargs)
        layer: nn.Linear
        for layer in self.layers:  # type: ignore
            nn.init.constant_(layer.bias.data, 0)
        if implementation == 'tcnn':
            tcnn_params = MLP(implementation='tcnn', *args, **kwargs).tcnn_encoding.params  # type: ignore
            offset = 0
            for layer in self.layers:  # type: ignore
                mlp_in = layer.in_features
                mlp_out = layer.out_features
                tcnn_in = 2 ** int(np.ceil(np.log2(mlp_in)))
                tcnn_out = 2 ** int(np.ceil(np.log2(mlp_out)))
                layer.weight.data = tcnn_params[offset : offset + mlp_out * mlp_in].reshape(mlp_out, mlp_in).to(layer.weight.data)
                offset += tcnn_out * tcnn_in

    def forward(self, in_tensor: Tensor) -> Tensor:
        if self.tcnn_encoding is None:
            in_tensor = in_tensor.to(self.layers[0].weight.data)  # type: ignore
        return super().forward(in_tensor)
