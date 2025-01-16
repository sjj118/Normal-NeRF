from typing import Dict, Literal, Optional

import nerfacc
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from normalnerf.field_components import DensityMLP


class NormalNeRFField(Field):
    """TCNN implementation of the Normal-NeRF field.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers_density: number of hidden layers for density network
        hidden_dim_density: dimension of hidden layers for density network
        geo_feat_dim: output geo feat dimensions
        num_layers_color: number of hidden layers for color network
        hidden_dim_color: dimension of hidden layers for color network
        num_layers_normal: number of hidden layers for normal network
        hidden_dim_normal: dimension of hidden layers for normal network
        appearance_embedding_dim: dimension of appearance embedding
        spatial_distortion: type of contraction
        num_levels: number of levels of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
    """

    def __init__(
        self,
        aabb: Tensor,
        spatial_distortion: Optional[SpatialDistortion] = None,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_res: int = 16,
        max_res: int = 2048,
        pe_dim: int = 2,
        num_layers_geometry: int = 3,
        hidden_dim_geometry: int = 64,
        geometry_dim: int = 32,
        num_layers_environment: int = 6,
        hidden_dim_environment: int = 128,
        environment_dim: int = 32,
        num_layers_material: int = 4,
        hidden_dim_material: int = 64,
        material_dim: int = 32,
        num_layers_diffuse: int = 2,
        hidden_dim_diffuse: int = 32,
        num_layers_specular: int = 4,
        hidden_dim_specular: int = 128,
        num_layers_normal: int = 3,
        hidden_dim_normal: int = 64,
        predict_normal_with_position: bool = True,
        predict_normal_with_hashgrid: bool = True,
        normal_reflect: Optional[Literal['pred_normal', 'dens_normal', 'tran_normal']] = 'pred_normal',
        density_activation: Literal['exp', 'softplus'] = 'exp',
        soft_density_grad: bool = True,
        implementation: Literal['tcnn', 'torch'] = 'tcnn',
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.normal_reflect: Optional[Literal['pred_normal', 'dens_normal', 'tran_normal']] = normal_reflect
        self.grad_scale: Tensor
        self.register_buffer('grad_scale', torch.tensor(1.0, requires_grad=False))
        self.predict_normal_with_position = predict_normal_with_position
        self.predict_normal_with_hashgrid = predict_normal_with_hashgrid
        self.density_activation = trunc_exp if density_activation == 'exp' else F.softplus
        self.soft_density_grad = soft_density_grad

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        self.position_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=pe_dim,
            min_freq_exp=0,
            max_freq_exp=pe_dim - 1,
            implementation=implementation,
        )

        self.grid = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )

        self.mlp_geometry = DensityMLP(
            in_dim=self.grid.get_out_dim(),
            num_layers=num_layers_geometry,
            layer_width=hidden_dim_geometry,
            out_dim=1 + geometry_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        self.mlp_normal = MLP(
            in_dim=predict_normal_with_hashgrid * geometry_dim + predict_normal_with_position * self.position_encoding.get_out_dim(),
            num_layers=num_layers_normal,
            layer_width=hidden_dim_normal,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        self.mlp_material = MLP(
            in_dim=geometry_dim,
            num_layers=num_layers_material,
            layer_width=hidden_dim_material,
            out_dim=material_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        self.mlp_diffuse = MLP(
            in_dim=material_dim,
            num_layers=num_layers_diffuse,
            layer_width=hidden_dim_diffuse,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        self.mlp_environment = MLP(
            in_dim=self.direction_encoding.get_out_dim(),
            num_layers=num_layers_environment,
            layer_width=hidden_dim_environment,
            out_dim=environment_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        self.mlp_specular = MLP(
            in_dim=material_dim + environment_dim,
            num_layers=num_layers_specular,
            layer_width=hidden_dim_specular,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

    def _get_density(self, positions: Float[Tensor, '*bs 3']):
        positions_flat = positions.view(-1, 3)
        grid_encoding = self.grid(positions_flat)
        geometry_feat = self.mlp_geometry(grid_encoding)
        density_before_activation, geometry_feat = torch.split(geometry_feat, [1, geometry_feat.shape[-1] - 1], dim=-1)
        density_before_activation = density_before_activation.view(*positions.shape[:-1], 1)
        density = self.density_activation(density_before_activation.to(positions))

        return density, density_before_activation, geometry_feat

    def get_density(self, ray_samples: RaySamples):
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        positions.requires_grad = True

        density, density_before_activation, geometry_feat = self._get_density(positions)
        density = density * selector[..., None]

        return density, (density_before_activation, positions, geometry_feat)

    def get_outputs(
        self,
        ray_samples: RaySamples,
        packed_info: Tensor,
    ) -> Dict[str, Tensor]:
        outputs = {}
        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # density
        with torch.enable_grad():
            density, (density_before_activation, positions, geometry_feat) = self.get_density(ray_samples)
        outputs['density'] = density
        outputs['density_before_activation'] = density_before_activation

        # normals
        with torch.enable_grad():
            if (self.density_activation is trunc_exp) and (not self.soft_density_grad):
                scaled_target = density_before_activation * self.grad_scale
            else:
                scaled_target = F.softplus(density_before_activation) * self.grad_scale
        density_grad = (
            torch.autograd.grad(
                scaled_target,
                positions,
                grad_outputs=torch.ones_like(scaled_target),
                create_graph=True,
            )[0]
            / self.grad_scale
        )
        if density_grad.isinf().any():
            self.grad_scale = self.grad_scale / 2

        if (self.density_activation is trunc_exp) and (not self.soft_density_grad):
            density_grad = density_grad * trunc_exp(density_before_activation)
        outputs['density_grad'] = density_grad

        dens_normal = -torch.nn.functional.normalize(density_grad, dim=-1)
        outputs['dens_normal'] = dens_normal

        grad_dt = density_grad * (ray_samples.frustums.ends - ray_samples.frustums.starts)
        trans_grad = torch.stack([nerfacc.inclusive_sum(grad_dt[..., i], packed_info) for i in range(3)], dim=-1)
        tran_normal = -torch.nn.functional.normalize(trans_grad, dim=-1)
        outputs['tran_normal'] = tran_normal

        positions.requires_grad = False
        positions = positions * 2
        positions_flat = self.position_encoding(positions.view(-1, 3))

        normal_input = []
        if self.predict_normal_with_position:
            normal_input.append(positions_flat)
        if self.predict_normal_with_hashgrid:
            normal_input.append(geometry_feat)
        normal_input = torch.cat(normal_input, dim=-1)
        pred_normal = self.mlp_normal(normal_input).view(*outputs_shape, -1).to(positions)
        pred_normal = torch.nn.functional.normalize(pred_normal, dim=-1)
        outputs['pred_normal'] = pred_normal

        # reflect viewdir
        directions = ray_samples.frustums.directions
        if self.normal_reflect:
            normals = outputs[self.normal_reflect]
            directions = 2.0 * (normals * directions).sum(dim=-1, keepdim=True) * normals - directions
        directions = get_normalized_directions(directions)

        directions_flat = self.direction_encoding(directions.view(-1, 3))

        # color
        material_feat = self.mlp_material(geometry_feat) + geometry_feat
        outputs['material'] = material_feat
        environment_feat = self.mlp_environment(directions_flat)

        specular_input = torch.cat([environment_feat, material_feat], dim=-1)
        specular = self.mlp_specular(specular_input).view(*outputs_shape, -1).to(directions)
        diffuse = self.mlp_diffuse(material_feat).view(*outputs_shape, -1).to(directions)

        diffuse = diffuse - np.log(5)
        specular = specular - np.log(5)
        rgb = diffuse.sigmoid() + specular.sigmoid()
        outputs['diffuse'] = diffuse.sigmoid()
        outputs['specular'] = specular.sigmoid()
        outputs['rgb'] = rgb

        return outputs

    def forward(self, ray_samples: RaySamples, **kwargs) -> Dict[str, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        return self.get_outputs(ray_samples, **kwargs)
