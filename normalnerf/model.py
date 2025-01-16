from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import nerfacc
import torch
from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from torch import nn
from torch.amp.autocast_mode import autocast
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from griddecay import GridDecayLoss
from normalnerf.field import NormalNeRFField
from normalnerf.losses import compute_weighted_mae, pred_normal_loss
from normalnerf.renderers import NormalsRenderer
from normalnerf.scheduler import ExponentialDecayScheduler, Scheduler
from normalnerf.shaders import NormalsShader
from normalnerf.utils import linear_to_srgb


@dataclass
class NormalNeRFModelConfig(ModelConfig):
    """Normal-NeRF Model Config"""

    _target: Type = field(default_factory=lambda: NormalNeRFModel)
    """target class to instantiate"""
    enable_collider: bool = False
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = None
    """Instant NGP doesn't use a collider."""
    grid_resolution: int = 128
    """Resolution of the grid used for the field."""
    grid_levels: int = 1
    """Levels of the grid used for the field."""
    alpha_thre: float = 0.01
    """Threshold for opacity skipping."""
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: Optional[float] = None
    """Minimum step size for rendering."""
    render_max_steps: int = 1000
    """Maximum number of steps per ray when render_step_size is None"""
    near_plane: float = 0.05
    """How far along ray to start sampling."""
    far_plane: float = 1e3
    """How far along ray to stop sampling."""
    net_arch: Dict = field(default_factory=lambda: {})
    """Hyperparameters for configuring the network architecture."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    background_color: Literal['random', 'black', 'white'] = 'white'
    """The color that is given to untrained areas."""
    normal_reflect: Optional[Literal['pred_normal', 'dens_normal', 'tran_normal']] = 'pred_normal'
    """The type of normal that is used for reflection computation"""
    normal_geometry: Optional[Literal['dens_normal', 'tran_normal']] = 'tran_normal'
    """The type of normal that is used to guide the predicted normal."""
    pred_normal_loss_mult: Scheduler[float] = ExponentialDecayScheduler(0.06, 0.003, 20000)
    """Weight for predicted normal loss."""
    normal_grad_ratio: Scheduler[float] = ExponentialDecayScheduler(1e-2, 1, 20000)
    """Proportion of gradients flowing from the predicted normal to the density field"""
    grid_decay_mult: float = 0.01
    """Weight of normalized weight decay on grids from zip-nerf"""
    soft_density_grad: bool = True
    """Whether to enable dual activated densities"""
    density_activation: Literal['exp', 'softplus'] = 'exp'
    """Activation function for density field"""


class NormalNeRFModel(Model):
    """Normal-NeRF model

    Args:
        config: Normal-NeRF configuration to instantiate model
    """

    config: NormalNeRFModelConfig
    field: NormalNeRFField

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.step = 0
        self.eval_num_rays_per_chunk: torch.Tensor
        self.register_buffer('eval_num_rays_per_chunk', torch.tensor(self.config.eval_num_rays_per_chunk))
        self.pred_normal_loss_mult = self.config.pred_normal_loss_mult.get(0)
        self.normal_grad_ratio = self.config.normal_grad_ratio.get(0)

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float('inf'))

        self.field = NormalNeRFField(
            aabb=self.scene_box.aabb,
            normal_reflect=self.config.normal_reflect,
            spatial_distortion=scene_contraction,
            density_activation=self.config.density_activation,
            soft_density_grad=self.config.soft_density_grad,
            **self.config.net_arch,
        )

        self.scene_aabb = nn.Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        if self.config.render_step_size is None:
            # auto step size: ~1000 samples in the base level grid
            self.render_step_size = ((self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2).sum().sqrt().item() / self.config.render_max_steps
        else:
            self.render_step_size = self.config.render_step_size

        # Occupancy Grid.
        self.occupancy_grid = OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        # Sampler
        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            # density_fn=self.field.density_fn,
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method='expected')
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()
        self.grid_decay_loss = GridDecayLoss(self.field.grid.tcnn_encoding)  # type: ignore

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        self.mae = compute_weighted_mae

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        callbacks = []

        def update_occupancy_grid(step: int):
            self.occupancy_grid.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.field.density_fn(x) * self.render_step_size,
            )

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            )
        )

        self.dynamic_params = []
        for name, scheduler in self.config.__dict__.items():
            if isinstance(scheduler, Scheduler):
                self.dynamic_params.append(name)
                callbacks.append(
                    TrainingCallback(
                        where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                        update_every_num_iters=1,
                        func=scheduler.step_cb(self, name),
                    )
                )

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=lambda step: setattr(self, 'step', step),
            )
        )

        return callbacks

    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        param_groups = {}
        param_groups['fields.grid'] = list(self.field.grid.parameters())
        param_groups['fields'] = [param for name, param in self.field.named_parameters() if not name.startswith('grid.')]
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        num_rays = len(ray_bundle)
        # sampling
        with autocast('cuda', enabled=False):
            with torch.no_grad():
                ray_samples, ray_indices = self.sampler(
                    ray_bundle=ray_bundle,
                    near_plane=self.config.near_plane,
                    far_plane=self.config.far_plane,
                    render_step_size=self.render_step_size,
                    alpha_thre=self.config.alpha_thre,
                    cone_angle=self.config.cone_angle,
                )
                ray_samples.spacing_starts = (ray_samples.frustums.starts - self.config.near_plane) / (self.config.far_plane - self.config.near_plane)
                ray_samples.spacing_ends = (ray_samples.frustums.ends - self.config.near_plane) / (self.config.far_plane - self.config.near_plane)

        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        field_outputs = self.field(ray_samples, packed_info=packed_info)
        # accumulate weights
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs['density'][..., 0],
            packed_info=packed_info,
        )[0]
        weights = weights[..., None]
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
        # combine color
        diffuse = self.renderer_rgb(
            rgb=field_outputs['diffuse'],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        specular = nerfacc.accumulate_along_rays(
            weights[..., 0],
            values=field_outputs['specular'],
            ray_indices=ray_indices,
            n_rays=num_rays,
        )
        rgb = diffuse + specular
        rgb = linear_to_srgb(rgb).clip(0, 1)

        depth = self.renderer_depth(
            weights=weights,
            ray_samples=ray_samples,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        pred_normal = self.renderer_normals(
            normals=field_outputs['pred_normal'],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
            normalize=False,
        )
        dens_normal = self.renderer_normals(
            normals=field_outputs['dens_normal'],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
            normalize=False,
        )
        tran_normal = self.renderer_normals(
            normals=field_outputs['tran_normal'],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
            normalize=False,
        )

        def get_metrics_dict(batch):
            image = batch['image'].to(self.device)
            metrics_dict = {}

            metrics_dict['psnr'] = self.psnr(rgb, image)
            if 'normal' in batch:
                normal = batch['normal'].to(self.device) * 2 - 1
                normal = torch.nn.functional.normalize(normal, dim=-1)
                alpha = batch['alpha'].to(self.device)
                w = accumulation * alpha
                metrics_dict['pred_normal_mae'] = self.mae(w, pred_normal, normal)
                metrics_dict['dens_normal_mae'] = self.mae(w, dens_normal, normal)
                metrics_dict['tran_normal_mae'] = self.mae(w, tran_normal, normal)
            if packed_info is not None:
                metrics_dict['num_samples_per_batch'] = packed_info[:, 1].sum()
                metrics_dict['num_samples_per_ray'] = packed_info[:, 1].to(torch.float).mean()
                metrics_dict['max_samples_per_ray'] = packed_info[:, 1].max()
            metrics_dict['accumulation'] = accumulation.sum()
            metrics_dict['grad_scale'] = self.field.grad_scale
            for params in self.dynamic_params:
                metrics_dict[params] = getattr(self, params)
            return metrics_dict

        def get_loss_dict(batch, metrics_dict=None):
            loss_dict = {}
            image = batch['image'].to(self.device)
            self.eval_num_rays_per_chunk[...] = image.shape[0]  # type: ignore

            loss_dict['rgb_loss'] = self.rgb_loss(rgb, image)

            if self.config.grid_decay_mult > 0:
                loss_dict['grid_decay_loss'] = self.config.grid_decay_mult * self.grid_decay_loss()

            if self.config.normal_reflect and self.config.normal_geometry and self.config.normal_reflect == 'pred_normal':
                if self.pred_normal_loss_mult > 0:
                    loss_dict['pred_normal_loss'] = self.pred_normal_loss_mult * pred_normal_loss(
                        weights.detach() * (1 - self.normal_grad_ratio) + weights * self.normal_grad_ratio,
                        field_outputs[self.config.normal_geometry] * self.normal_grad_ratio + field_outputs[self.config.normal_geometry].detach() * (1 - self.normal_grad_ratio),
                        field_outputs[self.config.normal_reflect],
                        num_rays,
                    )
            return loss_dict

        return dict(
            rgb=rgb,
            accumulation=accumulation,
            depth=depth,
            pred_normal=self.normals_shader(pred_normal, accumulation),
            dens_normal=self.normals_shader(dens_normal, accumulation),
            tran_normal=self.normals_shader(tran_normal, accumulation),
            get_metrics_dict=get_metrics_dict,
            get_loss_dict=get_loss_dict,
        )

    def get_metrics_dict(self, outputs, batch):
        return outputs['get_metrics_dict'](batch)

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        return outputs['get_loss_dict'](batch, metrics_dict)

    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch['image'].to(self.device)
        rgb = outputs['rgb']
        combined_rgb = torch.cat([image, rgb], dim=1)

        # # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        # all of these metrics will be logged as scalars
        metrics_dict = {
            'psnr': float(self.psnr(rgb, image).item()),
            'ssim': float(self.ssim(rgb, image)),  # type: ignore
            'lpips': float(self.lpips(rgb, image)),
        }
        images_dict = {
            'img': combined_rgb,
            'accumulation': colormaps.apply_colormap(outputs['accumulation']),
            'depth': colormaps.apply_depth_colormap(outputs['depth'], accumulation=outputs['accumulation']),
            'pred_normal': outputs['pred_normal'],
            'dens_normal': outputs['dens_normal'],
            'tran_normal': outputs['tran_normal'],
        }

        if 'normal' in batch:
            normal = batch['normal'].to(self.device) * 2 - 1
            normal = torch.nn.functional.normalize(normal, dim=-1)
            alpha = batch['alpha'].to(self.device)
            weights = outputs['accumulation'] * alpha
            pred_normal = self.normals_shader.reverse(outputs['pred_normal'], outputs['accumulation'])
            dens_normal = self.normals_shader.reverse(outputs['dens_normal'], outputs['accumulation'])
            tran_normal = self.normals_shader.reverse(outputs['tran_normal'], outputs['accumulation'])
            pred_normal_mae = self.mae(weights, pred_normal, normal)
            dens_normal_mae = self.mae(weights, dens_normal, normal)
            tran_normal_mae = self.mae(weights, tran_normal, normal)
            metrics_dict['pred_normal_mae'] = float(pred_normal_mae.item())
            metrics_dict['dens_normal_mae'] = float(dens_normal_mae.item())
            metrics_dict['tran_normal_mae'] = float(tran_normal_mae.item())
        return metrics_dict, images_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        with torch.no_grad():
            num_rays_per_chunk = min(
                self.config.eval_num_rays_per_chunk,
                int(self.eval_num_rays_per_chunk.item()),
            )
            image_height, image_width = camera_ray_bundle.origins.shape[:2]
            num_rays = len(camera_ray_bundle)
            rand_indices = torch.randperm(num_rays)
            outputs_lists = dict()
            for i in range(0, num_rays, num_rays_per_chunk):
                indices = rand_indices[i : i + num_rays_per_chunk]
                ray_bundle = camera_ray_bundle.flatten()[indices]
                outputs = self.forward(ray_bundle=ray_bundle)
                for output_name, output in outputs.items():  # type: ignore
                    if not torch.is_tensor(output):
                        # TODO: handle lists of tensors as well
                        continue
                    output: torch.Tensor
                    if output.dim() > 0 and output.shape[0] == len(indices):
                        if output_name not in outputs_lists:
                            outputs_lists[output_name] = torch.zeros(
                                (num_rays, *output.shape[1:]),
                                dtype=output.dtype,
                                device=output.device,
                            )
                        outputs_lists[output_name][indices] = output
            outputs = {}
            for output_name, outputs_list in outputs_lists.items():
                outputs[output_name] = outputs_list.view(image_height, image_width, *outputs_list.shape[1:])
            return outputs
