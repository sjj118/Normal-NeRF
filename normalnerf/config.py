"""
Normal-NeRF configuration file.
"""

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from .model import NormalNeRFModelConfig
from .shinyblender_dataparser import ShinyBlenderDataParserConfig
from .shinyblender_dataset import ShinyBlenderDataset

normalnerf = MethodSpecification(
    config=TrainerConfig(
        method_name='normalnerf',
        steps_per_save=2000,
        max_num_iterations=50001,
        mixed_precision=True,
        pipeline=DynamicBatchPipelineConfig(
            target_num_samples=1 << 19,
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[ShinyBlenderDataset],
                dataparser=ShinyBlenderDataParserConfig(alpha_color='white'),
            ),
            model=NormalNeRFModelConfig(
                eval_num_rays_per_chunk=8192,
                cone_angle=0,
                disable_scene_contraction=True,
                alpha_thre=0,
            ),
        ),
        optimizers={
            'fields.grid': {
                'optimizer': AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                'scheduler': ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50001),
            },
            'fields': {
                'optimizer': AdamOptimizerConfig(lr=5e-3, eps=1e-15),
                'scheduler': ExponentialDecaySchedulerConfig(
                    lr_final=1e-4,
                    max_steps=50001,
                    lr_pre_warmup=1e-4,
                    warmup_steps=5000,
                    ramp='cosine',
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis='viewer+tensorboard',
    ),
    description='Normal-NeRF for synthetic scenes.',
)
