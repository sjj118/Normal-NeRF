from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

from nerfstudio.data.dataparsers.blender_dataparser import (
    Blender,
    BlenderDataParserConfig,
)


@dataclass
class ShinyBlenderDataParserConfig(BlenderDataParserConfig):
    """ShinyBlender dataset parser config"""

    _target: Type = field(default_factory=lambda: ShinyBlender)
    """target class to instantiate"""
    data: Path = Path('data/ShinyBlender/helmet')
    """Directory specifying location of data."""


@dataclass
class ShinyBlender(Blender):
    """ShinyBlender Dataset"""

    config: BlenderDataParserConfig

    def __init__(self, config: BlenderDataParserConfig):
        super().__init__(config=config)

    def _generate_dataparser_outputs(self, split='train'):
        dataparser_outputs = super()._generate_dataparser_outputs(split=split)

        dataparser_outputs.metadata['normal_filenames'] = [path.parent / path.name.replace('.png', '_normal.png') for path in dataparser_outputs.image_filenames]

        return dataparser_outputs
