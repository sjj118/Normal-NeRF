import os
from typing import Dict

import cv2
import torch
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path


class ShinyBlenderDataset(InputDataset):
    """Dataset that returns images and normals.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ['normal', 'alpha']

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert 'normal_filenames' in dataparser_outputs.metadata.keys() and dataparser_outputs.metadata['normal_filenames'] is not None
        self.normal_filenames = self.metadata['normal_filenames']

    def get_metadata(self, data: Dict) -> Dict:
        metadata = {}
        normal_filepath = self.normal_filenames[data['image_idx']]
        if os.path.isfile(normal_filepath):
            normal = cv2.imread(str(normal_filepath.absolute()), cv2.IMREAD_COLOR)[..., :3]
            normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
            normal = torch.from_numpy(normal.astype('float32')) / 255
            metadata['normal'] = normal

        return metadata

    def get_image(self, image_idx: int):
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype('float32') / 255.0)
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert image.shape[-1] == 4
            alpha = image[:, :, -1:]
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        else:
            image = image[:, :, :3]
            alpha = None
        return image, alpha

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        image, alpha = self.get_image(image_idx)
        data = {'image_idx': image_idx, 'image': image, 'alpha': alpha}
        if self._dataparser_outputs.mask_filenames is not None:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data['mask'] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert data['mask'].shape[:2] == data['image'].shape[:2], f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data
