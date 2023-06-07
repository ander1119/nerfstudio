# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Custom datamanager.
"""
import numpy as np
import torch
import cv2
import time

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers import base_datamanager
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.model_components.fourier_masker import FourierMasker

@dataclass
class MaskedDataManagerConfig(base_datamanager.VanillaDataManagerConfig):
    """A masked datamanager - required to use with .setup()"""

    _target: Type = field(default_factory=lambda: MaskedDataManager)


class MaskedDataManager(base_datamanager.VanillaDataManager):  # pylint: disable=abstract-method
    """Data manager implementation for data that also requires processing depth data. 
    And also generate ray bundle according to mask
    Args:
        config: the DataManagerConfig used to instantiate class
    """

    masker: FourierMasker

    def add_masker(self, masker: FourierMasker):
        self.masker = masker

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        _, nonzero_indice = self.masker.mask(image_batch['image'], step)
        image_batch['mask'] = nonzero_indice
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def create_train_dataset(self) -> DepthDataset:
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")
        return DepthDataset(
            dataparser_outputs=self.train_dataparser_outputs,
        )

    def create_eval_dataset(self) -> DepthDataset:
        return DepthDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
        )