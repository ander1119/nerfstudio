# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
A pipeline that dynamically chooses the number of rays to sample (within sample mask). 
"""

from dataclasses import dataclass, field
from typing import Literal, Type

import torch

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.masked_datamanager import MaskedDataManager
from nerfstudio.model_components.fourier_masker import FourierMasker
from nerfstudio.models.base_model import Model
from nerfstudio.models.depth_instant_ngp import MaskedDepthNGPModel
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipeline, DynamicBatchPipelineConfig


@dataclass
class MaskedDynamicBatchPipelineConfig(DynamicBatchPipelineConfig):
    """Dynamic Batch Pipeline Config"""

    _target: Type = field(default_factory=lambda: MaskedDynamicBatchPipeline)

    init_alllowed_frequency: int = 20
    """Initial allowed threshold of frequency"""
    incremented_frequency: int = 1
    """Incremented value of pass-through frequency"""
    steps_per_mask: int = 1000
    """Numbers of steps to gradually change mask according to frequency of image"""
    mask_threshold: int = 5
    """Threshold value to binarize filtered image"""

class MaskedDynamicBatchPipeline(DynamicBatchPipeline):
    """Pipeline with logic for changing the number of rays per batch."""

    config: MaskedDynamicBatchPipelineConfig
    datamanager: MaskedDataManager
    model: MaskedDepthNGPModel
    masker: FourierMasker

    def __init__(
        self,
        config: MaskedDynamicBatchPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        assert isinstance(
            self.datamanager, VanillaDataManager
        ), "DynamicBatchPipeline only works with VanillaDataManager."

        self.masker = FourierMasker(config.init_alllowed_frequency, config.incremented_frequency, config.steps_per_mask, config.mask_threshold)
        self.datamanager.add_masker(self.masker)
        self.model

    
