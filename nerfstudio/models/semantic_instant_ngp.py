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
Implementation of Instant NGP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import nerfacc
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    SemanticRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig, NGPModel
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.utils import colormaps


@dataclass
class SemanticInstantNGPModelConfig(InstantNGPModelConfig):
    """Instant NGP Model Config"""

    _target: Type = field(
        default_factory=lambda: SemantiNGPModel
    )  # We can't write `SemantiNGPModel` directly, because `SemantiNGPModel` doesn't exist yet
    """target class to instantiate"""
    semantic_loss_weight: float = 1.0
    pass_semantic_gradients: bool = False



class SemantiNGPModel(NGPModel):
    """Instant NGP model

    Args:
        config: instant NGP configuration to instantiate model
    """

    config: SemanticInstantNGPModelConfig
    field: TCNNNerfactoField

    def __init__(self, config: SemanticInstantNGPModelConfig, metadata: Dict, **kwargs) -> None:
        assert "semantics" in metadata.keys() and isinstance(metadata["semantics"], Semantics)
        self.semantics = metadata["semantics"]
        super().__init__(config=config, **kwargs)
        self.colormap = self.semantics.colors.clone().detach().to(self.device)

    def populate_modules(self):
        super().populate_modules()

        self.field = TCNNNerfactoField(
            aabb=self.scene_box.aabb,
            num_images=self.num_train_data,
            log2_hashmap_size=self.config.log2_hashmap_size,
            max_res=self.config.max_res,
            spatial_distortion=self.scene_contraction,
            use_semantics=True,
            num_semantic_classes=len(self.semantics.classes),
            pass_semantic_gradients=self.config.pass_semantic_gradients,
        )

        # Sampler
        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

        # renderers
        self.renderer_semantics = SemanticRenderer()

        # losses
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean")

    def get_outputs(self, ray_bundle: RayBundle):
        assert self.field is not None
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )

        field_outputs = self.field(ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0]
        
        weights = weights[..., None]

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, 
            ray_samples=ray_samples, 
            ray_indices=ray_indices, 
            num_rays=num_rays
        )
        accumulation = self.renderer_accumulation(
            weights=weights, 
            ray_indices=ray_indices, 
            num_rays=num_rays
        )
        semantic_weight = weights
        if not self.config.pass_semantic_gradients:
            semantic_weight = semantic_weight.detach()
        semantics = self.renderer_semantics(
            semantics=field_outputs[FieldHeadNames.SEMANTICS], 
            weights=semantic_weight,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],
            "semantics": semantics
        }
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch)
        loss_dict["semantics_loss"] = self.config.semantic_loss_weight * self.cross_entropy_loss(
            outputs["semantics"], batch["semantics"][..., 0].long()
        )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        
        semantics = batch["semantics"].to(self.device)
        semantics_colormap = self.colormap.to(self.device)[semantics[...,0]]
        pred_semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1)
        pred_semantics_colormap = self.colormap.to(self.device)[pred_semantic_labels]
        combined_semantics = torch.cat([semantics_colormap, pred_semantics_colormap])
        images_dict["semantics_colormap"] = combined_semantics

        return metrics_dict, images_dict
