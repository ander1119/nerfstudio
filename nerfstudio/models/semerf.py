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
Semantic NeRF-W implementation which should be fast enough to view in the viewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
# from nerfstudio.fields.density_fields import HashMLPDensityField
# from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.fields.semantic_nerf_field import SemanticNerfField
from nerfstudio.model_components.losses import MSELoss, distortion_loss, interlevel_loss
from nerfstudio.model_components.ray_samplers import PDFSampler, ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    SemanticRenderer,
    UncertaintyRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.utils import colormaps


@dataclass
class SemerfModelConfig(NerfactoModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: SemerfWModel)
    semantic_loss_weight: float = 1.0
    num_coarse_samples: int = 64
    """Number of samples in coarse field evaluation"""
    num_importance_samples: int = 128
    """Number of samples in fine field evaluation"""    

class SemerfWModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: SemerfModelConfig

    def __init__(self, config: SemerfModelConfig, metadata: Dict, **kwargs) -> None:
        assert "semantics" in metadata.keys() and isinstance(metadata["semantics"], Semantics)
        self.semantics = metadata["semantics"]
        self.num_classes = len(self.semantics.colors)
        self.colormap = self.semantics.colors.clone()
        super().__init__(config=config, **kwargs)        

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # scene_contraction = SceneContraction(order=float("inf"))

        # if self.config.use_transient_embedding:
        #     raise ValueError("Transient embedding is not fully working for semantic nerf-w.")

        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=16, min_freq_exp=0.0, max_freq_exp=16.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        # Fields
        self.coarse_field = SemanticNerfField(
            num_semantic_classes=self.num_classes,
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )
        self.fine_field = SemanticNerfField(
            num_semantic_classes=self.num_classes,
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # Samplers
        # self.proposal_sampler = ProposalNetworkSampler(
        #     num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
        #     num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
        #     num_proposal_network_iterations=self.config.num_proposal_iterations,
        #     single_jitter=self.config.use_single_jitter,
        # )

        self.sampler_uniform = UniformSampler(self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        # self.renderer_uncertainty = UncertaintyRenderer()
        self.renderer_semantics = SemanticRenderer()

        # losses
        self.rgb_loss = MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        # param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fine_fields"] = list(self.fine_field.parameters())
        param_groups["coarse_fields"] = list(self.coarse_field.parameters())
        return param_groups

    # def get_training_callbacks(
    #     self, training_callback_attributes: TrainingCallbackAttributes
    # ) -> List[TrainingCallback]:
    #     callbacks = []
    #     if self.config.use_proposal_weight_anneal:
    #         # anneal the weights of the proposal network before doing PDF sampling
    #         N = self.config.proposal_weights_anneal_max_num_iters

    #         def set_anneal(step):
    #             # https://arxiv.org/pdf/2111.12077.pdf eq. 18
    #             train_frac = np.clip(step / N, 0, 1)

    #             def bias(x, b):
    #                 return b * x / ((b - 1) * x + 1)

    #             anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
    #             self.proposal_sampler.set_anneal(anneal)

    #         callbacks.append(
    #             TrainingCallback(
    #                 where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
    #                 update_every_num_iters=1,
    #                 func=set_anneal,
    #             )
    #         )
    #     return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        field_outputs_coarse = self.coarse_field.forward(ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse
        )
        semantic_coarse = self.renderer_semantics(
            field_outputs_coarse[FieldHeadNames.SEMANTICS], 
            weights=weights_coarse
        )
        semantic_coarse_labels = torch.argmax(torch.nn.functional.softmax(semantic_coarse, dim=-1), dim=-1)
        depth_coarse = self.renderer_depth(
            weights=weights_coarse, 
            ray_samples=ray_samples_uniform
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)

        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
        field_outputs_fine = self.fine_field.forward(ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        semantic_fine = self.renderer_semantics(
            field_outputs_fine[FieldHeadNames.SEMANTICS],
            weights=weights_fine,
        )
        semantic_fine_labels = torch.argmax(torch.nn.functional.softmax(semantic_fine, dim=-1), dim=-1)
        depth_fine = self.renderer_depth(
            weights=weights_fine, 
            ray_samples=ray_samples_pdf
        )
        accumulation_fine = self.renderer_accumulation(weights=weights_fine)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
            "semantics_coarse": semantic_coarse,
            "semantics_fine": semantic_fine,
            "semantics_coarse_colormap": self.colormap.to(self.device)[semantic_coarse_labels],
            "semantics_fine_colormap": self.colormap.to(self.device)[semantic_fine_labels],
        }

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["fine_psnr"] = self.psnr(outputs["rgb_fine"], image)
        metrics_dict["coarse_psnr"] = self.psnr(outputs["rgb_coarse"], image)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb_coarse"]) + self.rgb_loss(image, outputs["rgb_fine"])

        # semantic loss
        loss_dict["semantics_loss"] = self.config.semantic_loss_weight * \
            (self.cross_entropy_loss(outputs["semantics_fine"], batch["semantics"][..., 0].long()) + \
                self.cross_entropy_loss(outputs["semantics_coarse"], batch["semantics"][...,0].long()))
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        # depth = batch["depth"].to(self.device)
        semantics = batch["semantics"].to(self.device)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        # rgb = torch.clamp(rgb, min=0, max=1)
        semantics_coarse = outputs["semantics_coarse_colormap"]
        semantics_fine = outputs["semantics_fine_colormap"]
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])
        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.config.near_plane,
            far_plane=self.config.far_plane,
        )
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.near_plane,
            far_plane=self.config.far_plane,
        )

        semantics_colormap = self.colormap.to(self.device)[semantics[...,0]]

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)
        combined_semantics = torch.cat([semantics_colormap, semantics_coarse, semantics_fine], dim=1)


        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]  

        fine_psnr = self.psnr(image, rgb_fine)
        coarse_psnr = self.psnr(image, rgb_coarse)
        ssim = self.ssim(image, rgb_fine)
        lpips = self.lpips(image, rgb_coarse)

        # all of these metrics will be logged as scalars
        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr),
            "fine_psnr": float(fine_psnr),
            "fine_ssim": float(ssim),
            "fine_lpips": float(lpips),
        }
        # metrics_dict["lpips"] = float(lpips)

        images_dict = {
            "img": combined_rgb, 
            "depth": combined_depth,
            "acc": combined_acc,
            "semantics": combined_semantics
        }

        return metrics_dict, images_dict

        # for i in range(self.config.num_proposal_iterations):
        #     key = f"prop_depth_{i}"
        #     prop_depth_i = colormaps.apply_depth_colormap(
        #         outputs[key],
        #         accumulation=outputs["accumulation"],
        #     )
        #     images_dict[key] = prop_depth_i

        # semantics
        # semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1)
        # images_dict["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels]

        # # valid mask
        # images_dict["mask"] = batch["mask"].repeat(1, 1, 3)

        # return metrics_dict, images_dict
