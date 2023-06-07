"""
Instant NGP with depth metric
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

import cv2
import numpy as np
import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.instant_ngp import InstantNGPModelConfig, NGPModel
from nerfstudio.utils import colormaps


@dataclass
class DepthInstantNGPModelConfig(InstantNGPModelConfig):
    _target: Type = field(default_factory=lambda: DepthNGPModel)

class DepthNGPModel(NGPModel):
    config: DepthInstantNGPModelConfig

    def populate_modules(self):
        super().populate_modules()

    def get_outputs(self, ray_bundle: RayBundle):
        outputs = super().get_outputs(ray_bundle)
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metric_dict = super().get_metrics_dict(outputs, batch)
        ground_truth_depth = batch["depth_image"]
        metric_dict["depth_mse"] = torch.nn.functional.mse_loss(outputs["depth"], ground_truth_depth)
        
        return metric_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        mask = outputs["alive_ray_mask"]
        loss_dict = {"rgb_loss": self.rgb_loss(image[mask], outputs["rgb"][mask])}
        return loss_dict

    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], step) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics, images = super().get_image_metrics_and_images(outputs, batch)
        ground_truth_depth = batch["depth_image"]
        
        ground_truth_depth_colormap = colormaps.apply_depth_colormap(ground_truth_depth)
        predicted_depth_colormap = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=torch.min(ground_truth_depth),
            far_plane=torch.max(ground_truth_depth)
        )
        diff_depth_colormap = colormaps.apply_depth_colormap(
            torch.abs(outputs["depth"] - ground_truth_depth),
            accumulation=outputs["accumulation"]
        )
        images["depth"] = torch.cat([ground_truth_depth_colormap, predicted_depth_colormap, diff_depth_colormap])
        metrics["depth_mse"] = torch.nn.functional.mse_loss(outputs["depth"], ground_truth_depth)
        metrics["depth_l1"] = torch.nn.functional.l1_loss(outputs["depth"], ground_truth_depth)

        return metrics, images