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
from nerfstudio.model_components.fourier_masker import FourierMasker
from nerfstudio.models.depth_instant_ngp import (
    DepthInstantNGPModelConfig,
    DepthNGPModel,
)
from nerfstudio.utils import colormaps


@dataclass
class MaskedDepthInstantNGPModelConfig(DepthInstantNGPModelConfig):
    _target: Type = field(default_factory=lambda: MaskedDepthNGPModel)

class MaskedDepthNGPModel(DepthNGPModel):
    
    config: MaskedDepthInstantNGPModelConfig
    masker: FourierMasker

    def add_masker(self, masker: FourierMasker):
        self.masker = masker

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
        # mask = create_mask(batch["image"], max(0, 20 - step // 1000))
        
        for k, _ in batch.items():
            print(k)
        mask = self.masker.mask(None, step)
        images["mask"] = torch.cat([batch["image"], mask * batch["image"]], dim=1)
        metrics["depth_mse"] = torch.nn.functional.mse_loss(outputs["depth"], ground_truth_depth)
        metrics["depth_l1"] = torch.nn.functional.l1_loss(outputs["depth"], ground_truth_depth)

        return metrics, images