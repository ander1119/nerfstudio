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

""" Helper functions for visualizing outputs """

from dataclasses import dataclass
from typing import List, Literal, Optional

import matplotlib
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from nerfstudio.utils import colors

Colormaps = Literal["default", "turbo", "viridis", "magma", "inferno", "cividis", "pca", "semantic"]
semantic_colors = [
            [
                0,
                0,
                0
            ],
            [
                139,
                206,
                185
            ],
            [
                203,
                255,
                176
            ],
            [
                134,
                235,
                192
            ],
            [
                133,
                200,
                18
            ],
            [
                111,
                88,
                94
            ],
            [
                212,
                146,
                132
            ],
            [
                59,
                74,
                240
            ],
            [
                9,
                161,
                140
            ],
            [
                185,
                55,
                248
            ],
            [
                232,
                19,
                5
            ],
            [
                64,
                156,
                16
            ],
            [
                181,
                108,
                215
            ],
            [
                207,
                100,
                24
            ],
            [
                101,
                90,
                224
            ],
            [
                233,
                52,
                250
            ],
            [
                238,
                249,
                245
            ],
            [
                132,
                74,
                105
            ],
            [
                129,
                211,
                101
            ],
            [
                21,
                169,
                63
            ],
            [
                113,
                224,
                12
            ],
            [
                88,
                131,
                192
            ],
            [
                220,
                1,
                194
            ],
            [
                226,
                234,
                95
            ],
            [
                6,
                211,
                241
            ],
            [
                93,
                152,
                60
            ],
            [
                66,
                166,
                53
            ],
            [
                150,
                214,
                134
            ],
            [
                21,
                159,
                146
            ],
            [
                197,
                130,
                225
            ],
            [
                167,
                85,
                124
            ],
            [
                223,
                80,
                199
            ],
            [
                26,
                104,
                1
            ],
            [
                159,
                121,
                220
            ],
            [
                23,
                147,
                9
            ],
            [
                24,
                222,
                143
            ],
            [
                237,
                161,
                116
            ],
            [
                181,
                168,
                59
            ],
            [
                50,
                242,
                181
            ],
            [
                242,
                23,
                199
            ],
            [
                131,
                142,
                42
            ],
            [
                217,
                200,
                172
            ],
            [
                236,
                235,
                87
            ],
            [
                255,
                236,
                110
            ],
            [
                102,
                91,
                46
            ],
            [
                209,
                109,
                223
            ],
            [
                43,
                161,
                231
            ],
            [
                69,
                180,
                213
            ],
            [
                110,
                239,
                172
            ],
            [
                187,
                140,
                30
            ],
            [
                32,
                205,
                116
            ],
            [
                26,
                79,
                141
            ],
            [
                12,
                203,
                137
            ],
            [
                72,
                57,
                174
            ],
            [
                61,
                210,
                103
            ],
            [
                222,
                32,
                96
            ],
            [
                216,
                122,
                250
            ],
            [
                220,
                108,
                12
            ],
            [
                193,
                18,
                4
            ],
            [
                24,
                241,
                140
            ],
            [
                19,
                184,
                197
            ],
            [
                212,
                83,
                103
            ],
            [
                253,
                93,
                244
            ],
            [
                236,
                104,
                220
            ],
            [
                51,
                190,
                21
            ],
            [
                81,
                243,
                188
            ],
            [
                234,
                116,
                52
            ],
            [
                110,
                255,
                177
            ],
            [
                77,
                221,
                65
            ],
            [
                13,
                248,
                18
            ],
            [
                169,
                236,
                99
            ],
            [
                18,
                132,
                86
            ],
            [
                214,
                253,
                231
            ],
            [
                212,
                175,
                227
            ],
            [
                191,
                34,
                1
            ],
            [
                150,
                179,
                138
            ],
            [
                77,
                249,
                212
            ],
            [
                76,
                144,
                214
            ],
            [
                186,
                136,
                243
            ],
            [
                122,
                37,
                35
            ],
            [
                240,
                210,
                252
            ],
            [
                89,
                188,
                59
            ],
            [
                163,
                57,
                235
            ],
            [
                35,
                239,
                207
            ],
            [
                68,
                28,
                66
            ],
            [
                51,
                116,
                213
            ],
            [
                86,
                157,
                5
            ],
            [
                222,
                162,
                228
            ],
            [
                200,
                25,
                242
            ],
            [
                5,
                65,
                153
            ],
            [
                91,
                86,
                29
            ],
            [
                126,
                153,
                242
            ],
            [
                231,
                11,
                117
            ],
            [
                62,
                203,
                98
            ],
            [
                24,
                143,
                233
            ],
            [
                68,
                135,
                133
            ],
            [
                104,
                103,
                35
            ],
            [
                120,
                167,
                136
            ],
            [
                170,
                22,
                233
            ],
            [
                184,
                193,
                241
            ],
            [
                205,
                160,
                44
            ],
            [
                63,
                218,
                25
            ]
        ]

@dataclass(frozen=True)
class ColormapOptions:
    """Options for colormap"""

    colormap: Colormaps = "default"
    """ The colormap to use """
    normalize: bool = False
    """ Whether to normalize the input tensor image """
    colormap_min: float = 0
    """ Minimum value for the output colormap """
    colormap_max: float = 1
    """ Maximum value for the output colormap """
    invert: bool = False
    """ Whether to invert the output colormap """


def apply_colormap(
    image: Float[Tensor, "*bs channels"],
    colormap_options: ColormapOptions = ColormapOptions(),
    eps: float = 1e-9,
) -> Float[Tensor, "*bs rgb=3"]:
    """
    Applies a colormap to a tensor image.
    If single channel, applies a colormap to the image.
    If 3 channel, treats the channels as RGB.
    If more than 3 channel, applies a PCA reduction on the dimensions to 3 channels

    Args:
        image: Input tensor image.
        eps: Epsilon value for numerical stability.

    Returns:
        Tensor with the colormap applied.
    """

    # default for rgb images
    if image.shape[-1] == 3:
        return image
    
    if colormap_options.colormap == 'semantic':
        device = image.device
        image = torch.argmax(torch.nn.functional.softmax(image, dim=-1), dim=-1)
        image = torch.tensor(semantic_colors, dtype=torch.uint8).to(device)[image]
        return image

    # rendering depth outputs
    if image.shape[-1] == 1 and torch.is_floating_point(image):
        output = image
        if colormap_options.normalize:
            output = output - torch.min(output)
            output = output / (torch.max(output) + eps)
        output = (
            output * (colormap_options.colormap_max - colormap_options.colormap_min) + colormap_options.colormap_min
        )
        output = torch.clip(output, 0, 1)
        if colormap_options.invert:
            output = 1 - output
        return apply_float_colormap(output, colormap=colormap_options.colormap)

    # rendering boolean outputs
    if image.dtype == torch.bool:
        return apply_boolean_colormap(image)

    if image.shape[-1] > 3:
        return apply_pca_colormap(image)

    raise NotImplementedError


def apply_float_colormap(image: Float[Tensor, "*bs 1"], colormap: Colormaps = "viridis") -> Float[Tensor, "*bs rgb=3"]:
    """Convert single channel to a color image.

    Args:
        image: Single channel image.
        colormap: Colormap for image.

    Returns:
        Tensor: Colored image with colors in [0, 1]
    """
    if colormap == "default":
        colormap = "turbo"

    image = torch.nan_to_num(image, 0)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return torch.tensor(matplotlib.colormaps[colormap].colors, device=image.device)[image_long[..., 0]]


def apply_depth_colormap(
    depth: Float[Tensor, "*bs 1"],
    accumulation: Optional[Float[Tensor, "*bs 1"]] = None,
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    colormap_options: ColormapOptions = ColormapOptions(),
) -> Float[Tensor, "*bs rgb=3"]:
    """Converts a depth image to color for easier analysis.

    Args:
        depth: Depth image.
        accumulation: Ray accumulation used for masking vis.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.
        colormap: Colormap to apply.

    Returns:
        Colored depth image with colors in [0, 1]
    """

    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)
    # depth = torch.nan_to_num(depth, nan=0.0) # TODO(ethan): remove this

    colored_image = apply_colormap(depth, colormap_options=colormap_options)

    if accumulation is not None:
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image


def apply_boolean_colormap(
    image: Bool[Tensor, "*bs 1"],
    true_color: Float[Tensor, "*bs rgb=3"] = colors.WHITE,
    false_color: Float[Tensor, "*bs rgb=3"] = colors.BLACK,
) -> Float[Tensor, "*bs rgb=3"]:
    """Converts a depth image to color for easier analysis.

    Args:
        image: Boolean image.
        true_color: Color to use for True.
        false_color: Color to use for False.

    Returns:
        Colored boolean image
    """

    colored_image = torch.ones(image.shape[:-1] + (3,))
    colored_image[image[..., 0], :] = true_color
    colored_image[~image[..., 0], :] = false_color
    return colored_image


def apply_pca_colormap(image: Float[Tensor, "*bs dim"]) -> Float[Tensor, "*bs rgb=3"]:
    """Convert feature image to 3-channel RGB via PCA. The first three principle
    components are used for the color channels, with outlier rejection per-channel

    Args:
        image: image of arbitrary vectors

    Returns:
        Tensor: Colored image
    """
    original_shape = image.shape
    image = image.view(-1, image.shape[-1])
    _, _, v = torch.pca_lowrank(image)
    image = torch.matmul(image, v[..., :3])
    d = torch.abs(image - torch.median(image, dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    m = 3.0  # this is a hyperparam controlling how many std dev outside for outliers
    rins = image[s[:, 0] < m, 0]
    gins = image[s[:, 1] < m, 1]
    bins = image[s[:, 2] < m, 2]

    image[:, 0] -= rins.min()
    image[:, 1] -= gins.min()
    image[:, 2] -= bins.min()

    image[:, 0] /= rins.max() - rins.min()
    image[:, 1] /= gins.max() - gins.min()
    image[:, 2] /= bins.max() - bins.min()

    image = torch.clamp(image, 0, 1)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return image.view(*original_shape[:-1], 3)
