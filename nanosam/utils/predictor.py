# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from torch2trt import TRTModule
from typing import Tuple
import tensorrt as trt
import PIL.Image
import torch
import numpy as np
import torch.nn.functional as F

def load_mask_decoder_engine(path: str):

    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, "rb") as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    mask_decoder_trt = TRTModule(
        engine=engine,
        input_names=[
            "image_embeddings",
            # "point_coords",
            # "point_labels",
            # "mask_input",
            # "has_mask_input",
            "bbox",
        ],
        output_names=["iou_predictions", "low_res_masks"],
    )

    return mask_decoder_trt


def load_image_encoder_engine(path: str):
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, "rb") as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    image_encoder_trt = TRTModule(
        engine=engine, input_names=["image"], output_names=["image_embeddings"]
    )

    return image_encoder_trt


def preprocess_image(image, size: int = 512):

    if isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)

    image_mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
    image_std = torch.tensor([58.395, 57.12, 57.375])[:, None, None]

    image_pil = image
    aspect_ratio = image_pil.width / image_pil.height
    if aspect_ratio >= 1:
        resize_width = size
        resize_height = int(size / aspect_ratio)
    else:
        resize_height = size
        resize_width = int(size * aspect_ratio)

    image_pil_resized = image_pil.resize((resize_width, resize_height))
    image_np_resized = np.asarray(image_pil_resized)
    image_torch_resized = torch.from_numpy(image_np_resized).permute(2, 0, 1)
    image_torch_resized_normalized = (
        image_torch_resized.float() - image_mean
    ) / image_std
    image_tensor = torch.zeros((1, 3, size, size))
    image_tensor[0, :, :resize_height, :resize_width] = image_torch_resized_normalized
    return image_tensor.cuda()

def preprocess_image_tensor(image_tensor: torch.Tensor, size: int = 512):
    """
    Preprocess an image tensor.

    Args:
        image_tensor (torch.Tensor): Input image tensor of shape (C, H, W).
        size (int, optional): Target size for resizing. Defaults to 512.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 3, size, size).
    """
    image_tensor = image_tensor.cuda().float()

    # Ensure input tensor has shape (C, H, W)
    assert image_tensor.dim() == 3, "Input image tensor must have shape (C, H, W)"

    image_mean = torch.tensor([123.675, 116.28, 103.53], device=image_tensor.device).view(3, 1, 1)
    image_std = torch.tensor([58.395, 57.12, 57.375], device=image_tensor.device).view(3, 1, 1)

    _, H, W = image_tensor.shape
    aspect_ratio = W / H

    if aspect_ratio >= 1:
        resize_width = size
        resize_height = int(size / aspect_ratio)
    else:
        resize_height = size
        resize_width = int(size * aspect_ratio)

    # Resize the image tensor
    image_tensor_resized = torch.nn.functional.interpolate(
        image_tensor.unsqueeze(0),
        size=(resize_height, resize_width),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    # Normalize the image
    image_tensor_resized = (image_tensor_resized.float() - image_mean) / image_std

    # Create a padded tensor
    image_tensor_padded = torch.zeros((3, size, size), device=image_tensor.device)
    image_tensor_padded[:, :resize_height, :resize_width] = image_tensor_resized

    return image_tensor_padded.unsqueeze(0)

def preprocess_points(points, image_size, size: int = 1024):
    scale = size / max(*image_size)
    points = points * scale
    return points

def run_mask_decoder(
    mask_decoder_engine,
    features,
    points=None,
    point_labels=None,
    mask_input=None,
    bbox=None,
):
    if points is not None:
        assert point_labels is not None
        assert len(points) == len(point_labels)
        image_point_coords = torch.tensor([points]).float().cuda()
        image_point_labels = torch.tensor([point_labels]).float().cuda()
    if mask_input is None:
        mask_input = torch.zeros(1, 1, 256, 256).float().cuda()
        has_mask_input = torch.tensor([0]).float().cuda()
    else:
        has_mask_input = torch.tensor([1]).float().cuda()
    iou_predictions, low_res_masks = mask_decoder_engine(
        features,
        # image_point_coords,
        # image_point_labels,
        # mask_input,
        # has_mask_input,
        bbox,
    )

    return iou_predictions, low_res_masks


def upscale_mask(mask, image_shape, size=256):

    if image_shape[1] > image_shape[0]:
        lim_x = size
        lim_y = int(size * image_shape[0] / image_shape[1])
    else:
        lim_x = int(size * image_shape[1] / image_shape[0])
        lim_y = size

    mask[:, :, :lim_y, :lim_x]
    mask = F.interpolate(mask[:, :, :lim_y, :lim_x], image_shape, mode="bilinear")

    return mask


class Predictor(object):
    def __init__(
        self,
        image_encoder_engine: str,
        mask_decoder_engine: str,
        image_encoder_size: int = 1024,
        orig_image_encoder_size: int = 1024,
    ):
        self.image_encoder_engine = load_image_encoder_engine(image_encoder_engine)
        self.mask_decoder_engine = load_mask_decoder_engine(mask_decoder_engine)
        self.image_encoder_size = image_encoder_size
        self.orig_image_encoder_size = orig_image_encoder_size

        self.image = None
        self.img_height, self.img_width = None, None
        self.features = None
        self.image_tensor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def set_image(self, image: torch.Tensor):
        self.image = image
        self.img_height, self.img_width = self.image.shape[1], self.image.shape[2]
        self.image_tensor = preprocess_image_tensor(self.image, self.image_encoder_size)
        self.features = self.image_encoder_engine(self.image_tensor)

    def predict(self, points=None, point_labels=None, mask_input=None, bboxes=None):
        bboxes = bboxes.to(self.device)
        bboxes = preprocess_points(bboxes, (self.img_height, self.img_width), self.orig_image_encoder_size)
        mask_iou, low_res_mask = run_mask_decoder(
            self.mask_decoder_engine,
            self.features,
            bbox=bboxes,
        )

        hi_res_mask = upscale_mask(low_res_mask, (self.img_height, self.img_width))

        return hi_res_mask, mask_iou, low_res_mask
