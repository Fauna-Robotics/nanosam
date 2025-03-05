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

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import PIL.Image

import torchvision.transforms as transforms
import torch
from nanosam.utils.predictor import Predictor

# Function to convert PIL image to tensor (H, W, 3)
def pil_to_tensor(image: PIL.Image.Image) -> torch.Tensor:
    """
    Convert a PIL image to a PyTorch tensor with shape (H, W, 3).

    Args:
        image (PIL.Image.Image): Input image.

    Returns:
        torch.Tensor: Image tensor of shape (H, W, 3).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to (C, H, W) with values in [0, 1]
        transforms.Lambda(lambda x: x.permute(1, 2, 0) * 255)  # Convert to (H, W, 3) in range [0, 255]
    ])
    return transform(image).to(torch.uint8)  # Convert to uint8 for consistency

# Function to convert bounding boxes to tensor
def boxes_to_tensor(bboxes):
    """
    Convert bounding boxes to a PyTorch tensor.

    Args:
        bboxes (list of lists): List of bounding boxes [[x1, y1, x2, y2], ...]

    Returns:
        torch.Tensor: Bounding box tensor of shape (1, N, 4).
    """
    return torch.tensor(bboxes, dtype=torch.float32).unsqueeze(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_encoder", type=str, default="/opt/fauna/cache/online_mapping/resnet18_image_encoder.engine"
    )
    parser.add_argument(
        "--mask_decoder", type=str, default="/opt/fauna/cache/online_mapping/mobile_sam_mask_decoder.engine"
    )
    args = parser.parse_args()

    # Instantiate TensorRT predictor
    predictor = Predictor(args.image_encoder, args.mask_decoder)

    image_brambling = PIL.Image.open("assets/brambling.jpeg")
    image_dogs = PIL.Image.open("assets/dogs.jpg")

    # Segment using bounding box
    bbox_dogs = [[134, 112, 850, 759], [671, 173, 1177, 759]]
    bbox_brambling = [[239, 144, 277, 200]]

    # Run inference on an example
    start_time = time.time()
    predictor.set_image(pil_to_tensor(image_brambling).permute(2, 0, 1))
    print(f"Time taken by encoder is {time.time() - start_time}")

    start_time = time.time()
    mask, _, _ = predictor.predict(bboxes=boxes_to_tensor(bbox_brambling))
    mask = (mask[0, 0] > 0).detach().cpu().numpy()
    print(f"Time taken by decoder is {time.time() - start_time}")

    # Draw
    plt.imshow(image_brambling)
    plt.imshow(mask, alpha=0.5)
    x = [bbox_brambling[0][0], bbox_brambling[0][2], bbox_brambling[0][2], bbox_brambling[0][0], bbox_brambling[0][0]]
    y = [bbox_brambling[0][1], bbox_brambling[0][1], bbox_brambling[0][3], bbox_brambling[0][3], bbox_brambling[0][1]]
    plt.plot(x, y, "g-")
    plt.savefig("data/brambling_out.jpg")

    # Run inference on an example
    start_time = time.time()
    predictor.set_image(pil_to_tensor(image_dogs).permute(2,0, 1))
    print(f"Time taken by encoder is {time.time() - start_time}")

    start_time = time.time()
    mask, _, _ = predictor.predict(bboxes=boxes_to_tensor(bbox_dogs))
    mask = mask.detach().cpu().numpy()
    print(f"Time taken by decoder is {time.time() - start_time}")
    mask = (mask[0, 0] > 0) | (mask[1, 0] > 0)

    # Draw
    plt.imshow(image_dogs)
    plt.imshow(mask, alpha=0.5)
    x = [bbox_dogs[1][0], bbox_dogs[1][2], bbox_dogs[1][2], bbox_dogs[1][0], bbox_dogs[1][0]]
    y = [bbox_dogs[1][1], bbox_dogs[1][1], bbox_dogs[1][3], bbox_dogs[1][3], bbox_dogs[1][1]]
    plt.plot(x, y, "g-")
    x = [bbox_dogs[0][0], bbox_dogs[0][2], bbox_dogs[0][2], bbox_dogs[0][0], bbox_dogs[0][0]]
    y = [bbox_dogs[0][1], bbox_dogs[0][1], bbox_dogs[0][3], bbox_dogs[0][3], bbox_dogs[0][1]]
    plt.plot(x, y, "g-")
    plt.savefig("data/dogs_out.jpg")

    print("DONE")