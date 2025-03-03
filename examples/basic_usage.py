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
import PIL.Image

from nanosam.utils.predictor import Predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_encoder", type=str, default="data/resnet18_image_encoder.engine"
    )
    parser.add_argument(
        "--mask_decoder", type=str, default="data/mobile_sam_mask_decoder.engine"
    )
    args = parser.parse_args()

    # Instantiate TensorRT predictor
    predictor = Predictor(args.image_encoder, args.mask_decoder)

    # Read image and run image encoder
    image_brambling = PIL.Image.open("assets/brambling.jpeg")
    image_dogs = PIL.Image.open("assets/dogs.jpg")

    # Segment using bounding box
    bbox_dogs = [[134, 112, 850, 759], [671, 173, 1177, 759]]
    bbox_brambling = [[239, 144, 277, 200]]

    # Run inference on an example
    predictor.set_image(image_brambling)
    mask, _, _ = predictor.predict(bboxes=bbox_brambling)
    mask = (mask[0, 0] > 0).detach().cpu().numpy()

    # Draw
    plt.imshow(image_brambling)
    plt.imshow(mask, alpha=0.5)
    x = [bbox_brambling[0][0], bbox_brambling[0][2], bbox_brambling[0][2], bbox_brambling[0][0], bbox_brambling[0][0]]
    y = [bbox_brambling[0][1], bbox_brambling[0][1], bbox_brambling[0][3], bbox_brambling[0][3], bbox_brambling[0][1]]
    plt.plot(x, y, "g-")
    plt.savefig("data/brambling_out.jpg")

    # Run inference on an example
    predictor.set_image(image_brambling)
    predictor.set_image(image_dogs)
    mask, _, _ = predictor.predict(bboxes=bbox_dogs)
    mask = mask.detach().cpu().numpy()
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