import random
from typing import List

import torch
import torchvision.transforms.functional as TF
from PIL.Image import Image
from PIL.Image import LANCZOS
import numpy as np


def transform(images: List[Image], is_training: bool, temporal_aug: float):
    angle = random.randint(-5, 5)
    should_flip = True if is_training and random.random() < 0.5 else False

    for i, image in enumerate(images):
        if isinstance(image, np.ndarray) or torch.is_tensor(image):
            image = TF.to_pil_image(image)
        if is_training:
            if should_flip:
                image = TF.hflip(image)
            image = TF.rotate(image, angle)
        image = TF.resize(image, (64, 128), interpolation=LANCZOS)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.6779282, 0.5041335, 0.3496014], std=[0.16670398, 0.11772638, 0.10904928])
        images[i] = image

    images = torch.stack(images)

    """
    temporal jitter taken from https://github.com/sailordiary/LipNet-PyTorch/blob/master/augmentation.py#L79
    sailordiary/LipNet-PyTorch is licensed under the

    BSD 3-Clause "New" or "Revised" License
    """
    if is_training and temporal_aug > 0.0:
        length = images.size(0)
        output = images.clone()
        prob_del = torch.Tensor(length).bernoulli_(temporal_aug)
        prob_dup = prob_del.index_select(0, torch.linspace(length - 1, 0, length).long())
        output_count = 0
        for t in range(0, length):
            if prob_del[t] == 0:
                output[output_count, :, :] = images[t, :, :]
                output_count += 1
            if prob_dup[t] == 1 and output_count > 0:
                output[output_count, :, :] = images[output_count - 1, :, :]
                output_count += 1
        images = output

    # Dims: 0 Time, 1 Color, 2 Height, 3 Width -> 0 Time, 2 Height, 3 Width, 1 Color,
    images = images.permute(0, 2, 3, 1)
    return images
