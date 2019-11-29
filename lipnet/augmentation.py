import random
from typing import List

import torch
import torchvision.transforms.functional as TF
from PIL.Image import Image
from PIL.Image import NEAREST


def transform(images: List[Image], is_training: bool):
    angle = random.randint(-5, 5)
    should_flip = True if is_training and random.random() < 0.5 else False

    for i, image in enumerate(images):
        if is_training:
            if should_flip:
                image = TF.hflip(image)
            image = TF.rotate(image, angle)
        image = TF.resize(image, (64, 128), interpolation=NEAREST)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.7136, 0.4906, 0.3283], std=[0.113855171, 0.107828568, 0.0917060521])
        images[i] = image

    images = torch.stack(images)
    # Dims: 0 Time, 1 Color, 2 Height, 3 Width -> 0 Time, 2 Height, 3 Width, 1 Color,
    images = images.permute(0, 2, 3, 1)
    return images
