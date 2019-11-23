import random


def horizontal_flip(batch_img, p=0.5):
    # (T, H, W, C)
    if random.random() > p:
        batch_img = batch_img[:, :, ::-1, ...]
    return batch_img


def color_normalize(batch_img):
    batch_img = batch_img / 255.0
    return batch_img
