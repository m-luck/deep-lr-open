import argparse
import os
from collections import defaultdict

import torch
from PIL.Image import LANCZOS
import numpy as np
import torchvision.transforms.functional as TF

from lipnet.dataset import GridDataset
from utils import zones, progressbar_utils


def run(base_dir: str):
    means = []
    stds = []

    def helper(speakers_dict):
        progress_bar = progressbar_utils.get_adaptive_progressbar(len(speakers_dict.values())).start()

        for i, speaker_key in enumerate(speakers_dict):
            speaker = GridDataset._get_speaker_number_from_key(speaker_key)
            for sentence_id in speakers_dict[speaker_key]:
                if len(os.listdir(zones.get_grid_image_speaker_sentence_dir(base_dir, speaker, sentence_id))) < 75:
                    continue
                images = GridDataset._load_mouth_images(base_dir, speaker, sentence_id)

                images = [TF.resize(image, (64, 128), interpolation=LANCZOS) for image in images]
                images = [TF.to_tensor(image) for image in images]
                images = torch.stack(images).numpy()

                images_mean = np.mean(images, axis=(0, 2, 3))
                images_std = np.std(images, axis=(0, 2, 3))
                means.append(images_mean)
                stds.append(images_std)

                progress_bar.update(i)
        progress_bar.finish()

    speaker_dict = defaultdict(set)
    # Get unique videos from both training sets
    for k, v in GridDataset._load_speaker_dict(base_dir, is_training=True, is_overlapped=True).items():
        speaker_dict[k].update(v)
    for k, v in GridDataset._load_speaker_dict(base_dir, is_training=True, is_overlapped=False).items():
        speaker_dict[k].update(v)

    helper(speaker_dict)

    mean = np.array(means).mean(axis=0)
    std = np.array(stds).mean(axis=0)

    print("mean: {}".format(str(mean)))
    print("std: {}".format(str(std)))

    """
    mean: [0.6779282 0.5041335 0.3496014]
    std: [0.16670398 0.11772638 0.10904928]
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and unpack the GRID Corpus')
    parser.add_argument('--base_dir', type=str, required=True)
    args = parser.parse_args()
    run(args.base_dir)
