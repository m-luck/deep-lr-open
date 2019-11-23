import json
import os

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset

from lipnet.augmentation import horizontal_flip, color_normalize
from utils import zones
from utils.dataset import alignments


class GridDataset(Dataset):
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self, base_dir, is_training=True, is_overlapped=False):
        self.target_text_length = 20
        self.target_images_length = 75
        self.base_dir = base_dir
        self.is_training = is_training
        self.speakers_dict = self._load_speaker_dict(base_dir, is_training, is_overlapped)

        self.data = []
        for speaker_key in self.speakers_dict:
            speaker = self._get_speaker_number_from_key(speaker_key)
            for sentence_id in self.speakers_dict[speaker_key]:
                align_file_path = zones.get_grid_align_file_path(base_dir, speaker, sentence_id)
                aligns = alignments.load_frame_alignments(align_file_path)

                prev_words = []

                for word, start_frame, end_frame in aligns:
                    if word in ("sil", "sp",):
                        continue
                    record = {
                        "word": word,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "speaker": speaker,
                        "sentence_id": sentence_id,
                        "prev_words": prev_words.copy()
                    }
                    prev_words.append(word)
                    self.data.append(record)

    def __getitem__(self, idx):
        record = self.data[idx]
        images = self._load_mouth_images(self.base_dir, record["speaker"], record["sentence_id"])
        images = images[record["start_frame"]:record["end_frame"]]

        if self.is_training:
            images = horizontal_flip(images)

        images = color_normalize(images)

        images_length = images.shape[0]
        images = self._pad_array(images, self.target_images_length)

        word = self._convert_text_to_array(record["word"])
        word_length = word.shape[0]
        word = self._pad_array(word, self.target_text_length)

        spoken_words = " ".join(record["prev_words"])
        gpt2_words = np.zeros(0)  # get gpt2 output

        return {"images": torch.FloatTensor(images.transpose(3, 0, 1, 2)),
                "images_length": images_length,
                "word": torch.LongTensor(word),
                "word_length": word_length,
                }

    @staticmethod
    def _pad_array(array, target_length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(target_length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)

    @staticmethod
    def _load_speaker_dict(base_dir, is_training, is_overlapped):
        file_path = zones.get_resource_dataset_split_file_path(base_dir, is_training, is_overlapped)
        with open(file_path) as f:
            return json.load(f)

    @staticmethod
    def _get_speaker_number_from_key(speaker_key: str) -> int:
        """ Ex: s_14 -> 14 """
        return int(speaker_key.split('_')[1])

    @staticmethod
    def _load_mouth_images(base_dir: str, speaker: int, sentence_id: str):
        images_dir = zones.get_grid_image_speaker_sentence_dir(base_dir, speaker, sentence_id)
        images = []
        for image_name in os.listdir(images_dir):
            image_file_path = os.path.join(images_dir, image_name)
            image = imageio.imread(image_file_path)
            images.append(image)
        return np.array(images).astype(np.float32)

    @staticmethod
    def _convert_array_to_text(array):
        text = []
        for n in array:
            if n < 0 or n >= len(GridDataset.letters):
                continue
            text.append(GridDataset.letters[n])
        return ''.join(text)

    @staticmethod
    def _convert_text_to_array(text):
        text = text.upper()
        array = [GridDataset.letters.index(c) for c in text]
        return np.array(array)
