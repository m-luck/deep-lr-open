import json
import os
from typing import Dict

import cv2
import editdistance
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from lipnet.augmentation import horizontal_flip, color_normalize
from utils import zones
from utils.dataset import alignments


class GridDataset(Dataset):
    LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z']
    TARGET_TEXT_LENGTH = 20
    TARGET_IMAGES_LENGTH = 45  # 45 is biggest end-start

    def __init__(self, base_dir: str, is_training: bool, is_overlapped: bool):
        self.base_dir = base_dir
        self.is_training = is_training
        self.speakers_dict = self._load_speaker_dict(base_dir, is_training, is_overlapped)

        max_frames = 0

        self.data = []
        for speaker_key in self.speakers_dict:
            speaker = self._get_speaker_number_from_key(speaker_key)
            for sentence_id in self.speakers_dict[speaker_key]:

                if len(os.listdir(zones.get_grid_image_speaker_sentence_dir(base_dir, speaker, sentence_id))) < 75:
                    # skipping for now
                    continue

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        record = self.data[idx]
        images = self._load_mouth_images(self.base_dir, record["speaker"], record["sentence_id"])
        images = images[record["start_frame"]:record["end_frame"]]

        if self.is_training:
            images = horizontal_flip(images, p=0.5)

        images = color_normalize(images)

        images_length = images.shape[0]

        if images_length == 0:
            print("{}, {} is empty".format(record["speaker"], record["sentence_id"]))
            raise Exception("corrupted/bad image folder for {}, {}".format(record["speaker"], record["sentence_id"]))

        images = self._pad_array(images, GridDataset.TARGET_IMAGES_LENGTH)

        word_tensor = self.convert_text_to_array(record["word"])

        word_length = word_tensor.shape[0]
        word_tensor = self._pad_array(word_tensor, GridDataset.TARGET_TEXT_LENGTH)

        spoken_words = " ".join(record["prev_words"])
        gpt2_words = np.zeros(0)  # get gpt2 output

        return {"images_tensor": torch.FloatTensor(images.transpose(3, 0, 1, 2)),
                "images_length": images_length,
                "word_tensor": torch.LongTensor(word_tensor),
                "word_length": word_length,
                "word_str": record["word"],
                }

    @staticmethod
    def _pad_array(array: np.ndarray, target_length: int) -> np.ndarray:
        array = list(array)  # only convert outer dim to list, keep inner as np.ndarray
        size = array[0].shape
        for i in range(target_length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)

    @staticmethod
    def _load_speaker_dict(base_dir: str, is_training: bool, is_overlapped: bool) -> Dict:
        file_path = zones.get_resource_dataset_split_file_path(base_dir, is_training, is_overlapped)
        with open(file_path) as f:
            return json.load(f)

    @staticmethod
    def _get_speaker_number_from_key(speaker_key: str) -> int:
        """ Ex: s_14 -> 14 """
        return int(speaker_key.split('_')[1])

    @staticmethod
    def _load_mouth_images(base_dir: str, speaker: int, sentence_id: str) -> np.ndarray:
        images_dir = zones.get_grid_image_speaker_sentence_dir(base_dir, speaker, sentence_id)
        images = []
        for image_name in os.listdir(images_dir):
            image_file_path = os.path.join(images_dir, image_name)
            image = imageio.imread(image_file_path)
            image = cv2.resize(image, (128, 64), interpolation=cv2.INTER_LANCZOS4)
            images.append(image)
        return np.array(images).astype(np.float32)

    @staticmethod
    def convert_ctc_array_to_text(array: np.ndarray):
        prev_index = -1
        text = []
        for n in array:
            if n < 0 or n >= len(GridDataset.LETTERS) or n == prev_index:
                continue
            text.append(GridDataset.LETTERS[n])
            prev_index = n
        return ''.join(text)

    @staticmethod
    def convert_array_to_text(array: np.ndarray) -> str:
        text = []
        for n in array:
            if n < 0 or n >= len(GridDataset.LETTERS):
                continue
            text.append(GridDataset.LETTERS[n])
        return ''.join(text)

    @staticmethod
    def convert_text_to_array(text: str) -> np.ndarray:
        text = text.upper()
        array = [GridDataset.LETTERS.index(c) for c in text]
        return np.array(array)

    @staticmethod
    def cer(predict, truth):
        cer = [1.0 * editdistance.eval(p[0], p[1]) / len(p[1]) for p in zip(predict, truth)]
        return cer

    def get_data_loader(self, batch_size: int, num_workers: int, shuffle: bool):
        return DataLoader(self,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          drop_last=False)
