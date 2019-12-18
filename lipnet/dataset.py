import json
import os
import pickle
from enum import Enum
from typing import Dict, Optional, List

import editdistance
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from lipnet import augmentation
from utils import zones, progressbar_utils
from utils.dataset import alignments


class InputType(Enum):
    SENTENCES = "SENTENCES"
    WORDS = "WORDS"
    BOTH = "BOTH"


class GridDataset(Dataset):
    LETTERS = ['-', ' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
               'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    TARGET_TEXT_LENGTH = 31
    TARGET_IMAGES_LENGTH = 74

    def __init__(self, base_dir: str, is_training: bool, is_overlapped: bool, input_type: InputType,
                 temporal_aug: Optional[float] = None, cache_in_ram: bool = False):
        self.base_dir = base_dir
        self.is_training = is_training
        self.speakers_dict = self._load_speaker_dict(base_dir, is_training, is_overlapped)
        self.temporal_aug = temporal_aug if temporal_aug is not None else 0.0
        self.input_type = input_type
        self.cache_in_ram = cache_in_ram
        self.data = []
        self.images_cache = {}

        skipped = 0
        video_count = 0

        max_text_len = 0
        print("Loading dataset count: {}".format(len(self.speakers_dict.values())))
        progress_bar = progressbar_utils.get_adaptive_progressbar(len(self.speakers_dict.values())).start()

        for i, speaker_key in enumerate(self.speakers_dict):
            speaker = self._get_speaker_number_from_key(speaker_key)
            for sentence_id in self.speakers_dict[speaker_key]:

                images_dir = zones.get_grid_image_speaker_sentence_dir(base_dir, speaker, sentence_id)
                if len(os.listdir(images_dir)) < 75 and not any(
                        "images.pkl" in file_name for file_name in os.listdir(images_dir)):
                    # skipping videos that didn't successfully convert to 75 images
                    skipped += 1
                    continue
                else:
                    video_count += 1

                align_file_path = zones.get_grid_align_file_path(base_dir, speaker, sentence_id)
                aligns = alignments.load_frame_alignments(align_file_path)

                if self.input_type == InputType.BOTH or self.input_type == InputType.WORDS:
                    prev_words = []

                    for word, start_frame, end_frame in aligns:
                        if word in ("sil", "sp",):
                            continue

                        record = {
                            "text": word,
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "speaker": speaker,
                            "sentence_id": sentence_id,
                            "prev_words": prev_words.copy()
                        }
                        prev_words.append(word)
                        self.data.append(record)
                        if self.cache_in_ram:
                            self._cache_in_ram(speaker, sentence_id)
                if self.input_type == InputType.BOTH or self.input_type == InputType.SENTENCES:
                    words = []
                    min_start_frame = 10000
                    max_end_frame = 0
                    for word, start_frame, end_frame in aligns:
                        min_start_frame = min(min_start_frame, start_frame)
                        max_end_frame = max(max_end_frame, end_frame)
                        if word in ("sil", "sp",):
                            continue
                        words.append(word)
                    sentence = " ".join(words)
                    max_text_len = max(max_text_len, len(sentence))
                    record = {
                        "text": sentence,
                        "start_frame": min_start_frame,
                        "end_frame": max_end_frame,
                        "speaker": speaker,
                        "sentence_id": sentence_id,
                        "prev_words": []
                    }
                    self.data.append(record)
                    if self.cache_in_ram:
                        self._cache_in_ram(speaker, sentence_id)

            progress_bar.update(i)

        progress_bar.finish()
        print("Skipped videos {}/{}={:2f}%".format(skipped, video_count, 100 * skipped / video_count))
        print("max text len {}".format(max_text_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        record = self.data[idx]

        if self.cache_in_ram:
            key = GridDataset._get_image_cache_key(int(record["speaker"]),  record["sentence_id"])
            images = self.images_cache[key]
        else:
            images = self._load_mouth_images(self.base_dir, record["speaker"], record["sentence_id"])
        images = images[record["start_frame"]:record["end_frame"]]

        images = augmentation.transform(images, self.is_training, self.temporal_aug)
        images_length = images.shape[0]  # get length before padding
        images = self._pad_array(images, GridDataset.TARGET_IMAGES_LENGTH)

        text_tensor = self.convert_text_to_array(record["text"])
        text_length = text_tensor.shape[0]  # get length before padding
        text_tensor = self._pad_array(text_tensor, GridDataset.TARGET_TEXT_LENGTH)

        return {"images_tensor": torch.FloatTensor(images.transpose(3, 0, 1, 2)),
                "images_length": images_length,
                "text_tensor": torch.LongTensor(text_tensor),
                "text_length": text_length,
                "text_str": record["text"],
		"prev_words": record["prev_words"]
                }

    def _cache_in_ram(self, speaker_number: int, sentence_id: str):
        images = self._load_mouth_images(self.base_dir, speaker_number, sentence_id)
        key = GridDataset._get_image_cache_key(speaker_number, sentence_id)
        self.images_cache[key] = images

    @staticmethod
    def _get_image_cache_key(speaker: int, sentence_id: str) -> str:
        return "{}_{}".format(speaker, sentence_id)

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
    def _load_mouth_images(base_dir: str, speaker: int, sentence_id: str):
        images_dir = zones.get_grid_image_speaker_sentence_dir(base_dir, speaker, sentence_id)
        images = []
        if any("images.pkl" in file_name for file_name in os.listdir(images_dir)):
            with open(os.path.join(images_dir, "images.pkl"), "rb") as f:
                images = pickle.load(f)
        else:
            for image_name in os.listdir(images_dir):
                image_file_path = os.path.join(images_dir, image_name)
                image = Image.open(image_file_path)
                images.append(image)
        return images

    @staticmethod
    def convert_ctc_array_to_text(array: np.ndarray, target_length: int) -> str:
        array = array[:target_length]

        prev_index = -1
        text = []
        for n in array:
            if n < 0 or n >= len(GridDataset.LETTERS) or n == prev_index:
                continue
            text.append(GridDataset.LETTERS[n])
            prev_index = n
        return ''.join(text).replace('-', '').strip()

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
    def cer(predict: List[str], truth: List[str]) -> List[float]:
        """ Ignore blank tokens in CER
            Greedy ctc decoders ignore it https://github.com/SeanNaren/deepspeech.pytorch/blob/master/decoder.py
        """
        cers = []
        for p, t in zip(predict, truth):
            p = p.replace("-", "").upper()
            t = t.replace("-", "").upper()
            cer = 1.0 * editdistance.eval(p, t) / len(t)
            cers.append(cer)
        return cers

    @staticmethod
    def wer(predict: List[str], truth: List[str]):
        sentence_pairs = [(p[0].upper().split(' '), p[1].upper().split(' ')) for p in zip(predict, truth)]
        #  edit distance lib does WER on lists
        wer = [1.0 * editdistance.eval(s[0], s[1]) / len(s[1]) for s in sentence_pairs]  # s is a List[str]
        return wer

    def get_data_loader(self, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
        return DataLoader(self,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          drop_last=False,
                          pin_memory=True)
