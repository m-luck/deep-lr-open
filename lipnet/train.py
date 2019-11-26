import os
from typing import List

import torch
from progressbar import ProgressBar
from torch import nn, optim
from torch.optim.optimizer import Optimizer
from torch import device
import numpy as np

from lipnet.dataset import GridDataset
from lipnet.model import LipNet
from utils import zones


def run(base_dir: str, use_overlapped: bool, batch_size: int, num_workers: int, target_device: device):
    train_dataset = GridDataset(base_dir, is_training=True, is_overlapped=use_overlapped)
    test_dataset = GridDataset(base_dir, is_training=False, is_overlapped=use_overlapped)

    loss_fn = nn.CTCLoss(blank=GridDataset.LETTERS.index(' '), reduction='mean', zero_infinity=True).to(target_device)

    model: LipNet = LipNet(base_dir)
    model.to(target_device)

    optimizer = optim.Adam(model.parameters(),
                           lr=1e-4,
                           weight_decay=0.,
                           amsgrad=True)

    start_epoch, train_losses, test_losses = 0, [], []

    model_file_path = zones.get_model_latest_file_path(base_dir)
    if model_file_path is not None and os.path.isfile(model_file_path):
        start_epoch, train_losses, test_losses = model.load_existing_model_checkpoint(optimizer, target_device)

    train(model, train_dataset, optimizer, loss_fn, batch_size, num_workers, target_device, start_epoch, train_losses, test_losses)


def train(model: LipNet, train_dataset: GridDataset, optimizer: Optimizer, loss_fn: nn.CTCLoss, batch_size: int,
          num_workers: int, target_device: torch.device, start_epoch: int, train_losses: List[float], test_losses: List[float]):

    loader = train_dataset.get_data_loader(batch_size, num_workers, shuffle=True)

    best_train_loss = float('inf') if len(train_losses) == 0 else min(train_losses)
    train_cer = []
    for epoch in range(start_epoch, start_epoch+10):
        print("Starting epoch {} out of {}".format(epoch, start_epoch+10))
        progress_bar = ProgressBar(len(loader)).start()

        batch_losses = []
        for (i, record) in enumerate(loader):
            model.train()
            images_tensor = record['images_tensor'].to(target_device)
            word_tensor = record['word_tensor'].to(target_device)
            images_length = record['images_length'].to(target_device)
            word_length = record['word_length'].to(target_device)

            optimizer.zero_grad()

            logits = model(images_tensor)

            loss = loss_fn(logits.transpose(0, 1).log_softmax(-1), word_tensor, images_length.view(-1), word_length.view(-1))
            loss.backward()

            batch_losses.append(loss.item())

            optimizer.step()

            pred_text = ctc_decode(logits)
            actual_text = record["word_str"]

            if i % 100 == 0:
                for a, p in zip(actual_text, pred_text):
                    print("truth, pred: {}, {}".format(a, p))
                    print(loss.item())

            train_cer.extend(GridDataset.cer(pred_text, actual_text))
            progress_bar.update(i)

        progress_bar.finish()
        epoch_loss = np.mean(batch_losses)
        train_losses.append(epoch_loss)

        if epoch_loss < best_train_loss:
            model.save(epoch, optimizer, train_losses, [])


def ctc_decode(y):
    y = y.argmax(-1)
    return [GridDataset.convert_ctc_array_to_text(y[_]) for _ in range(y.size(0))]
