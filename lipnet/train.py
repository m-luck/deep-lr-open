import os
from typing import List

import numpy as np
import progressbar
import torch
from torch import device
from torch import nn, optim
from torch.optim.optimizer import Optimizer

from lipnet.dataset import GridDataset
from lipnet.model import LipNet
from utils import zones, progressbar_utils


def run(base_dir: str, use_overlapped: bool, batch_size: int, num_workers: int, target_device: device):
    train_dataset = GridDataset(base_dir, is_training=True, is_overlapped=use_overlapped)
    val_dataset = GridDataset(base_dir, is_training=False, is_overlapped=use_overlapped)

    loss_fn = nn.CTCLoss(blank=GridDataset.LETTERS.index(' '), reduction='mean', zero_infinity=True).to(target_device)

    model: LipNet = LipNet(base_dir)
    model.to(target_device)

    optimizer = optim.Adam(model.parameters(),
                           lr=2e-5,
                           weight_decay=0.,
                           amsgrad=True)

    start_epoch, train_losses, val_losses, train_cers, val_cers = 0, [], [], [], []

    model_file_path = zones.get_model_latest_file_path(base_dir)
    if model_file_path is not None and os.path.isfile(model_file_path):
        last_epoch, train_losses, val_losses, train_cers, val_cers = model.load_existing_model_checkpoint(optimizer,
                                                                                                          target_device)
        start_epoch = last_epoch + 1

    train(model, train_dataset, val_dataset, optimizer, loss_fn, batch_size, num_workers, target_device, start_epoch,
          train_losses,
          val_losses, train_cers, val_cers)


def validate(model: LipNet, val_dataset: GridDataset, loss_fn: nn.CTCLoss, batch_size: int,
             num_workers: int, target_device: torch.device):
    with torch.no_grad():
        model.eval()
        val_loader = val_dataset.get_data_loader(batch_size, num_workers, shuffle=False)

        print("Starting validation")
        progress_bar = progressbar_utils.get_adaptive_progressbar(len(val_loader)).start()

        batch_cers = []
        batch_losses = []
        for (i, record) in enumerate(val_loader):
            images_tensor = record['images_tensor'].to(target_device)
            word_tensor = record['word_tensor'].to(target_device)
            images_length = record['images_length'].to(target_device)
            word_length = record['word_length'].to(target_device)

            logits = model(images_tensor)

            loss = loss_fn(logits.transpose(0, 1).log_softmax(-1), word_tensor, images_length.view(-1),
                           word_length.view(-1))
            loss = loss.cpu().numpy()
            batch_losses.append(loss)

            pred_text = ctc_decode(logits.cpu().numpy(), images_length.cpu().numpy())
            actual_text = record["word_str"]

            cers = GridDataset.cer(pred_text, actual_text)
            batch_cers.append(np.mean(cers))

            progress_bar.update(i)

        progress_bar.finish()

        epoch_loss = np.mean(batch_losses)
        epoch_cer = np.mean(batch_cers)

    return epoch_loss, epoch_cer


def train(model: LipNet, train_dataset: GridDataset, val_dataset: GridDataset, optimizer: Optimizer,
          loss_fn: nn.CTCLoss, batch_size: int,
          num_workers: int, target_device: torch.device, start_epoch: int, train_losses: List[float],
          val_losses: List[float], train_cers: List[float],
          val_cers: List[float]):
    loader = train_dataset.get_data_loader(batch_size, num_workers, shuffle=True)

    best_val_loss = float('inf') if len(val_losses) == 0 else min(val_losses)

    for epoch in range(start_epoch, start_epoch + 20):
        print("Starting epoch {} out of {}".format(epoch, start_epoch + 20))
        progress_bar = progressbar_utils.get_adaptive_progressbar(len(loader)).start()

        batch_losses = []
        batch_cers = []

        for (i, record) in enumerate(loader):
            model.train()
            images_tensor = record['images_tensor'].to(target_device)
            word_tensor = record['word_tensor'].to(target_device)
            images_length = record['images_length'].to(target_device)
            word_length = record['word_length'].to(target_device)

            optimizer.zero_grad()

            logits = model(images_tensor)

            loss = loss_fn(logits.transpose(0, 1).log_softmax(-1), word_tensor, images_length.view(-1),
                           word_length.view(-1))
            loss.backward()

            batch_losses.append(loss.item())

            optimizer.step()

            pred_text = ctc_decode(logits.detach().cpu().numpy(), images_length.cpu().numpy())
            actual_text = record["word_str"]

            cers = GridDataset.cer(pred_text, actual_text)
            batch_cers.append(np.mean(cers))

            progress_bar.update(i)

        progress_bar.finish()

        epoch_loss = np.mean(batch_losses)
        train_losses.append(epoch_loss)
        epoch_cer = np.mean(batch_cers)
        train_cers.append(epoch_cer)

        val_epoch_loss, val_epoch_cer = validate(model, val_dataset, loss_fn, batch_size, num_workers, target_device)
        val_losses.append(val_epoch_loss)
        val_cers.append(val_epoch_cer)

        print(
            "Epoch train loss: {:02f}, train cer: {:02f}, val loss {:02f}, val cer {:02f}".format(epoch_loss, epoch_cer,
                                                                                                  val_epoch_loss,
                                                                                                  val_epoch_cer))

        if val_epoch_loss < best_val_loss:
            print("epoch val loss: {:02f} better than previous best: {:02f}".format(val_epoch_loss, best_val_loss))
            best_val_loss = val_epoch_loss
            model.save(epoch, optimizer, train_losses, val_losses, train_cers, val_cers)


def ctc_decode(y: np.ndarray, images_length: np.ndarray) -> List[str]:
    y = y.argmax(-1)

    result = []
    for i in range(y.shape[0]):
        target_length = images_length[i]
        text = GridDataset.convert_ctc_array_to_text(y[i], target_length)
        result.append(text)
    return result
