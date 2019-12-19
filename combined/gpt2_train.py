import os
from typing import List

import numpy as np
import torch
from torch import device
from torch import nn, optim
from torch.optim.optimizer import Optimizer

from lipnet.dataset import GridDataset, InputType
from combined.gpt2_model import LipNet
from utils import zones, progressbar_utils

from language.transformers_run import GPT2_Adapter


def run(base_dir: str, use_overlapped: bool, batch_size: int, num_workers: int, target_device: device,
        temporal_aug: float, cache_in_ram: bool):
    train_dataset = GridDataset(base_dir, is_training=True, is_overlapped=use_overlapped, input_type=InputType.WORDS,
                                temporal_aug=temporal_aug, cache_in_ram=cache_in_ram)
    val_dataset = GridDataset(base_dir, is_training=False, is_overlapped=use_overlapped, input_type=InputType.WORDS,
                              temporal_aug=temporal_aug, cache_in_ram=cache_in_ram)

    loss_fn = nn.CTCLoss(blank=GridDataset.LETTERS.index(' '), reduction='mean', zero_infinity=True).to(target_device)

    model: LipNet = LipNet(base_dir)
    model.to(target_device)

    optimizer = optim.Adam(model.parameters(),
                           lr=1e-4,  # 2e-5,
                           weight_decay=0.,
                           amsgrad=True)

    start_epoch, train_losses, val_losses, train_cers, val_cers, train_wers, val_wers = 0, [], [], [], [], [], []

    model_file_path = zones.get_model_latest_file_path(base_dir)
    if model_file_path is not None and os.path.isfile(model_file_path):
        last_epoch, train_losses, val_losses, train_cers, val_cers, train_wers, val_wers = model.load_existing_model_checkpoint(
            optimizer,
            target_device)
        start_epoch = last_epoch + 1
        if start_epoch >= 200:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 2e-5

    train(model, train_dataset, val_dataset, optimizer, loss_fn, batch_size, num_workers, target_device, start_epoch,
          train_losses,
          val_losses, train_cers, val_cers, train_wers, val_wers)


ranked_predictions_cache = {}


def validate(model: LipNet, val_dataset: GridDataset, loss_fn: nn.CTCLoss, batch_size: int,
             num_workers: int, target_device: torch.device):
    with torch.no_grad():
        model.eval()
        val_loader = val_dataset.get_data_loader(batch_size, num_workers, shuffle=False)

        print("Starting validation")
        progress_bar = progressbar_utils.get_adaptive_progressbar(len(val_loader)).start()

        batch_cers = []
        batch_wers = []
        batch_losses = []

        preds_and_actuals = None

        gpt2adap = GPT2_Adapter(cuda_avail=True, verbose=False)
        dummy_pred = gpt2adap.context_to_flat_prediction_tensor("lip net")[0].tolist()

        for (i, record) in enumerate(val_loader):
            images_tensor = record['images_tensor'].to(target_device)
            text_tensor = record['text_tensor'].to(target_device)
            images_length = record['images_length'].to(target_device)
            text_length = record['text_length'].to(target_device)

            prev_words = record['prev_words']

            ranked_predictions = None

            with open('log', "a+") as l:
                l.write('prev word' + str(prev_words) + "\n")
            if prev_words:
                temp = []
                zipped_words = zip(*prev_words)
                for words in zipped_words:
                    key = ' '.join(words[-3:])
                    with open('log', "a+") as l:
                        l.write('key' + str(key) + "\n")
                    if key not in ranked_predictions_cache:
                        res = gpt2adap.context_to_flat_prediction_tensor(key)[0].tolist()
                        print(res)
                        temp.append(res)
                        ranked_predictions_cache[key] = res
                    else:
                        res = ranked_predictions_cache[key]
                        temp.append(res)
                ranked_predictions = temp
            else:
                ranked_predictions = [dummy_pred] * len(text_length)

            # k = pred_shape.view(-1) / 297
            ranked_predictions = torch.FloatTensor(ranked_predictions).to(target_device)

            logits = model(images_tensor, ranked_predictions)

            loss = loss_fn(logits.transpose(0, 1).log_softmax(-1), text_tensor, images_length.view(-1),
                           text_length.view(-1))
            loss = loss.cpu().numpy()
            batch_losses.append(loss)

            actual_text = record["text_str"]
            pred_text = ctc_decode(logits.cpu().numpy(), actual_text, images_length.cpu().numpy())

            cers = GridDataset.cer(pred_text, actual_text)
            batch_cers.append(np.mean(cers) if cers else 0.1337)

            _add_batch_wer_to_metrics(batch_wers=batch_wers, batch_pred_text=pred_text, batch_actual_text=actual_text)

            if i == len(val_loader) - 1:
                r = min(10, len(pred_text))
                preds_and_actuals = list(zip(pred_text, actual_text))[:r]

            progress_bar.update(i)

        progress_bar.finish()

        epoch_loss = np.mean(batch_losses) if batch_losses else 0.1337
        epoch_cer = np.mean(batch_cers) if batch_cers else 0.1337
        epoch_wer = np.mean(batch_wers) if batch_wers else 0.1337

        for p, a in preds_and_actuals:
            print("pred: {}, actual: {}".format(p, a))

    return epoch_loss, epoch_cer, epoch_wer


def train(model: LipNet, train_dataset: GridDataset, val_dataset: GridDataset, optimizer: Optimizer,
          loss_fn: nn.CTCLoss, batch_size: int,
          num_workers: int, target_device: torch.device, start_epoch: int, train_losses: List[float],
          val_losses: List[float], train_cers: List[float],
          val_cers: List[float], train_wers: List[float],
          val_wers: List[float]):
    loader = train_dataset.get_data_loader(batch_size, num_workers, shuffle=True)

    best_val_loss = float('inf') if len(val_losses) == 0 else min(val_losses)

    gpt2adap = GPT2_Adapter(cuda_avail=True, verbose=False)
    dummy_pred = gpt2adap.context_to_flat_prediction_tensor("lip net")[0].tolist()

    for epoch in range(start_epoch, start_epoch + 10):
        print("Starting epoch {} out of {}".format(epoch, start_epoch + 100))

        progress_bar = progressbar_utils.get_adaptive_progressbar(len(loader)).start()

        batch_losses = []
        batch_cers = []
        batch_wers = []

        for (i, record) in enumerate(loader):
            model.train()
            images_tensor = record['images_tensor'].to(target_device)
            text_tensor = record['text_tensor'].to(target_device)
            images_length = record['images_length'].to(target_device)
            text_length = record['text_length'].to(target_device)
            prev_words = record['prev_words']

            ranked_predictions = None

            with open('log', "a+") as l:
                l.write('prev word' + str(prev_words) + "\n")
            if prev_words:
                temp = []
                zipped_words = zip(*prev_words)
                for words in zipped_words:
                    key = ' '.join(words[-3:])
                    with open('log', "a+") as l:
                        l.write('key' + str(key) + "\n")
                    if key not in ranked_predictions_cache:
                        res = gpt2adap.context_to_flat_prediction_tensor(key)[0].tolist()
                        print(res)
                        temp.append(res)
                        ranked_predictions_cache[key] = res
                    else:
                        res = ranked_prediction_cache[key]
                        temp.append(res)
                ranked_predictions = temp
            else:
                ranked_predictions = [dummy_pred] * len(text_length)

            # k = pred_shape.view(-1) / 297
            ranked_predictions = torch.FloatTensor(ranked_predictions).to(target_device)

            optimizer.zero_grad()

            logits = model(images_tensor, ranked_predictions)

            loss = loss_fn(logits.transpose(0, 1).log_softmax(-1), text_tensor, images_length.view(-1),
                           text_length.view(-1))
            loss.backward()

            batch_losses.append(loss.item())

            optimizer.step()

            actual_text = record["text_str"]
            pred_text = ctc_decode(logits.detach().cpu().numpy(), actual_text, images_length.cpu().numpy())

            cers = GridDataset.cer(pred_text, actual_text)
            batch_cers.append(np.mean(cers))

            _add_batch_wer_to_metrics(batch_wers=batch_wers, batch_pred_text=pred_text, batch_actual_text=actual_text)

            progress_bar.update(i)

        progress_bar.finish()

        epoch_loss = np.mean(batch_losses)
        train_losses.append(epoch_loss)
        epoch_cer = np.mean(batch_cers)
        train_cers.append(epoch_cer)
        epoch_wer = np.mean(batch_wers)
        train_wers.append(epoch_wer)

        val_epoch_loss, val_epoch_cer, val_epoch_wer = validate(model, val_dataset, loss_fn, batch_size, num_workers,
                                                                target_device)
        val_losses.append(val_epoch_loss)
        val_cers.append(val_epoch_cer)
        val_wers.append(val_epoch_wer)

        print(
            "Epoch train loss: {:02f}, train cer: {:02f}, train wer: {:02f}, val loss {:02f}, val cer {:02f}, val wer {:02f}".format(
                epoch_loss, epoch_cer, epoch_wer,
                val_epoch_loss, val_epoch_cer, val_epoch_wer))

        if val_epoch_loss < best_val_loss:
            print("epoch val loss: {:02f} better than previous best: {:02f}".format(val_epoch_loss, best_val_loss))
            best_val_loss = val_epoch_loss
            model.save(epoch, optimizer, train_losses, val_losses, train_cers, val_cers, train_wers, val_wers)


def ctc_decode(y: np.ndarray, actual_text: List[str], images_length: np.ndarray) -> List[str]:
    y = y.argmax(-1)

    result = []
    for i in range(y.shape[0]):
        target_length = images_length[i]
        text = GridDataset.convert_ctc_array_to_text(y[i], target_length)
        result.append(text)
    return result


def _add_batch_wer_to_metrics(batch_wers: List[float], batch_pred_text: List[str], batch_actual_text: List[str]):
    for p_text, a_text in zip(batch_pred_text, batch_actual_text):
        # need to check to see if sentence-type inputs are in batch and only calc wer on them
        sentence_pred_text, sentence_actual_text = [], []
        if " " in a_text:
            sentence_pred_text.append(p_text)
            sentence_actual_text.append(a_text)
        if len(sentence_actual_text) > 0:
            wers = GridDataset.wer(sentence_pred_text, sentence_actual_text)
            batch_wers.append(np.mean(wers))
