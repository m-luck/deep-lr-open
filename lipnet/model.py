import math
import os
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim.optimizer import Optimizer

from utils import zones


class LipNet(torch.nn.Module):
    def __init__(self, base_dir: str, dropout_p=0.5):
        super(LipNet, self).__init__()

        self.base_dir = base_dir

        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.gru1 = nn.GRU(96 * 4 * 8, 256, 1, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)

        self.FC = nn.Linear(512, 27 + 1)
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)
        self._init()

    def _init(self):

        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.constant_(self.conv1.bias, 0)

        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.constant_(self.conv2.bias, 0)

        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        init.constant_(self.conv3.bias, 0)

        init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
        init.constant_(self.FC.bias, 0)

        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(m.weight_ih_l0[i: i + 256],
                              -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i: i + 256])
                init.constant_(m.bias_ih_l0[i: i + 256], 0)
                init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                              -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool3(x)

        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)

        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()

        x, h = self.gru1(x)
        x = self.dropout(x)
        x, h = self.gru2(x)
        x = self.dropout(x)

        x = self.FC(x)
        x = x.permute(1, 0, 2).contiguous()
        return x

    def save(self, epoch: int, optimizer: Optimizer, train_epoch_losses: List[float],
             val_epoch_losses: List[float], train_epoch_cers: List[float], val_epoch_cers: List[float],
             train_epoch_wers: List[float], val_epoch_wers: List[float]):

        file_path = zones.get_model_file_path(self.base_dir, epoch)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_epoch_losses': train_epoch_losses,
            'val_epoch_losses': val_epoch_losses,
            'train_epoch_cers': train_epoch_cers,
            'val_epoch_cers': val_epoch_cers,
            'train_epoch_wers': train_epoch_wers,
            'val_epoch_wers': val_epoch_wers,
        }, file_path)

    def load_existing_model_checkpoint(self, optimizer: Optimizer, target_device: torch.device) -> Tuple[
        int, List[float], List[float], List[float], List[float], List[float], List[float]]:

        file_path = zones.get_model_latest_file_path(self.base_dir)
        if file_path is None or not os.path.isfile(file_path):
            raise ValueError("{} not found".format(file_path))

        checkpoint = torch.load(file_path)

        self.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(target_device)
        self.to(target_device)

        return checkpoint['epoch'], checkpoint["train_epoch_losses"], checkpoint["val_epoch_losses"], checkpoint[
            "train_epoch_cers"], checkpoint["val_epoch_cers"], checkpoint[
                   "train_epoch_wers"], checkpoint["val_epoch_wers"]
