import argparse
import os

import matplotlib.pyplot as plt
import torch

from utils import zones


def run(base_dir: str, name: str, is_sentences: bool):
    file_path = zones.get_model_latest_file_path(base_dir)
    if file_path is None or not os.path.isfile(file_path):
        raise ValueError("{} not found".format(file_path))

    checkpoint = torch.load(file_path)

    train_epoch_losses = checkpoint["train_epoch_losses"]
    val_epoch_losses = checkpoint["val_epoch_losses"]
    train_epoch_cers = checkpoint["train_epoch_cers"]
    val_epoch_cers = checkpoint["val_epoch_cers"]
    train_epoch_wers = checkpoint["train_epoch_wers"]
    val_epoch_wers = checkpoint["val_epoch_wers"]

    num_epochs = len(train_epoch_losses)
    print("num epochs {}".format(num_epochs))
    xs = list(range(num_epochs))

    plt.plot(xs, train_epoch_losses, label="Train Loss")
    plt.plot(xs, val_epoch_losses, label="Val Loss")
    plt.plot(xs, train_epoch_cers, label="Train CER")
    plt.plot(xs, val_epoch_cers, label="Val CER")
    plt.plot(xs, train_epoch_wers, label="Train WER")
    plt.plot(xs, val_epoch_wers, label="Val WER")

    if is_sentences:
        plt.axvline(x=6, c="grey", label="Both->Sent.", linestyle="dashed")
        plt.title("LipNet Sentence Level")

    plt.legend()

    plot_path = zones.get_plot_path(base_dir, name)
    plt.savefig(plot_path)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and unpack the GRID Corpus')
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--sentences', required=False, default=False, action="store_true")

    args = parser.parse_args()
    run(args.base_dir, args.name, args.sentences)

