import argparse

import torch

from lipnet.dataset import InputType
from lipnet.train import run


def main(args):
    run(args.base_dir, False, batch_size=args.batch_size, num_workers=args.num_workers,
        target_device=torch.device("cuda"), temporal_aug=args.temporal_aug, cache_in_ram=args.cache_in_ram,
        input_type=InputType[args.input_type.upper()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train lipnet')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory to the project data folder')
    parser.add_argument('--test_overlapped', required=False, default=False, action='store_true',
                        help='Use test overlapped')
    parser.add_argument('--temporal_aug', required=False, default=0.05, type=float,
                        help='temporal jittering probability')
    parser.add_argument('--batch_size', required=False, default=50, type=int, help='Batch size')
    parser.add_argument('--num_workers', required=False, default=4, type=int, help='Num workers for data loaders')
    parser.add_argument('--cache_in_ram', required=False, default=False, action='store_true',
                        help='Cache images in ram')
    parser.add_argument('--input_type', required=False, default="sentences",
                        help='Train on "sentences", "words", or "both"')
    args = parser.parse_args()
    main(args)
