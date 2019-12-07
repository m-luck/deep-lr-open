import argparse

import torch

from lipnet.train import run


def main(args):
    run(args.base_dir, False, batch_size=args.batch_size, num_workers=args.num_workers,
        target_device=torch.device("cuda"), temporal_aug=args.temporal_aug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and unpack the GRID Corpus')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory to the project data folder')
    parser.add_argument('--test_overlapped', required=False, default=False, action='store_true',
                        help='Use test overlapped')
    parser.add_argument('--temporal_aug', required=False, default=0.05, type=float,
                        help='temporal jittering probability')
    parser.add_argument('--batch_size', required=False, default=50, type=int, help='Batch size')
    parser.add_argument('--num_workers', required=False, default=4, type=int, help='Num workers for data loaders')

    args = parser.parse_args()
    main(args)
