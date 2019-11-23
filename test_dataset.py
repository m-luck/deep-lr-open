import argparse

from lipnet.dataset import GridDataset


def main(args):
    dataset = GridDataset(args.base_dir)
    print(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and unpack the GRID Corpus')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory to the project data folder')
    parser.add_argument('--test_overlapped', required=False, default=False, action='store_true',
                        help='Use test overlapped')
    parser.add_argument('--temporal_aug', default=0.05, type=float, help='temporal jittering probability')
    parser.add_argument('--use_words', default=True, type=bool, help='whether to use word training samples')
    parser.add_argument('--min_timesteps', default=2, type=int, help='min frames, for filtering bad data')
    parser.add_argument('--max_timesteps', default=75, type=int, help='maximum number of frames per sub, for preallocation')
    args = parser.parse_args()
    main(args)
