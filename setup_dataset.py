import argparse

from utils.dataset import download_grid


def main(args):
    download_grid.download(args.base_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and unpack the GRID Corpus')
    parser.add_argument('--base_dir', metavar='D', type=str, required=True,
                        help='Base directory to the project data folder')

    args = parser.parse_args()
    main(args)
