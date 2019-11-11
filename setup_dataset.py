import argparse

from utils.dataset import download_grid
from utils.dataset.raw_video import save_mouth_images


def main(args):
    download_grid.download(args.base_dir)
    save_mouth_images(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and unpack the GRID Corpus')
    parser.add_argument('--base_dir', metavar='D', type=str, required=True,
                        help='Base directory to the project data folder')
    parser.add_argument('--ffmpeg_bin_dir', metavar='D', type=str, required=True,
                        help='Bin directory holding ffmpeg')
    args = parser.parse_args()
    main(args)
