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
    parser.add_argument('--num_processes', type=int, required=False, default=5,
                        help='Number of processes to spawn to convert videos to images')
    parser.add_argument('--parallel', required=False, default=False, action="store_true",
                        help='Use multiprocessing')
    args = parser.parse_args()
    main(args)
