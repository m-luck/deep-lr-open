import os
import shutil
import tarfile
import requests

from utils import zones


def download(base_dir):
    for speaker in range(1, 35):
        if speaker == 21:
            continue

        if os.path.isdir(zones.get_grid_video_speaker_part_dir(base_dir=base_dir, speaker=speaker, part=2)):
            continue

        print("Downloading speaker {}".format(speaker))

        for part in (1, 2,):
            vid_url = _get_video_url(speaker, part)

            file_path = os.path.join(zones.get_grid_video_speaker_dir(base_dir, speaker), "s{}_{}.tar".format(speaker, part))
            _download_url(vid_url, file_path)

            unpack_path = zones.get_grid_video_speaker_part_dir(base_dir, speaker, part)
            _extract_tar(file_path, unpack_path)

        align_url = _get_alignment_url(speaker)
        file_path = os.path.join(zones.get_grid_align_dir(base_dir), "align_s{}.tar".format(speaker))
        _download_url(align_url, file_path)
        unpack_path = zones.get_grid_align_speaker_dir(base_dir, speaker)
        _extract_tar(file_path, unpack_path)


def _download_url(url: str, file_path: str):
    print("Saving {} to {}".format(url, file_path))
    if os.path.isfile(file_path):
        file_size = _get_file_size(file_path)
        download_size = _get_url_download_size(url)
        if file_size == download_size:
            print("File exists, skipping")
            return
        print("File exists, but download is incomplete. Disk: {}, download: {}".format(file_size, download_size))

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with requests.get(url, stream=True) as r:
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


def _get_url_download_size(url: str) -> int:
    return int(requests.get(url, stream=True).headers['Content-length'])


def _get_file_size(file_path: str) -> int:
    return int(os.path.getsize(file_path))


def _extract_tar(tar_file_path, dest_dir):
    print("Extracting {} to {}".format(tar_file_path, dest_dir))
    f = tarfile.open(tar_file_path)
    f.extractall(path=dest_dir)
    f.close()

    for root, dirs, files in os.walk(dest_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            dest_file_path = os.path.join(dest_dir, file_name)
            if os.path.isfile(dest_file_path):
                continue
            shutil.copy(file_path, dest_dir)

    for root, dirs, files in os.walk(dest_dir):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            shutil.rmtree(dir_path)

    os.remove(tar_file_path)


def _get_video_url(speaker: int, part: int) -> str:
    url = "http://spandh.dcs.shef.ac.uk/gridcorpus/s{}/video/s{}.mpg_6000.part{}.tar".format(speaker, speaker, part)
    return url


def _get_alignment_url(speaker: int) -> str:
    url = "http://spandh.dcs.shef.ac.uk/gridcorpus/s{}/align/s{}.tar".format(speaker, speaker)
    return url
