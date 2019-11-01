import os


def get_grid_video_base_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "grid", "video")


def get_grid_video_speaker_dir(base_dir: str, speaker: int) -> str:
    return os.path.join(get_grid_video_base_dir(base_dir), "s_{}".format(speaker))


def get_grid_video_speaker_part_dir(base_dir: str, speaker: int, part: int) -> str:
    return os.path.join(get_grid_video_speaker_dir(base_dir, speaker), "p_{}".format(part))


def get_grid_align_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "grid", "align")


def get_grid_align_speaker_dir(base_dir: str, speaker: int) -> str:
    return os.path.join(get_grid_align_dir(base_dir), "s_{}".format(speaker))
