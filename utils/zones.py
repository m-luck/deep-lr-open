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


def get_grid_align_file_path(base_dir: str, speaker: int, video_name: str) -> str:
    align_speaker_dir = get_grid_align_speaker_dir(base_dir, speaker)
    return os.path.join(align_speaker_dir, "{}.align".format(video_name))


def get_grid_image_base_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "grid", "image")


def get_grid_image_speaker_dir(base_dir: str, speaker: int) -> str:
    """
    Folder containing all folders of mouth images for a given speaker
    """
    return os.path.join(get_grid_image_base_dir(base_dir), "s_{}".format(speaker))


def get_grid_image_speaker_sentence_dir(base_dir: str, speaker: int, sentence_id: str) -> str:
    """
    The folder containing the mouth images of a given sentence (one video)
    """
    speaker_dir = get_grid_image_speaker_dir(base_dir, speaker=speaker)
    return os.path.join(speaker_dir, sentence_id)


def get_dlib_face_predictor_path(base_dir: str) -> str:
    return os.path.join(base_dir, "dlib", "shape_predictor_68_face_landmarks.dat")


def get_resource_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "grid", "resources")


def get_resource_fengdalu_dir(base_dir: str) -> str:
    resources_dir = get_resource_dir(base_dir)
    return os.path.join(resources_dir, "fengdalu")


def get_resource_dataset_split_dir(base_dir: str) -> str:
    resources_dir = get_resource_dir(base_dir)
    return os.path.join(resources_dir, "dataset_split")


def get_resource_dataset_split_file_path(base_dir: str, is_training, is_overlapped):
    prefix = "overlap" if is_overlapped else "unseen"
    postfix = "train" if is_training else "val"
    file_name = "{}_{}.json".format(prefix, postfix)
    file_path = os.path.join(get_resource_dataset_split_dir(base_dir), file_name)
    return file_path


def get_cache_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "grid", "cache")
