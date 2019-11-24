import multiprocessing
import os

import numpy as np
import dlib
from skimage.transform import resize
from skimage.io import imsave
import skimage.util
from utils import zones

MOUTH_WIDTH = 100
MOUTH_HEIGHT = 50
HORIZONTAL_PAD = 0.19


def _get_video_frames(args, video_path):
    import skvideo
    skvideo.setFFmpegPath(args.ffmpeg_bin_dir)
    import skvideo.io
    videogen = skvideo.io.vreader(video_path)
    frames = np.array([frame for frame in videogen])
    return frames


def _get_mouth_frames(frames, face_detector, face_predictor):
    """modified from https://github.com/rizkiarm/LipNet/blob/master/lipnet/lipreading/videos.py"""
    normalize_ratio = None
    mouth_frames = []
    for frame in frames:
        dets = face_detector(frame, 1)
        shape = None

        for d in dets:
            shape = face_predictor(frame, d)

        if shape is None:  # Detector doesn't detect face, just return as is
            return frames
        i = -1
        mouth_points = []
        for part in shape.parts():
            i += 1
            if i < 48:  # Only take mouth region
                continue
            mouth_points.append((part.x, part.y))
        np_mouth_points = np.array(mouth_points)

        mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

        if normalize_ratio is None:
            mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
            mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

            normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

        new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
        resized_img = resize(frame, new_img_shape)

        mouth_centroid_norm = mouth_centroid * normalize_ratio

        mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
        mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
        mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
        mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

        mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]

        mouth_frames.append(mouth_crop_image)
    return mouth_frames


def _convert_and_save(args, video_file_path: str, output_dir: str):
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(zones.get_dlib_face_predictor_path(base_dir=args.base_dir))

    try:
        frames = _get_video_frames(args, video_file_path)
    except ValueError as e:
        print(e)
        print("Failed read {}".format(video_file_path))
        return

    mouth_frames = _get_mouth_frames(frames, face_detector, face_predictor)

    os.makedirs(output_dir, exist_ok=True)
    for i, mouth_frame in enumerate(mouth_frames):
        try:
            mouth_frame = skimage.util.img_as_ubyte(mouth_frame)
        except ValueError as e:
            print(e)
            return
        file_path = os.path.join(output_dir, "{}.png".format(i))
        imsave(file_path, mouth_frame)

    print("Finished {}".format(video_file_path))


def save_mouth_images(args):
    conversions = []
    for speaker in range(1, 35):
        if speaker == 21:
            continue
        for part in (1, 2,):
            video_dir = zones.get_grid_video_speaker_part_dir(base_dir=args.base_dir, speaker=speaker, part=part)

            for file_name in os.listdir(video_dir):
                video_file_path = os.path.join(video_dir, file_name)
                sentence_id = os.path.splitext(file_name)[0]
                output_dir = os.path.join(zones.get_grid_image_speaker_dir(args.base_dir, speaker=speaker), sentence_id)
                if os.path.isdir(output_dir) and len(os.listdir(output_dir)) >= 74:
                    continue

                conversions.append((args, video_file_path, output_dir,))

    if args.parallel:
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            pool.starmap(_convert_and_save, conversions)
    else:
        for args, video_file_path, output_dir in conversions:
            _convert_and_save(args, video_file_path, output_dir)
    print("Done")
