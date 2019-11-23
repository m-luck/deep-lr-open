def load_frame_alignments(alignment_file_path: str, frames=(0, 74)):
    """
    Takes an alignment file and returns a list of 3-tuples with word, start_frame of the word, and end_frame of the word.

    Params:
        alignment_file - relative path to the alignment text file
    Returns:
        List[(word: str, start_frame: int, end_frame: int)]
    """

    slices = []

    with open(alignment_file_path, "r") as f:

        for line in f:
            start, end, word = line.strip('\n').split(' ')
            slices.append((int(start), int(end), word))

    time_start = slices[0][0]
    time_end = slices[-1][1]

    word_frames = []

    for start, end, word in slices:
        start_frame = int(_map_range((time_start, time_end), frames, start))
        end_frame = int(_map_range((time_start, time_end), frames, end))
        word_frames.append((word, start_frame, end_frame))

    return word_frames


def _map_range(a, b, t):
    """
    Takes original range a and target range b, outputting b equivalent of value t.
    """
    (a_start, a_end), (b_start, b_end) = a, b
    return b_start + ((t - a_start) * (b_end - b_start) / (a_end - a_start))
