
def load_alignment_file(file_path: str):
    alignments = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.split(' ')
            start = parts[0]
            end = parts[1]
            word = parts[2].strip()

            alignments.append((start, end, word))

    return alignments
