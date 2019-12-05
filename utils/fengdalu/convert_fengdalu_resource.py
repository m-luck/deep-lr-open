import argparse
import json
import os
from collections import defaultdict

from utils import zones

"""
This file converts Fengdalu's dataset files https://github.com/Fengdalu/LipNet-PyTorch/tree/master/data
into a json representation more easily usable by our code
"""


def convert(input_path: str, output_path: str):
    speaker_dict = defaultdict(list)

    with open(input_path) as f:
        for line in f:
            split = line.split('/')
            speaker = split[0]
            speaker = "s_{}".format(speaker.split('s')[1])
            sentence_id = split[3].strip()

            speaker_dict[speaker].append(sentence_id)

    with open(output_path, 'w') as f:
        json.dump(speaker_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Fengdalus resource files to json')
    parser.add_argument('--base_dir', type=str, required=True)
    args = parser.parse_args()

    input_dir = zones.get_resource_fengdalu_dir(args.base_dir)
    output_dir = zones.get_resource_dataset_split_dir(args.base_dir)

    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, file_name)

        output_file_name = "{}.json".format(file_name.split('.')[0])
        output_path = os.path.join(output_dir, output_file_name)

        convert(input_file_path, output_path)

