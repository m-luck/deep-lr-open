import os
import json

with open(os.path.join("..", "grid", "resources", "dataset_split", "unseen_train.json"), "r") as jfile:
    train_dict = json.load(jfile)

print(train_dict.keys())
with open("corpus_training_set.txt", "w") as corp:
    for i in range(1,37):
        if i != 24 and i != 21:
            speaker_str = f"s_{i}"
            if speaker_str in train_dict.keys():
                speaker_aligns = train_dict[speaker_str]
                speaker_aligns = set(speaker_aligns)
                folder = os.path.join("..","grid","align",speaker_str)
                for filename in os.listdir(folder):
                    if filename.strip(".align") in speaker_aligns:
                        file_path = os.path.join("..","grid","align",speaker_str, filename)
                        print(file_path)

                        with open(file_path, "r") as align:
                            new_sentence = []
                            for line in align:
                                word = line.split(' ')[2].strip("\n")
                                if word != "sil":
                                    new_sentence.append(word)
                            sentence_str = ' '.join(new_sentence)
                            sentence_str = ''.join([sentence_str, '. '])
                            corp.write(sentence_str)
                        