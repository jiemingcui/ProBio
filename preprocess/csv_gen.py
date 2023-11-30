import csv
import shutil
import os
import random
import cv2
import glob

import numpy as np
from tqdm import tqdm

from multiprocessing import Process
from multiprocessing import cpu_count
import pandas as pd

def seprate_set(total_labels, set):
    name = ['label', 'video_id', 'time_start', 'time_end', "split", "is_cc"]
    # total_labels = random.shuffle(total_labels)
    val_num = int(0.1 * len(total_labels))
    test_num = int(0.2 * len(total_labels))
    val_labels = random.sample(total_labels, val_num)

    for i in range(len(val_labels)):
        if val_labels[i] in total_labels:
            total_labels.remove(val_labels[i])
            val_labels[i] += ["val", 0]
        else:
            continue

    val_csv = pd.DataFrame(columns=name, data=val_labels)
    val_csv.to_csv("/home/cjm/AutoBio/ActionCLIP/data/autobio_" + set + "/autobio_val.csv", index=None)

    test_labels = random.sample(total_labels, test_num)

    for i in range(len(test_labels)):
        if test_labels[i] in total_labels:
            total_labels.remove(test_labels[i])
            test_labels[i] += ["test", 0]
        else:
            continue

    test_csv = pd.DataFrame(columns=name, data=test_labels)
    test_csv.to_csv("/home/cjm/AutoBio/ActionCLIP/data/autobio_" + set + "/autobio_test.csv", index=None)

    for i in range(len(total_labels)):
        total_labels[i] += ["train", 0]

    train_csv = pd.DataFrame(columns=name, data=total_labels)
    train_csv.to_csv("/home/cjm/AutoBio/ActionCLIP/data/autobio_" + set + "/autobio_train.csv", index=None)



if __name__ == '__main__':
    # for set in ["easy"]:
    for set in ["easy", "mid", "hard"]:
        file_path = "/home/cjm/AutoBio/ActionCLIP/data/autobio_" + set + "/video_clips/"
        frame_name = []
        label_list = os.listdir(file_path)

        for label in label_list:
            file_list = os.listdir(file_path+label)
            for file in file_list:
                frame_name.append([label, file_path+label + "/" + file, 0, 15])


        name = ['label_id']
        kinetics_csv = pd.DataFrame(columns=name, data=label_list)
        kinetics_csv.to_csv("/home/cjm/AutoBio/ActionCLIP/data/autobio_" + set + "/autobio_labels.csv")


        seprate_set(frame_name, set)




