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

def gen_label_list(input_dir, set_csv, set):
    output_file = "./data/autobio/autobio_" + set + "_frames.txt"
    input_label_csv = "./data/autobio/autobio_labels.csv"
    label_list = pd.read_csv(input_label_csv)
    label_list = label_list.to_dict()
    label_list = {value: key for key, value in label_list['label_id'].items()}
    # label_list = label_list.values.tolist()
    dataset = pd.read_csv(set_csv)
    dataset = dataset.values.tolist()
    video_list = []

    for row in tqdm(dataset):
        label_name = row[0]
        video_name = row[1].split(".")[1].split("/")[-1]
        image_path = input_dir + label_name + "/" + video_name
        image_num = len(os.listdir(image_path)) - 1
        label_num = label_list[label_name]
        f_str = image_path + ' ' + str(image_num) + ' ' + str(label_num) + '\n'
        video_list.append(f_str)
    random.shuffle(video_list)
    with open(output_file, 'w') as f:
        for video in video_list:
            f.write(video)

if __name__ == '__main__':
    video_path = "./data/autobio/videos/"
    image_path = "./data/autobio/clip_images/"
    for set_name in ["train", "test", "val"]:
        set_csv = "./data/autobio/autobio_" + set_name + ".csv"
        gen_label_list(image_path, set_csv, set_name)

