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

def parse_kinetics_annotations(input_csv, ignore_is_cc=False):
    """Returns a parsed DataFrame.

    arguments:
    ---------
    input_csv: str
        Path to CSV file containing the following columns:
          'YouTube Identifier,Start time,End time,Class label'

    returns:
    -------
    dataset: DataFrame
        Pandas with the following columns:
            'video-id', 'start-time', 'end-time', 'label-name'
    """
    df = pd.read_csv(input_csv)

    return df

def dump_frames(video_path,img_out_path):
    cap = cv2.VideoCapture(video_path)
    if os.path.exists(video_path) == False:
        print('no file:',video_path)

    fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(fcount)
    for i in range(fcount):
        try:
            ret, frame = cap.read()
            assert ret
            frame = cv2.resize(frame,(224,224))
            cv2.imwrite('%s/img_%05d.jpg' % (img_out_path, i), frame)
        except Exception as e:
            print(str(e))
            break

    return fcount

def process_video(data_list,label_list,input_dir,output_dir):
    for row in tqdm(data_list):
        label = row[0]
        clip_id = row[1].split('.')[1]
        # time_start = row[2]
        # time_end = row[3]

        video_name = row[1]
        # video_name = '%s_%06d_%06d.mp4' % (youtube_id,time_start,time_end)
        video_path = os.path.join(input_dir, label.replace(' ','_'), video_name.split('/')[-1])
        # print(video_path)
        img_out_path = os.path.join(output_dir, label.replace(' ','_'), clip_id.split('/')[-1])
        print(img_out_path)
        if os.path.exists(img_out_path) == False:
            os.makedirs(img_out_path)

        frame_count = dump_frames(video_path,img_out_path)


def dump_video_frames(input_dir, set):
    output_dir = "./data/autobio/clip_images"
    input_csv = "./data/autobio/autobio_" + set + ".csv"
    input_label_csv = "./data/autobio/autobio_labels.csv"

    dataset = parse_kinetics_annotations(input_csv)
    label_list = pd.read_csv(input_label_csv)

    data_list = dataset.values.tolist()
    n_processes = 5
    # n_processes = cpu_count()
    processes_list = []
    random.shuffle(data_list)
    for n in range(n_processes):
        sub_list = data_list[
                   n * int(len(data_list) / n_processes + 1): min((n + 1) * int(len(data_list) / n_processes + 1),
                                                                  len(data_list))]
        processes_list.append(Process(target=process_video, \
                                      args=(sub_list, \
                                            label_list, \
                                            input_dir, \
                                            output_dir)))
    for p in processes_list:
        p.start()


if __name__ == '__main__':
    video_path = "./data/autobio/videos/"
    image_path = "./data/autobio/clip_images/"
    for set_name in ["train", "test", "val"]:
        dump_video_frames(video_path, set_name)
