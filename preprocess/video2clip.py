import docx
import json
import cv2
import os
import pandas as pd
import random
import itertools

random.seed(1)


def check(train_labels, val_labels, test_labels, categories):
    count = [[0, 0, 0] for _ in categories]
    for idx, split in enumerate([train_labels, val_labels, test_labels]):
        for label in split:
            count[categories.index(label[0])][idx] += 1

    for x, y in zip(categories, count):
        if y[1] == 0 and y[2] == 0:
            print(x, y, sum(y))


def seprate_set(total_labels, categories):
    name = ['label', 'video_id', 'time_start', 'time_end', 'split', 'is_cc']
    val_num = [int(0.2 * len(total_labels[category])) for category in categories]
    test_num = [int(0.1 * len(total_labels[category])) for category in categories]

    train_labels, val_labels, test_labels = [], [], []

    for idx, category in enumerate(categories):
        labels = total_labels[category]

        test_label = random.sample(labels, test_num[idx])
        for _ in test_label:
            labels.remove(_)
            _.append("test")
            _.append(0)
        test_labels.extend(test_label)

        val_label = random.sample(labels, val_num[idx])
        for _ in val_label:
            labels.remove(_)
            _.append("val")
            _.append(0)
        val_labels.extend(val_label)

        for _ in labels:
            _.append("train")
            _.append(0)
        train_labels.extend(labels)

    train_csv = pd.DataFrame(columns=name, data=train_labels)
    train_csv.to_csv("./data/autobio/autobio_train.csv", index=False)

    val_csv = pd.DataFrame(columns=name, data=val_labels)
    val_csv.to_csv("./data/autobio/autobio_val.csv", index=False)

    test_csv = pd.DataFrame(columns=name, data=test_labels)
    test_csv.to_csv("./data/autobio/autobio_test.csv", index=False)

    check(train_labels, val_labels, test_labels, categories)


if __name__ == "__main__":
    video_path = "./data/autobio/videos/"
    total_labels, categories = {}, []
    for root, labels, _ in os.walk(video_path):
        for label in labels:
            total_labels[label] = []
            categories.append(label)
            for file in os.listdir(os.path.join(root, label)):
                total_labels[label].append([label, os.path.join(root, file), 0, 10])
    seprate_set(total_labels, categories)
