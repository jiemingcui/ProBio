import os
import clip
import torch.nn as nn
from datasets import Action_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
# import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
import numpy
from modules.Visual_Prompt import visual_prompt
from utils.Augmentation import get_augmentation
import torch
from utils.Text_Prompt import *
import json

class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)

def ambiguity_split(config):

    ambiguity_file = json.load(open(config.data.ambiguity_list, 'r'))
    label_list = open(config.data.label_list, 'r')
    
    category2label, label2category = {}, {}
    for line in label_list.readlines()[1:]:
        label, category = line.strip().split(',')
        label = int(label)

        category2label[category] = label
        label2category[label] = category

    ambiguity2label, label2ambiguity = {'easy':[], 'mid': [], 'hard': []}, {}
    for mode in ambiguity_file.keys():
        for category in ambiguity_file[mode]:
            label = category2label[category]

            label2ambiguity[label] = mode
            ambiguity2label[mode].append(label)

    return label2ambiguity, ambiguity2label

def validate(epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug):

    label2ambiguity, ambiguity2label = ambiguity_split(config)
    print('Ambiguity split -> easy: {}, mid: {}, hard: {}.'.format(len(ambiguity2label['easy']), len(ambiguity2label['mid']), len(ambiguity2label['hard'])))

    model.eval()
    fusion_model.eval()
    num = [0 for _ in range(len(label2ambiguity.keys()) + 1)]
    corr_1 = [0 for _ in range(len(label2ambiguity.keys()) + 1)]
    corr_5 = [0 for _ in range(len(label2ambiguity.keys()) + 1)]

    with torch.no_grad():
        text_inputs = classes.to(device)
        text_features = model.encode_text(text_inputs)

        for batch_idx, (image, class_id, _) in enumerate(tqdm(val_loader)):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T)
            similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)
            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)
            num[0] += b
            for i in range(b):
                num[class_id[i]] +=1

                if indices_1[i] == class_id[i]:
                    corr_1[0] += 1
                    corr_1[class_id[i]] += 1

                if class_id[i] in indices_5[i]:
                    corr_5[0] += 1
                    corr_5[class_id[i]] += 1

    top1 = [float(corr_1[i]) / num[i] * 100 if num[i] > 0 else -1 for i in range(len(num))]
    top5 = [float(corr_5[i]) / num[i] * 100 if num[i] > 0 else -1 for i in range(len(num))]

    valid_num, mean_top1, mean_top5 = 0, 0, 0
    for idx in range(1, len(num)):
        if num[idx] > 0:
            valid_num += 1
            mean_top1 += top1[idx]
            mean_top5 += top5[idx]
    mean_top1, mean_top5 = mean_top1 / valid_num, mean_top5 / valid_num

    header = "| Category | Top1  | Top5  | Mean Top1 | Mean Top5 |"
    divider = "+" + "-" * (len(header) - 2) + "+"
    print(divider)
    print(header)
    print("| Overall | {0:.2f} | {1:.2f} | {2:.2f}    | {3:.2f}    |".format(top1[0], top5[0], mean_top1, mean_top5))
    print(divider)
    return top1[0]

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               args.log_time)
    # wandb.init(project=config['network']['type'],
    #            name='{}_{}_{}_{}'.format(args.log_time, config['network']['type'], config['network']['arch'],
    #                                      config['data']['dataset']))

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('test.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                                   T=config.data.num_segments, dropout=config.network.drop_out,
                                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32

    transform_val = get_augmentation(False, config)

    # ======================= old_version =======================
    # fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)
    # model_text = TextCLIP(model)
    # model_image = ImageCLIP(model)
    # model_text = torch.nn.DataParallel(model_text).cuda()
    # model_image = torch.nn.DataParallel(model_image).cuda()
    # fusion_model = torch.nn.DataParallel(fusion_model).cuda()

    model_image = ImageCLIP(model).cuda()
    model_text = TextCLIP(model).cuda()
    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments).cuda()

    # wandb.watch(model)
    # wandb.watch(fusion_model)

    val_data = Action_DATASETS(config.data.val_list, config.data.label_list, num_segments=config.data.num_segments,
                        image_tmpl=config.data.image_tmpl,
                        transform=transform_val, random_shift=config.random_shift)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False,
                            pin_memory=True, drop_last=True)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(
            model_text)  # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes, num_text_aug, text_dict = text_prompt(val_data)

    best_prec1 = 0.0
    prec1 = validate(start_epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug)

if __name__ == '__main__':
    main()
