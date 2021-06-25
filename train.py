import argparse
import os


from config import CFG
import torch
import torch.nn as nn
import pandas as pd
from utils import seed_torch

if __name__ == '__main__':
    seed_torch(DefaultConfig.seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=str(0))
    args = parser.parse_args()
    device = args.device

    os.environ['CUDA_VISIBLE_DEVICES'] = device
    print(f"available GPU devices: {device}")

    data = pd.read_csv('folds.csv')
    data['filepath'] = data['image'].apply(lambda x: os.path.join('train_images', x))

    encoder = LabelEncoder()
    data['label_group'] = encoder.fit_transform(data['label_group'])

    run(data)
