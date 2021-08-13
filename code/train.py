import argparse
import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import CFG
from train_utils import run
from utils import seed_torch


if __name__ == '__main__':
    seed_torch(CFG.seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=str(0))
    args = parser.parse_args()
    device = args.device

    os.environ['CUDA_VISIBLE_DEVICES'] = device
    print(f"available GPU devices: {device}")

    data = pd.read_csv(CFG.dataset_path)
    data['filepath'] = data['image'].apply(lambda x: os.path.join(CFG.image_folder, x))

    encoder = LabelEncoder()
    data['label_group'] = encoder.fit_transform(data['label_group'])

    run(data)
