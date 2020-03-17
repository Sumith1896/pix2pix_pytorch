import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage import io, transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

class FacadeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir 
        self.frames = [f for f in os.listdir(root_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.frames[idx])
        image = io.imread(img_name)
        _, W, _ = image.shape
        image0 = image[:, :W // 2, :]
        image1 = image[:, W // 2:, :]

        sample = {'image0': image0, 'image1': image1}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == "main":
    train_set = FacadeDataset("datasets/facades/train")
    val_set = FacadeDataset("datasets/facades/val")
    test_set = FacadeDataset("datasets/facades/test")
