import os

from enum import Enum

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

class Split(Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"

Transform_T = transforms.Compose | nn.Module

class FlowersDataset(Dataset):
    def __init__(
            self, 
            path: str, 
            split: Split = Split.TRAIN, 
            transform: Transform_T | None = None
        ):
        self.path = path
        self.data_path = os.path.join(self.path, 'data')
        self.classes = os.listdir(self.data_path)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.split = split
        with open(os.path.join(self.path, 'splits', f'{self.split.value}.list')) as f:
            self.images = f.read().splitlines()
        self.transform = transform
        if not transform:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        label = self.images[idx].split('/')[0]
        label = self.class_to_idx[label]
        image_path = os.path.join(self.data_path, self.images[idx])
        image = Image.open(image_path)
        image = self.transform(image)
        return image, label

