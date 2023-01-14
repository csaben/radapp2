import torch
import os
import numpy as np
import re
import glob
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
# from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, label=None):
        self.root_dir = root_dir
        self.transform = transform
        self.label = label

        self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        label = self.label
        image = np.array(image)
        return torch.tensor(image, dtype=torch.float32), label

