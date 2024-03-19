import torch
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torch.utils.data import Dataset

import pandas as pd
import os

class Data(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        def create_annotations():
            # Create annotations file for train if it doesn't exist
            if not os.path.exists(img_dir + "/train_annotations.csv"):
                train_dir = img_dir + '/train'
                file = img_dir + '/train_annotations.csv'
                try:
                    with open(file, 'w') as file:
                        for entry in os.listdir(train_dir):
                            label = entry[-6:-4]
                            file.write(f"{entry},{label}\n")
                except FileNotFoundError:
                    print("The directory does not exist")

            # Create annotations file for test if it doesn't exist
            if not os.path.exists(img_dir + "/test/test_annotations.csv"):
                test_dir = img_dir + '/test'
                file = img_dir + '/test_annotations.csv'
                try:
                    with open(file, 'w') as file:
                        for entry in os.listdir(test_dir):
                            label = entry[-6:-4]
                            file.write(f"{entry},{label}\n")
                except FileNotFoundError:
                    print("The directory does not exist")

        create_annotations()
        self.train_annotations = img_dir + "/train_annotations.csv"
        self.test_annotations = img_dir + "/test_annotations.csv"

    def __len__(self):
        return len(self.train_annotations)






