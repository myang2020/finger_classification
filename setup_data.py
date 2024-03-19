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
            if not os.path.exists(img_dir + "/labels.csv"):
                file = img_dir + '/labels.csv'
                try:
                    with open(file, 'w') as file:
                        for entry in os.listdir(img_dir + "/fingers"):
                            if not entry[:2] == "._":
                                label = entry[-6:-4]
                                file.write(f"{entry},{label}\n")
                except FileNotFoundError:
                    print("The directory does not exist")

        create_annotations()
        self.labels = pd.read_csv(img_dir + "/labels.csv")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, "fingers/" + self.labels.iloc[idx, 0])
        print(img_path)
        image = read_image(img_path)
        label = self.labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label








