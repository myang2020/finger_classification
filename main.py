from setup_data import Data
import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader


def main():
    data = Data(os.path.abspath("/Volumes/DATAUSB/archive"))
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3

    train_size = int(0.8 * len(data.labels))
    test_size = len(data.labels) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    print(len(train_dataset))
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
        img, label = train_dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
