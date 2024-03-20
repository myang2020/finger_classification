from setup_data import Data
import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from trainer import Trainer


def main():
    label_map = {"0L": 0, "1L": 1, "2L": 2, "3L": 3, "4L": 4, "5L": 5,
                 "0R": 6, "1R": 7, "2R": 8, "3R": 9, "4R": 10, "5R": 11}

    def target_transform_func(x):
        return label_map[x]

    data = Data(os.path.abspath("/Volumes/DATAUSB/archive"), target_transform=target_transform_func)
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3

    train_size = int(0.8 * len(data.labels))
    test_size = len(data.labels) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    trainer = Trainer(train_dataloader, test_dataloader)
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        trainer.train()
        trainer.test()
    print("Done!")

    # Check if data mapped correcly
    # for X, y in test_dataloader:
    #     print(f"Shape of X [N, C, H, W]: {X.shape}")
    #     print(y)
    #     break
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    #     img, label = train_dataset[sample_idx]
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(label)
    #     plt.axis("off")
    #     plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()


if __name__ == "__main__":
    main()
