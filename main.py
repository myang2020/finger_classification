from setup_data import Data
import torch
import matplotlib.pyplot as plt
import os


def main():
    data = Data(os.path.abspath("/Volumes/DATAUSB/archive"))
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

if __name__ == "__main__":
    main()
