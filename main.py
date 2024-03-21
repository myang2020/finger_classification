from setup_data import Data
import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from trainer import Trainer
from network import Network


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
    model = Network()
    trainer = Trainer(train_dataloader, test_dataloader, model)
    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        trainer.train()
        trainer.test()
    print("Done!")

    trainer.save()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = Network().to(device)
    model.load_state_dict(torch.load("model.pth"))

    classes = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    ]

    model.eval()
    x, y = test_dataset[0][0], test_dataset[0][1]
    with torch.no_grad():
        x = x.to(device)
        x = x.unsqueeze(0).float()
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


if __name__ == "__main__":
    main()
