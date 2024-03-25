from torch import nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 10, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(10, 20, 5, 1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Linear(16820, 12)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.flatten(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output