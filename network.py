from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten to work with any input dimension
        self.flatten = nn.Flatten()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 10, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(16384,12)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.flatten(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output
