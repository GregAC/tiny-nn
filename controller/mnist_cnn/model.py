import torch.nn as nn
import torch.nn.functional as F
import torch

class MiniCNN(nn.Module):
    def __init__(self):
        super(MiniCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 4)
        self.conv2 = nn.Conv2d(8, 10, 3)
        self.fc1 = nn.Linear(40, 10)
        self.fc2 = nn.Linear(10, 10) # 10 outputs (for 10 digits)
        self.pool1 = nn.AvgPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2,stride=1)

    def forward(self, x):
        x = self.pool1(x)
        pool1_results = x

        x = self.conv1(x)
        x = F.relu(x)
        conv1_results = x

        x = self.pool2(x)

        x = self.conv2(x)
        x = F.relu(x)
        conv2_results = x


        x = self.pool3(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return (x, pool1_results, conv1_results, conv2_results)
