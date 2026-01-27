"""Test PyTorch model for CNN extractor verification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniCNN(nn.Module):
    """Example CNN for MNIST matching the schema in the plan."""

    def __init__(self):
        super().__init__()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=4, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(40, 10)

    def forward(self, x):
        x = self.pool1(x)
        x = F.relu(self.conv1(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


class SimpleCNN(nn.Module):
    """Simpler CNN without initial pooling."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 13 * 13, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


class UnsupportedCNN(nn.Module):
    """CNN with unsupported features for testing error handling."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=4, stride=2)  # Unsupported stride

    def forward(self, x):
        return self.conv1(x)


class UnsupportedActivationCNN(nn.Module):
    """CNN with unsupported activation."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=4, stride=1)

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))  # Unsupported activation
        return x


if __name__ == "__main__":
    # Quick test
    model = MiniCNN()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(f"MiniCNN output shape: {y.shape}")

    model2 = SimpleCNN()
    y2 = model2(x)
    print(f"SimpleCNN output shape: {y2.shape}")
