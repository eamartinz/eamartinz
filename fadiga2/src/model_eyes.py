# ARQUIVO 02
# src/model_eyes.py

# dataset_eyes.py
# Carrega imagens de olhos (open/closed)

import torch.nn as nn

class EyesCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(EyesCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
