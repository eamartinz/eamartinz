# ARQUIVO 06
# src/model_mouth.py

# model_mouth.py
# Modelo CNN para detecção de boca (Yawn / No_Yawn)

import torch.nn as nn

class MouthCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(MouthCNN, self).__init__()

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
