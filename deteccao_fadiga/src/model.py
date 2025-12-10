# Definição da rede neural (CNN) para classificação
# src/model.py

"""
Define a arquitetura da rede neural convolucional simples para classificar
imagens de olhos como abertos ou fechados.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

class EyeCNN(nn.Module):
    """
    Rede neural convolucional simples para classificação de olhos:
    saída de duas classes: aberto (1) e fechado (0).
    """
    """"
    MUDANÇA DE RUMO
    def __init__(self):
        # 1 canal (grayscale) -> conv + pool
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # depois das convs + pooling, feature map reduzido
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # ajustar conforme o tamanho da imagem de entrada
        self.fc2 = nn.Linear(128, 2)           # 2 classes: open / closed

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)  # achata o tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Fim do arquivo model.py