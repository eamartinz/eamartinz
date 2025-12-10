# Novo arquivo de modelo
# src/model.py

# ----------------------------------------------------------------------
# Profundidade:
# 3 blocos convolucionais detectam:
#     - bloco1 -> bordas / texturas
#     - bloco2 -> formas
#     - bloco3 -> padrões mais complexos (olho fechado/aberto, boca aberta)
#
# BatchNorm:
# a rede treina muito mais rápido e sem divergir.
#
# Dropout:
# reduz overfitting e melhora o validation loss.
#
# Padding mantém tamanho da imagem
# a rede não perde muita informação.

import torch
import torch.nn as nn
import torch.nn.functional as F

class FadigaCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(FadigaCNN, self).__init__()

        # ----------
        # BLOCO 1
        # ----------
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # 64 -> 32
        )

        # ----------
        # BLOCO 2
        # ----------
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) # 32 -> 16
        )

        # ----------
        # BLOCO 3
        # ----------
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2) # 16 -> 8
        )

        # ----------
        # CAMADAS FINAIS
        # ----------
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
def get_model(device, num_classes=4):
    model = RobustCNN(num_classes=num_classes)
    return model.to(device)