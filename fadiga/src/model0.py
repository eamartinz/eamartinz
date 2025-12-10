# ARQUIVO 02
# src/dataset.py

# ============================================================
#  Objetivo:
#    - Definir uma CNN simples
#    - A rede recebe imagens 1 canal (64x64) e produz 2 saídas:
#          saída[0] -> Atenção
#          saída[1] -> Fadiga
#
#  Estrutura geral da rede (simples, mas eficaz):
#    Entrada: 1 x 64 x 64
#
#    Conv2d(1 -> 16 filtros) + ReLU
#    MaxPool2d
#
#    Conv2d(16 → 32 filtros) + ReLU
#    MaxPool2d
#
#    Flatten
#
#    FC1 Linear -> 128 neurônios
#    FC2 Linear -> num_classes neurônios (saída final)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class FadigaCNN(nn.Module):
    def __init__(self, num_classes=2):
        """
        Docstring para __init__
        param num_classes:
            0 -> atenção
            1 -> fadiga
        """
        super(FadigaCNN, self).__init__()

        # ----------------------------------------------------
        # Criar as camadas convolucionais
        # ----------------------------------------------------

        # Primeira camada:
        # 1 canal de entrada (imagem grayscale)
        # 16 filtros de convolução
        # kernel 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)

        # Segunda camada:
        # 16 canais de entrada
        # 32 filtros
        # kernel 3x3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)

        # ----------------------------------------------------
        # Depois das duas convoluções
        # ----------------------------------------------------
        # Entrada: 64x64
        # Após Conv1 (3x3, sem padding): 62x62
        # Após Pool: 31x31
        #
        # Após Conv2: 29x29
        # Após Pool: 14x14
        #
        # Portanto: 32 canais * 14 * 14 = 6272
        # ----------------------------------------------------
        self.flatten_dim = 32 * 14 * 14
        
        # Por este motivo, o tamanho da entrada do primeiro Linear é 32*14*14
        self.fc1 = nn.Linear(self.flatten_dim, 128)

        # Camada final recebe o número de classes (atenção e fadiga)
        self.fc2 = nn.Linear(128, num_classes)

        # MaxPooling para reduzir a resolução
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # --------------------------------------------------------
    #   DEFINIÇÃO DO FLUXO DA REDE (forward)
    # --------------------------------------------------------
    def forward(self,x):
        # Passa pela primeira convolução -> ReLU -> Pooling
        x = self.pool(F.relu(self.conv1(x)))

        # Passa pela segunda convolução -> ReLU -> Pooling
        x = self.pool(F.relu(self.conv2(x)))

        # Achatar tensor para entrar na parte densa
        x = x.view(x.size(0), -1)   # flatten

        # Primeira camada totalmente conectada
        # FC1 -> ReLU
        x = F.relu(self.fc1(x))

        # Última camada com duas saídas finais
        # FC2 -> saída final
        x = self.fc2(x)

        return x
    
# --------------------------------------------------------
# Função auxiliar para criar o modelo e enviá-lo para GPU
# --------------------------------------------------------
def get_model(device):
    """
    Cria o modelo, envia para a GPU (se disponível) e retorna.
    """
    model = SimpleCNN(num_classes=num_classes)
    return model.to(device)