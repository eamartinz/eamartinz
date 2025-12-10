# scr/modelo.py

"""
Modelo CNN para classificação de olhos:
0 = olhos fechados
1 = olhos abertos

Este modelo está configurado como:
- 2 camadas convolucionais
- 1 camada totalmente conectada
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DetetorDeFadigaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Definição da primeira camada convolucional
        # - Entrada: 3 canais (RGB)
        # - Filtros: 16
        # - Tamanho do filtro: 3x3
        # - Função da camada: detecção de padrões (bordas, detalhes)
        self.conv1 = nn.Conv2d(
            in_channels = 3,    # imagem colorida
            out_channels = 16,  # número de filtros
            kernel_size = 3,    # tamanho do filtro
            stride = 1,         # 
            padding = 1         # mantém o tamanho da imagem
        )

        # Definiçã da segunda camada convolucional
        self.conv2 = nn.Conv2d(
            in_channels = 16,
            out_channels = 32,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )

        # Definição da camada totalmente conectada
        # a imagem já passou por:
        # - conv1 + pool
        # - conv2 + pool
        #
        # Usado MaxPool 2x2 duas vezes (reduz o tamanho em 4x)
        #
        # Assumindo imagens redimensionadas para 64x64
        # o tamanho final será:
        # 64 / 2 / 2 = 16
        #
        # Saída final da CNN esper a32 filtros * 16 * 16
        self.fc = nn.Linear(32 * 16 * 16, 2)    # 2 classes

        # Não foi usado Softmax aqui pq o PyTorch usa CrossEntropyLoss que já tem SoftMax

    def forward(self, x):
        # Passo 1: primeira convolução + ReLU + MaxPool
        x = F.relu(self.conv1(x))   # ativação
        x = F.max_pool2d(x, 2)       # reduz pela metade

        # Passo 2: segunda convolução + ReLU + MaxPool
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Flatten: transformar a matriz em vetor
        x = torch.flatten(x, start_dim=1)

        # Camada final
        x = self.fc(x)

        return x


# -----------------------------------------------------------
# Teste rápido (rodado quando executa python modelo.py)
# -----------------------------------------------------------
if __name__ == "__main__":
    modelo = DetetorDeFadigaCNN()
    print(modelo)

    # Criar um batch falso de 4 imagens 64x64 RGB
    entrada = torch.randn(4, 3, 64, 64)
    saida = modelo(entrada)
    print("Saída:", saida)
