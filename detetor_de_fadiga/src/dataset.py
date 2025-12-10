# src/dataset.py

"""
Este script é responsável por:
- ler as imagens em data/olhos_abertos e data/olhos_fechados
- aplicar as transformações tipo redimensionar e normalizar
- criar os datasets e dataloaders para uso no treinamento

Este dataset lê imagens em duas pastas:
- data/olhos_abertos     → label = 1
- data/olhos_fechados    → label = 0

As imagens são redimensionadas para 64x64 pixels e convertidas em tensores (formato PyTorch).
"""
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.transforms as transforms

class FadigaDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: diretório 'data' que contém as pastas
        - olhos_abertos
        - olhos_fechados
        """

        self.root_dir = root_dir

        # Caminhos das pastas de dataset rotulados
        self.pasta_abertos = os.path.join(root_dir, "olhos_abertos")
        self.pasta_fechados = os.path.join(root_dir, "olhos_fechados")

        # Lista de arquivos em cada classe
        self.arquivos_abertos = [
            os.path.join(self.pasta_abertos, f)
            for f in os.listdir(self.pasta_abertos)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        self.arquivos_fechados = [
            os.path.join(self.pasta_fechados, f)
            for f in os.listdir(self.pasta_fechados)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        # Agrupa em uma única lista com tuplas (caminho, label)
        self.imagens = []
        for caminho in self.arquivos_abertos:
            self.imagens.append((caminho, 1)) # 1 = aberto

        for caminho in self.arquivos_fechados:
            self.imagens.append((caminho, 0)) # 0 = fechado
        
        # Transformação das imagens
        self.transformacao = transforms.Compose([
            transforms.Resize((64, 64)),  # redimensionar
            transforms.ToTensor(),       # converter para tensor
            transforms.Normalize(        # normalização de valores
                mean = [0.5, 0.5, 0.5],
                std = [0.5, 0.5, 0.5]
            )
        ])
    
    def __len__(self):
        # Quantas imagens tem no dataset?
        return len(self.imagens)
    
    def __getitem__(self, idx):
        """
        Retorna:
        - imagem transformada (tensor)
        - label (0 ou 1)
        """
        caminho_img, label = self.imagens[idx]

        # Abrir imagem com PIL
        imagem = Image.open(caminho_img).convert("RGB")

        # Aplicar as transformações
        imagem = self.transformacao(imagem)

        return imagem, label
    
# Função auxiliar para criação do dataloader
def criar_dataloaders(batch_size=32):
    transformacoes = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder("data", transform=transformacoes)

    # Separando treino (80%) e validação (20%)
    tamanho_total = len(dataset)
    tamanho_treino = int(0.8 * tamanho_total)
    tamanho_validacao = tamanho_total - tamanho_treino

    treino_dataset, valid_dataset = random_split(dataset, [tamanho_treino, tamanho_validacao])

    train_loader = DataLoader(treino_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader

# -------------------------------------------------------------
# Teste rápido (executa quando usar: python dataset.py)
# -------------------------------------------------------------
if __name__ == "__main__":
    loader = criar_dataloaders()

    print("Total de imagens:", len(loader.dataset))

    # Pega um único batch
    for imgs, labels in loader:
        print("Batch de imagens:", imgs.shape)
        print("Batch de labels:", labels)
        break