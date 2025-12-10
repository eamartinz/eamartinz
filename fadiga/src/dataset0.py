# ARQUIVO 01
# src/dataset.py

# ============================================================
#  Objetivo:
#    - Carregar automaticamente o dataset nas pastas Train / Val / Test
#    - Aplicar transformações básicas nas imagens
#    - Converter as 4 classes originais em 2 classes binárias
#      (0 = Atenção, 1 = Fadiga)
#
#  Classes originais do dataset:
#      Closed_Eyes  -> fadiga
#      Open_Eyes    -> atenção
#      Yawm         -> fadiga
#      No_yawm      -> atenção
# ============================================================

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# ------------------------------------------------------------
# 1) Mapeamento manual das classes
# ------------------------------------------------------------
# ImageFolder atribui índices automaticamente baseados na ordem alfabética das pastas. Exemplo:
#       Closed_Eyes -> 0
#       No_yawm     -> 1
#       Open_Eyes   -> 2
#       Yawm        -> 3
#
# Esses índices não são úteis para o problema.
# Então devem ser convertidos para:
#   - Atenção = 0 = (Open_Eyes, No_yawm)
#   - Fadiga  = 1 = (Closed_Eyes, Yawm)
# ------------------------------------------------------------

binary_class_map = {
    "Closed_Eyes": 1,   # classifica como fadiga
    "Yawm": 1,          # classifica como fadiga
    "Open_Eyes": 0,     # classifica como atenção
    "No_yawm": 0        # classifica como atenção
}

# ------------------------------------------------------------
# 2) Função que sobrescreve as labels originais e converte para 0 ou 1
# ------------------------------------------------------------
def target_transform(original_label, class_to_idx):
    """
    Recebe o rótulo inteiro gerado automaticamente pelo ImageFolder
    (por exemplo 0, 1, 2, 3) e converte para 0 = atenção, 1 = fadiga.

    original_label : int
        índice gerado pelo ImageFolder
    class_to_idx : dict
        mapeamento nome_da_pasta -> índice
    """
    # Inverter o dicionário {nome: índice} -> {índice: nome}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    class_name = idx_to_class[original_label]   # captura o nome da pasta
    return binary_class_map[class_name]         # retorna 0 ou 1

# ------------------------------------------------------------
# 3) Função principal para carregar os datasets
# ------------------------------------------------------------
def load_datasets(data_root, batch_size=32):
    '''
    # RETIRADO PARA INCREMENTAR O DATA AUGMENTATION
    # Transformações básicas aplicadas às imagens
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),    # converte para 1 canal
        transforms.Resize((64,64)),                     # reduz o tamanho (CNN simples)
        transforms.ToTensor(),                          # converte para tensor PyTorch
        transforms.Normalize((0.5), (0.5))              # normaliza pixels
    ])
    '''

    # ------------------------------------------------------------
    # Transformações com Data Augmentation para o TREINO
    # ------------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),

        # Aumentação de dados (Apenas no TREINO)
        transforms.RandomHorizontalFlip(p=0.5),     # espelhamento horizontal
        transforms.RandomRotation(5),               # rotação leve
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),                                          # variação de brilho e contraste
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3)],
            p=0.2
        ),                                          # borrão aleatório (20% das vezes)

        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # ------------------------------------------------------------
    # Transformações SEM data augmentation (Val e Teste)
    # ------------------------------------------------------------
    basic_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    ''' A partir daqui não foi alterado para data augmentation'''
    # Paths
    train_path = os.path.join(data_root, "Train")
    val_path = os.path.join(data_root, "Val")
    test_path = os.path.join(data_root, "Test")

    # Carregar os datasets co ImageFolder
    train_dataset = datasets.ImageFolder(
        train_path,
        transform=transform
    )
    val_dataset = datasets.ImageFolder(
        val_path,
        transform=transform
    )
    test_dataset = datasets.ImageFolder(
        test_path,
        transform=transform
    )

    # Aplicar o target transform para converter rótulos em binário
    train_dataset.targets = [
        target_transform(label, train_dataset.class_to_idx)
        for label in train_dataset.targets
    ]
    val_dataset.targets = [
        target_transform(label, val_dataset.class_to_idx)
        for label in val_dataset.targets
    ]
    test_dataset.targets = [
        target_transform(label, test_dataset.class_to_idx)
        for label in test_dataset.targets
    ]

    # Criar DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader