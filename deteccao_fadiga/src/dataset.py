# SCRIPT PARA PREPARAÇÃO DO DATASET
# src/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class EyeDataset(Dataset):
    """
    Dataset para imagens dos olhos, separados em classes:
    - open -> olhos abertos
    - closed -> olhos fechados
    Estrutura esperada:
        root_folder/
            open/ <- olhos abertos
            closed/ <- olhos fechados
    """

    def __init__(self, root_folder):
        self.images = []
        self.labels = []
        # Percorrer as pastas das duas classes
        for cls_name, cls_label in [("open", 1), ("closed", 0)]:
            folder = os.path.join(root_folder, cls_name)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(os.path.join(folder, fname))
                    self.labels.append(cls_label)

        # Transformações: converte as imagens para escala de cinza, redimensiona, converte para tensor e normaliza
        self.transform = T.Compose([
            T.Grayscale(),  # imagens em escala de cinza
            T.Resize((64, 64)), # redimensiona para 64x64
            T.ToTensor(),       # converte para tensor
            T.Normalize((0.5,), (0.5,)) # normaliza
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        #img_path = self.images[idx]
        #label = self.labels[idx]
        #img = Image.open(img_path).convert("RGB") # abre a imagem
        #img = self.transform(img)                 # aplica as transformações
        #return img, label
        img = Image.open(self.images[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label
# Fim do arquivo dataset.py
