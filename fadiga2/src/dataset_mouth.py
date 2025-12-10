# ARQUIVO 05
# src/dataset_mouth.py

# dataset_mouth.py
# Carrega imagens de boca (yawn/no_yawn)

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MouthDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.classes = ["No_Yawn", "Yawn"]
        self.images = []

        for idx, classe in enumerate(self.classes):
            pasta = os.path.join(root_dir, classe)
            for img in os.listdir(pasta):
                caminho = os.path.join(pasta, img)
                self.images.append((caminho, idx))

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        caminho_img, label = self.images[idx]
        img = Image.open(caminho_img)
        img = self.transform(img)
        return img, label
