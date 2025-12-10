#!/usr/bin/env python3
"""
Simple evaluator: classifica todas as imagens em `data/Test_images` como
"Olhos fechados" ou "Olhos abertos" e imprime nome, resultado e probabilidades.
"""
import os
from pathlib import Path
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms


class EyeClassifierCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(EyeClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def get_transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def classify_folder(model, device, folder='data/Test_images'):
    folder = Path(folder)
    if not folder.exists():
        print(f"Pasta de imagens não encontrada: {folder}")
        return

    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    images = []
    for e in exts:
        images.extend(sorted(folder.glob(e)))

    if not images:
        print(f"Nenhuma imagem encontrada em: {folder}")
        return

    transform = get_transform()
    softmax = nn.Softmax(dim=1)
    label_map = {0: 'Olhos fechados', 1: 'Olhos abertos'}

    model.eval()
    with torch.no_grad():
        for img_path in images:
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(img_path.name)
                print('ERRO AO ABRIR IMAGEM')
                print(e)
                print()
                continue

            inp = transform(img).unsqueeze(0).to(device)
            outputs = model(inp)
            probs = softmax(outputs).cpu().numpy()[0]
            pred = int(probs.argmax())

            prob_closed = float(probs[0])
            prob_open = float(probs[1])

            # Saída simples conforme pedido
            print(img_path.name)
            print(label_map[pred])
            print(f"Probabilidade de ser Olhos fechados: {prob_closed:.4f}")
            print(f"Probabilidade de ser Olhos abertos: {prob_open:.4f}")
            print()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EyeClassifierCNN(num_classes=2).to(device)

    model_path = Path('models/eyes_model_best.pth')
    if not model_path.exists():
        print(f"Arquivo de pesos não encontrado: {model_path}. Coloque seu .pth em models/.")
        return

    try:
        state = torch.load(str(model_path), map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        print('Erro ao carregar pesos do modelo:')
        print(e)
        return

    classify_folder(model, device, folder='data/Test_images')


if __name__ == '__main__':
    main()
