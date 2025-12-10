# SCRIPT DE TREINO DO MODELO
# src/train.py

"""
Script para treinar a rede neural convolucional para classificação de olhos
abertos e fechados usando PyTorch.
"""

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from dataset import EyeDataset
from model import EyeCNN
import os

def train():
    # Carrega o dataset
    dataset = EyeDataset("data/cew")
    n = len(dataset)
    # Divide em treino e validação (80% treino, 20% validação)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    model = EyeCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / n_train
        epoch_acc = correct / total
        print(f"[TRAIN] Epoch {epoch+1}/{epochs} -- Loss: {epoch_loss:.4f} -- Acc: {epoch_acc:.4f}")

        # Validação
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = val_correct / val_total
        print(f"[VAL]   Epoch {epoch+1}/{epochs} -- val_Acc: {val_acc:.4f}")

    # Salva o modelo treinado
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/eye_cnn.pth")
    print("Treinamento concluído e modelo salvo em models/eye_cnn.pth")

if __name__ == "__main__":
    train()