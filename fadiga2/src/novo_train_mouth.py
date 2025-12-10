# ARQUIVO 07
# src/novo_train_mouth.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from dataset_mouth import MouthDataset
from model_mouth import MouthCNN


# ========================
# CONFIGURAÇÕES
# ========================
DATASET_PATH = "data/mouth"

BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 20
VAL_SPLIT = 0.2

USE_GPU = True
DEVICE = "cuda" if (torch.cuda.is_available() and USE_GPU) else "cpu"



# ========================
# EARLY STOPPING OPCIONAL
# ========================
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = np.inf
        self.counter = 0

    def check(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience



# ========================
# TREINAMENTO
# ========================
def main():
    print("DEBUG: script entrou no main()")

    print(f"\nUsando dispositivo: {DEVICE}")
    print("Carregando dataset...\n")

    dataset = MouthDataset(DATASET_PATH)

    # Separação treino / validação
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=VAL_SPLIT, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_idx))
    val_loader   = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_idx))

    # Modelo + loss + otimizador
    model = MouthCNN(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses, val_accs = [], [], []

    early = EarlyStopping(patience=5)

    print("Iniciando treinamento...\n")

    # --------------------------
    # LOOP DE TREINAMENTO
    # --------------------------
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # --------------------------
        # VALIDAÇÃO
        # --------------------------
        model.eval()
        val_loss = 0
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_accuracy = correct / total

        train_losses.append(running_loss)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        print(f"[Época {epoch+1}/{EPOCHS}]  "
              f"Treino Loss: {running_loss:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  |  "
              f"Val Acc: {val_accuracy*100:.2f}%")

        # Early stopping
        if early.check(val_loss):
            print("\nEarly stopping ativado!")
            break

    # Salvar modelo
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mouth_model.pth")
    print("\nModelo salvo em models/mouth_model.pth")

    gerar_graficos(train_losses, val_losses, val_accs)
    gerar_matriz_confusao(all_labels, all_preds)
