# ARQUIVO 03
# src/train_eyes.py

# train_eyes.py
# Treina a CNN para classificar olhos: abertos vs fechados

#  Treina o modelo CNN para classificação de olhos:
#     - Open_Eyes
#     - Closed_Eyes
#
#     - Constantes configuráveis
#     - Registro de loss e accuracy
#     - Separação treino/validação com DataLoader adequado
#     - Gráficos automáticos de aprendizado
# ===============================================================

# ===============================================================
#  train_eyes.py (versão com matriz de confusão)
# ===============================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import os

from dataset_eyes import EyesDataset
from model_eyes import EyesCNN


# ===============================================================
#  CONSTANTES — ALTERE AQUI
# ===============================================================

DATASET_PATH = "data/eyes"

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
VAL_SPLIT = 0.2

USE_GPU = True
DEVICE = "cuda" if (torch.cuda.is_available() and USE_GPU) else "cpu"



# ===============================================================
#  FUNÇÃO PRINCIPAL
# ===============================================================
def main():

    print(f"\nUsando dispositivo: {DEVICE}")
    print("Carregando dataset...\n")

    dataset = EyesDataset(DATASET_PATH)

    # -----------------------------------------------------------
    # SEPARAR TREINO/VALIDAÇÃO
    # -----------------------------------------------------------
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=VAL_SPLIT, shuffle=True)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_idx))
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_idx))

    # -----------------------------------------------------------
    # MODELO + LOSS + OTIMIZADOR
    # -----------------------------------------------------------
    model = EyesCNN(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    val_accuracies = []

    print("Iniciando treinamento...\n")

    # -----------------------------------------------------------
    # LOOP DE TREINAMENTO
    # -----------------------------------------------------------
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

        # -------------------------------------------------------
        # AVALIAÇÃO
        # -------------------------------------------------------
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

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
        val_accuracies.append(val_accuracy)

        print(f"[Época {epoch+1}/{EPOCHS}]  "
              f"Treino Loss: {running_loss:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  |  "
              f"Val Acc: {val_accuracy*100:.2f}%")

    # -----------------------------------------------------------
    # SALVAR MODELO
    # -----------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/eyes_model.pth")
    print("\nModelo salvo em models/eyes_model.pth")

    # -----------------------------------------------------------
    # GRÁFICOS
    # -----------------------------------------------------------
    gerar_graficos(train_losses, val_losses, val_accuracies)

    # -----------------------------------------------------------
    # MATRIZ DE CONFUSÃO
    # -----------------------------------------------------------
    gerar_matriz_confusao(all_labels, all_preds)


# ===============================================================
#  GRÁFICOS
# ===============================================================
def gerar_graficos(train_loss, val_loss, val_acc):

    epochs_range = np.arange(1, len(train_loss) + 1)

    # -------- LOSS --------
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_range, train_loss, label="Treino", marker='o')
    plt.plot(epochs_range, val_loss, label="Validação", marker='o')
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Loss por Época")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("models/eyes_loss.png")
    plt.show()

    # -------- ACC --------
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_range, val_acc, label="Validação", marker='o')
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.title("Acurácia de Validação")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("models/eyes_accuracy.png")
    plt.show()


# ===============================================================
#  MATRIZ DE CONFUSÃO
# ===============================================================
def gerar_matriz_confusao(labels, preds):

    cm = confusion_matrix(labels, preds)
    classes = ["Closed_Eyes", "Open_Eyes"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predição")
    plt.ylabel("Valor Real")
    plt.title("Matriz de Confusão — Classificador de Olhos")
    plt.tight_layout()
    plt.savefig("models/eyes_confusion_matrix.png")
    plt.show()


# ===============================================================
#  EXECUÇÃO
# ===============================================================
if __name__ == "__main__":
    main()
