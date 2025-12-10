# ARQUIVO 3
# src/treinar.py

# ================================================
# Script para treinar a CNN
# com 3 classes: open, closed e yawn.
# 
# Este script:
# - Carrega o dataset
# - Usa GPU automaticamente se disponível
# - Treina o modelo CNN
# - Salva métricas de loss/accuracy
# - Gera 2 gráficos:
# - - Loss treino vs validação
# - - Accuracy treino vs validação
# - Salva o modelo final em model.pth
# ================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model1 import FadigaCNN    # Importa a FadigaCNN


# ==========================================================
# 1. Hiperparâmetros
# ==========================================================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20     # comecei com 10 para ter resultados rápidos
IMAGE_SIZE = 64     # altura e largura da imagem

DATASET_DIR = "data"  # pasta principal do da3 taset
MODEL_SAVE_PATH = "models/model.pth"  # arquivo do modelo salvo

# ==========================================================
# 2. Configuração para usar GPU automaticamente
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ==========================================================
# 3. Transformações nas imagens
#    Padronizam tamanho, tensor e normalização
# ==========================================================
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==========================================================
# 4. Carregar o dataset
# ==========================================================
train_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "Train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "Val"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "Test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

CLASSES = train_dataset.classes
print("Classes detectadas:", CLASSES)

# ==========================================================
# 5. Inicializando modelo, função de perda e otimizador
# ==========================================================
# model = FadigaCNN(num_classes=len(CLASSES)).to(device)
model = FadigaCNN(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()   # função de perda
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Para armazenar as métricas
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# ==========================================================
# 6. Função para calcular acurácia
# ==========================================================
def calcular_acuracia(model, loader):
    model.eval()
    hits = 0
    total = 0

    with torch.no_grad():   # retira cálculo do gradiente
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            hits += (preds == labels).sum().item()
            total += labels.size(0)

    return hits / total

# ==========================================================
# 7. Laço principal de treinamento
# ==========================================================
print("\n===== INICIANDO TREINAMENTO =====")

for epoch in range(NUM_EPOCHS):
    print(f"\n===== ÉPOCA {epoch+1}/{NUM_EPOCHS} =====")

    # -------------------------
    # TREINO
    # -------------------------
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Estatísticas
        running_loss += loss.item() * imgs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # -------------------------
    # VALIDAÇÃO
    # -------------------------
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(val_loader.dataset)
    val_acc = val_correct / val_total

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Treino    — Loss: {train_loss:.4f} | Acurácia: {train_acc*100:.2f}%")
    print(f"Validação — Loss: {val_loss:.4f}   | Acurácia: {val_acc*100:.2f}%")
    
# ==========================================================
# 8. Salvar o modelo
# ==========================================================
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("\nModelo salvo como:", MODEL_SAVE_PATH)

# ==========================================================
# 9. Gráficos das métricas
# ==========================================================
# ============================
# GRÁFICO 1 — LOSS
# ============================
plt.figure(figsize=(10,5))
plt.plot(train_losses, label="Treino")
plt.plot(val_losses, label="Validação")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Loss por Época")
plt.legend()
plt.grid(True)
plt.savefig("models/loss.png")
plt.show()

# ============================
# GRÁFICO 2 — ACURÁCIA
# ============================
plt.figure(figsize=(10,5))
plt.plot(train_accuracies, label="Treino")
plt.plot(val_accuracies, label="Validação")
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.title("Acurácia por Época")
plt.legend()
plt.grid(True)
plt.savefig("models/accuracy.png")
plt.show()

print("\nGráficos salvos como 'loss.png' e 'accuracy.png'.")
print("===== TREINAMENTO FINALIZADO =====")
