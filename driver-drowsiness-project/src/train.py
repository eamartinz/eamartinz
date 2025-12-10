# Script de treinamento simples com rede CNN pequena usando PyTorch
# src/train.py

"""
Treinamento da rede CNN pequena com PyTorch para detecção de fadiga com classificação em duas classes.
"""

import os
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# === HIPERPARÂMETROS ===
DATA_DIR = "../data"
BATH_SIZE = 32
IMAGE_SIZE = 128    # tamanho das imagens (128x128)
LR = 0.001         # taxa de aprendizado
EPOCHS = 10        # número de épocas
MODEL_DIR = "../models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")
# =======================

# === Cria diretório para salvar modelos, se não existir
os.makedirs(MODEL_DIR, exist_ok=True)
# =======================

# === Transformações para pré-processamento das imagens
train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === Carrega os datasets de treino e validação
# Exige que os dados estejam organizados em subdiretórios por classe
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_tf)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

num_classes = len(train_ds.classes)
print(f"Número de classes: {num_classes}")
# =======================

# === Define a arquitetura da rede CNN pequena (transfer learning com MobileNetV2)
# Transfer learning
model = models.mobilenet_v2(pretrained=True)
# Ajuse da última camada
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Congelar backbone nas primeiras épocas
for param in model.features.parameters():
    param.requires_grad = True # Definir como False para congelar

model = model.to(DEVICE)
# =======================

# === Define a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
best_val_acc = 0.0
best_model_path = os.path.join(MODEL_DIR, "best_model.pth")

for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    t0 = time.time()

    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    # === Validação
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    val_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data).item()
            val_total += inputs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_epoch_loss = val_running_loss / val_total
    val_epoch_acc = val_running_corrects / val_total
    val_losses.append(val_epoch_loss)
    val_accs.append(val_epoch_acc)

    elapsed = time.time() - t0
    print(f"Epoch {epoch}/{EPOCHS} train_loss: {epoch_loss:.4f} train_acc: {epoch_acc} | val_loss: {val_epoch_loss:.4f} val_acc: {val_epoch_acc:.4f} time:{elapsed:.1f} s")

    # Salvar o melhor modelo
    if val_epoch_acc > best_val_acc:
        best_val_acc = val_epoch_acc
        torch.save({
            'epoch': epoch,
            'model_sate_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': best_val_acc,
            'classes': train_ds.classes
        }, best_model_path)
        print("  -> Melhor modelo salvo em", best_model_path)
# =======================

# === Métricas e matriz de confusão
# Carregar o melhor modelo
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print("\nVal Accuracy (best model):", acc)
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=train_ds.classes))

cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:\n", cm)

# Salvar gráficos de loss/acc e matriz de confusão
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label="train_loss")
plt.plot(val_losses, label="val_loss")
plt.title("Loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(train_accs, label="train_acc")
plt.plot(val_accs, label="val_acc")
plt.title("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"))
print("Gráficos salvos em", MODEL_DIR)

# Matriz de confusão plot simples
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest')
plt.colorbar()
plt.title("Confusion matrix")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.xticks([0,1], train_ds.classes, rotation=45)
plt.yticks([0,1], train_ds.classes)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
print("Matrix de confusão salva em", MODEL_DIR)