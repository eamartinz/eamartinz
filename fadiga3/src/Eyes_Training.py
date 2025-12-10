# ARQUIVO 01
# Inicialmente, preciso organizar um dataset de imagens para treinamento de uma
# CNN para identificar olhos abertos e fechados
#
# Os arquivos são imagens e estão rotulados e organizados da seguinte forma:
# data
# data/eyes
# data/eyes/Closed_Eyes
# data/eyes/Open_Eyes

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path

# Verificar se a GPU está presente
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Configurações
DATA_PATH = "data/eyes"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

# Diretório para salvar modelos e gráficos — aponta para a pasta `models/` na raiz do projeto
# Usa Path(__file__).parents[1] para ir até a raiz (src/ -> projeto)
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Transformações de imagem
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Carregar dataset
print("Carregando dataset...")
full_dataset = ImageFolder(root=DATA_PATH, transform=train_transforms)

# Dividir o dataset em treino, validação e teste
train_size = int(TRAIN_SIZE * len(full_dataset))
val_size = int(VAL_SIZE * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size]
)

# Aplicar transformações diferentes aos conjuntos de validação e teste
val_dataset.dataset.transform = val_test_transforms
test_dataset.dataset.transform = val_test_transforms

# Criar DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Tamanho do treino: {len(train_dataset)}")
print(f"Tamanho da validação: {len(val_dataset)}")
print(f"Tamanho do teste: {len(test_dataset)}")

# Função para treinar a CNN
def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Treina o modelo por uma época.
    
    Args:
        model: Modelo neural
        train_loader: DataLoader de treino
        criterion: Função de perda
        optimizer: Otimizador
        device: Dispositivo (CPU ou GPU)
    
    Returns:
        Perda média da época
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Validação direta
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

# Função para validar a CNN
def validate(model, val_loader, criterion, device):
    """
    Valida o modelo no conjunto de validação.
    
    Args:
        model: Modelo neural
        val_loader: DataLoader de validação
        criterion: Função de perda
        device: Dispositivo (CPU ou GPU)
    
    Returns:
        Tupla (perda média, acurácia)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


# ========================================================================
# ARQUITETURA DA CNN PARA CLASSIFICAÇÃO DE OLHOS (Abertos vs Fechados)
# ========================================================================

class EyeClassifierCNN(nn.Module):
    """
    Rede Neural Convolucional para classificar se os olhos estão abertos ou fechados.
    
    ARQUITETURA:
    - Entrada: Imagens RGB 64x64 pixels
    - Saída: 2 classes (Olhos Abertos, Olhos Fechados)
    
    COMPONENTES:
    1. BLOCO CONVOLUCIONAL 1:
       - Conv2d(3, 32, 3): 3 canais de entrada (RGB) → 32 filtros de 3x3
       - ReLU: Função de ativação não-linear
       - MaxPool2d(2, 2): Reduz dimensão em 50% (preserva features importantes)
       - Resultado: 32 mapas de features de 31x31
    
    2. BLOCO CONVOLUCIONAL 2:
       - Conv2d(32, 64, 3): 32 canais → 64 filtros de 3x3
       - ReLU: Ativação não-linear
       - MaxPool2d(2, 2): Reduz dimensão novamente
       - Resultado: 64 mapas de features de 15x15
    
    3. BLOCO CONVOLUCIONAL 3:
       - Conv2d(64, 128, 3): 64 canais → 128 filtros de 3x3
       - ReLU: Ativação não-linear
       - MaxPool2d(2, 2): Reduz dimensão para 7x7
       - Resultado: 128 mapas de features de 7x7
    
    4. CAMADA FULLY CONNECTED (Classificador):
       - Flatten: Converte 128x7x7 = 6272 neurônios
       - FC1: 6272 → 256 neurônios (camada escondida)
       - ReLU + Dropout(0.5): Reduz overfitting
       - FC2: 256 → 2 neurônios (saída - uma para cada classe)
    
    VANTAGENS DESTA ARQUITETURA:
    - Redução progressiva de dimensão: Cria representações cada vez mais abstratas
    - Múltiplos filtros: Detecta padrões em diferentes escalas
    - Pooling: Torna a rede mais robusta a pequenas variações
    - Dropout: Previne overfitting ao treino
    - Tamanho moderado: Evita lentidão sem sacrificar performance
    """
    
    def __init__(self, num_classes=2):
        """
        Inicializa as camadas da CNN.
        
        Args:
            num_classes (int): Número de classes (padrão: 2 para olhos abertos/fechados)
        """
        super(EyeClassifierCNN, self).__init__()
        
        # ========================
        # BLOCO CONVOLUCIONAL 1
        # ========================
        # Entrada: 3 canais (RGB) x 64x64 pixels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # Kernel de 3x3 com padding=1 mantém as dimensões (64x64)
        # 32 filtros aprendem diferentes features da imagem
        
        self.bn1 = nn.BatchNorm2d(32)  # Normalização por batch melhora convergência
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Reduz de 64x64 para 32x32 (pega o valor máximo de janelas 2x2)
        # Mantém features mais importantes e reduz computação
        
        # ========================
        # BLOCO CONVOLUCIONAL 2
        # ========================
        # Entrada: 32 canais de 32x32 pixels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Aumenta número de filtros para detectar features mais complexas
        # 64 filtros capturam padrões em múltiplas escalas
        
        self.bn2 = nn.BatchNorm2d(64)  # Normalização para estabilizar treinamento
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Reduz de 32x32 para 16x16
        
        # ========================
        # BLOCO CONVOLUCIONAL 3
        # ========================
        # Entrada: 64 canais de 16x16 pixels
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # 128 filtros para capturar features cada vez mais abstratas
        # A rede aprende padrões de alto nível (formas, texturas, etc)
        
        self.bn3 = nn.BatchNorm2d(128)  # Normalização para melhora de performance
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Reduz de 16x16 para 8x8
        # Resultado final: 128 mapas de features de 8x8 = 8192 features
        
        # ========================
        # CAMADA DE DROPOUT
        # ========================
        self.dropout = nn.Dropout(p=0.5)
        # Desativa aleatoriamente 50% dos neurônios durante treinamento
        # Previne co-adaptação e reduz overfitting
        # Não é aplicado durante inferência
        
        # ========================
        # CAMADAS FULLY CONNECTED (Classificador)
        # ========================
        # Flatten: converte 128x8x8 = 8192 features em um vetor
        
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        # Camada escondida: 8192 → 256 neurônios
        # Aprende combinações não-lineares das features
        
        self.fc2 = nn.Linear(256, num_classes)
        # Camada de saída: 256 → 2 neurônios (um por classe)
        # Produz scores para cada classe (olhos abertos vs fechados)
    
    def forward(self, x):
        """
        Passa a entrada pela rede neural.
        
        Args:
            x: Tensor de imagens (batch_size, 3, 64, 64)
        
        Returns:
            Tensor de scores de classe (batch_size, num_classes)
        """
        # ========================
        # BLOCO CONVOLUCIONAL 1
        # ========================
        x = self.conv1(x)           # (batch, 32, 64, 64)
        x = self.bn1(x)             # Normaliza os valores
        x = F.relu(x)               # Ativação ReLU (não-linearidade)
        x = self.pool1(x)           # (batch, 32, 32, 32)
        
        # ========================
        # BLOCO CONVOLUCIONAL 2
        # ========================
        x = self.conv2(x)           # (batch, 64, 32, 32)
        x = self.bn2(x)             # Normaliza
        x = F.relu(x)               # Ativação ReLU
        x = self.pool2(x)           # (batch, 64, 16, 16)
        
        # ========================
        # BLOCO CONVOLUCIONAL 3
        # ========================
        x = self.conv3(x)           # (batch, 128, 16, 16)
        x = self.bn3(x)             # Normaliza
        x = F.relu(x)               # Ativação ReLU
        x = self.pool3(x)           # (batch, 128, 8, 8)
        
        # ========================
        # FLATTEN + FULLY CONNECTED
        # ========================
        x = x.view(x.size(0), -1)   # Flatten: (batch, 128*8*8) = (batch, 8192)
        x = self.dropout(x)         # Dropout durante treinamento
        
        x = self.fc1(x)             # (batch, 256)
        x = F.relu(x)               # Ativação ReLU
        x = self.dropout(x)         # Dropout
        
        x = self.fc2(x)             # (batch, 2) - scores finais
        
        return x


# ========================================================================
# CRIAR E INICIALIZAR O MODELO
# ========================================================================

# Instancia a CNN
model = EyeClassifierCNN(num_classes=2)
model = model.to(device)

print("Modelo criado com sucesso!")
print(f"Total de parâmetros: {sum(p.numel() for p in model.parameters()):,}")
print(f"Parâmetros treináveis: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ========================================================================
# CONFIGURAR FUNÇÃO DE PERDA E OTIMIZADOR
# ========================================================================

# CrossEntropyLoss: combina LogSoftmax + NLLLoss
# Apropriado para classificação multi-classe
criterion = nn.CrossEntropyLoss()

# Adam: otimizador adaptativo
# learning_rate=0.001: velocidade de aprendizado
# Adapta dinamicamente a taxa de aprendizado para cada parâmetro
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ========================================================================
# TREINAR O MODELO
# ========================================================================

# Listas para armazenar métricas
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

print("\n" + "="*70)
print("INICIANDO TREINAMENTO DA CNN")
print("="*70 + "\n")

# Loop de treinamento (sem early stopping)
best_val_acc = 0

for epoch in range(NUM_EPOCHS):
    # Treinar por uma época
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validar após cada época
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Armazenar métricas
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    # Exibir progresso
    print(f"Época {epoch+1}/{NUM_EPOCHS}")
    print(f"  Treino - Loss: {train_loss:.4f}, Acurácia: {train_acc:.2f}%")
    print(f"  Val    - Loss: {val_loss:.4f}, Acurácia: {val_acc:.2f}%")
    
    # Salvar melhor modelo (mantém salvamento sem interromper o treino)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), str(MODELS_DIR / 'eyes_model_best.pth'))
        print(f"  ✓ Melhor modelo salvo (Acurácia: {val_acc:.2f}%)")
    
    print()

print("="*70)
print("TREINAMENTO CONCLUÍDO")
print("="*70 + "\n")

# ========================================================================
# AVALIAR NO CONJUNTO DE TESTE
# ========================================================================

print("Avaliando no conjunto de teste...")
test_loss, test_acc = validate(model, test_loader, criterion, device)
print(f"Teste - Loss: {test_loss:.4f}, Acurácia: {test_acc:.2f}%\n")

# ========================================================================
# GERAR MATRIZ DE CONFUSÃO
# ========================================================================

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def get_predictions(model, dataloader, device):
    """
    Obtém predições e labels verdadeiros do modelo.
    
    Returns:
        Tupla (predições, labels verdadeiros)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)

# Obter predições para teste
test_predictions, test_labels = get_predictions(model, test_loader, device)

# Calcular matriz de confusão
cm = confusion_matrix(test_labels, test_predictions)

print("Matriz de Confusão:")
print(cm)
print()

# Gerar relatório de classificação
print("Relatório de Classificação:")
class_names = ['Olhos Fechados', 'Olhos Abertos']
print(classification_report(test_labels, test_predictions, target_names=class_names))

# ========================================================================
# GERAR GRÁFICOS
# ========================================================================

import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. GRÁFICO DE LOSS (Treino vs Validação)
print("Gerando gráfico de Loss...")
fig, ax = plt.subplots(figsize=(10, 6))

epochs_range = range(1, len(train_losses) + 1)
ax.plot(epochs_range, train_losses, 'o-', label='Treino', linewidth=2, markersize=4)
ax.plot(epochs_range, val_losses, 's-', label='Validação', linewidth=2, markersize=4)

ax.set_xlabel('Época', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss (CrossEntropyLoss)', fontsize=12, fontweight='bold')
ax.set_title('Loss durante Treino e Validação', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(MODELS_DIR / 'eyes_loss_plot.png'), dpi=300, bbox_inches='tight')
print(f"✓ Gráfico de Loss salvo: {MODELS_DIR / 'eyes_loss_plot.png'}")
plt.close()

# 2. GRÁFICO DE ACURÁCIA (Treino vs Validação)
print("Gerando gráfico de Acurácia...")
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(epochs_range, train_accuracies, 'o-', label='Treino', linewidth=2, markersize=4, color='#2ecc71')
ax.plot(epochs_range, val_accuracies, 's-', label='Validação', linewidth=2, markersize=4, color='#e74c3c')

ax.set_xlabel('Época', fontsize=12, fontweight='bold')
ax.set_ylabel('Acurácia (%)', fontsize=12, fontweight='bold')
ax.set_title('Acurácia durante Treino e Validação', fontsize=14, fontweight='bold')
ax.set_ylim([0, 105])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(MODELS_DIR / 'eyes_accuracy_plot.png'), dpi=300, bbox_inches='tight')
print(f"✓ Gráfico de Acurácia salvo: {MODELS_DIR / 'eyes_accuracy_plot.png'}")
plt.close()

# 3. MATRIZ DE CONFUSÃO
print("Gerando matriz de confusão...")
fig, ax = plt.subplots(figsize=(8, 6))

# Plotar matriz de confusão com anotações
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Número de Amostras'},
            ax=ax)

ax.set_xlabel('Predito', fontsize=12, fontweight='bold')
ax.set_ylabel('Verdadeiro', fontsize=12, fontweight='bold')
ax.set_title('Matriz de Confusão - Conjunto de Teste', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(str(MODELS_DIR / 'eyes_confusion_matrix.png'), dpi=300, bbox_inches='tight')
print(f"✓ Matriz de Confusão salvo: {MODELS_DIR / 'eyes_confusion_matrix.png'}")
plt.close()

# 4. GRÁFICO COMBINADO (Loss + Acurácia)
print("Gerando gráfico combinado...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss
axes[0].plot(epochs_range, train_losses, 'o-', label='Treino', linewidth=2)
axes[0].plot(epochs_range, val_losses, 's-', label='Validação', linewidth=2)
axes[0].set_xlabel('Época', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Loss', fontsize=11, fontweight='bold')
axes[0].set_title('Loss - Treino vs Validação', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Acurácia
axes[1].plot(epochs_range, train_accuracies, 'o-', label='Treino', linewidth=2, color='#2ecc71')
axes[1].plot(epochs_range, val_accuracies, 's-', label='Validação', linewidth=2, color='#e74c3c')
axes[1].set_xlabel('Época', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Acurácia (%)', fontsize=11, fontweight='bold')
axes[1].set_title('Acurácia - Treino vs Validação', fontsize=12, fontweight='bold')
axes[1].set_ylim([0, 105])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(MODELS_DIR / 'eyes_training_metrics_combined.png'), dpi=300, bbox_inches='tight')
print(f"✓ Gráfico combinado salvo: {MODELS_DIR / 'eyes_training_metrics_combined.png'}")
plt.close()

# ========================================================================
# SALVAR O MODELO TREINADO
# ========================================================================

print("\nSalvando modelo treinado...")

# Salvar apenas os pesos (mais leve)
torch.save(model.state_dict(), str(MODELS_DIR / 'eyes_model_final.pth'))
print(f"✓ Pesos do modelo salvos: {MODELS_DIR / 'eyes_model_final.pth'}")

# Salvar modelo completo (inclui arquitetura)
torch.save(model, str(MODELS_DIR / 'eyes_model_complete.pth'))
print(f"✓ Modelo completo salvo: {MODELS_DIR / 'eyes_model_complete.pth'}")

# Salvar checkpoint com otimizador e épocas
checkpoint = {
    'epoch': len(train_losses),
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_losses[-1],
    'val_loss': val_losses[-1],
    'train_accuracy': train_accuracies[-1],
    'val_accuracy': val_accuracies[-1],
    'test_accuracy': test_acc,
    'best_val_accuracy': best_val_acc
}

torch.save(checkpoint, str(MODELS_DIR / 'eyes_model_checkpoint.pth'))
print(f"✓ Checkpoint salvo: {MODELS_DIR / 'eyes_model_checkpoint.pth'}")

# ========================================================================
# RESUMO FINAL
# ========================================================================

print("\n" + "="*70)
print("RESUMO DO TREINAMENTO")
print("="*70)
print(f"Total de épocas: {len(train_losses)}")
print(f"Melhor acurácia de validação: {best_val_acc:.2f}%")
print(f"Acurácia no teste: {test_acc:.2f}%")
print(f"Loss final no treino: {train_losses[-1]:.4f}")
print(f"Loss final na validação: {val_losses[-1]:.4f}")
print("\nArquivos salvos em 'models/':")
print("  - eyes_loss_plot.png: Gráfico de Loss (Treino vs Validação)")
print("  - eyes_accuracy_plot.png: Gráfico de Acurácia (Treino vs Validação)")
print("  - eyes_confusion_matrix.png: Matriz de Confusão do Teste")
print("  - eyes_training_metrics_combined.png: Gráficos combinados")
print("  - eyes_model_final.pth: Pesos do modelo treinado")
print("  - eyes_model_complete.pth: Modelo completo")
print("  - eyes_model_checkpoint.pth: Checkpoint com metadados")
print("="*70 + "\n")