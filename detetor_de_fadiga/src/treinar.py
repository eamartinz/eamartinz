import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt

from dataset import criar_dataloaders
from modelo import DetetorDeFadigaCNN

# --------------------------------------------------------------
# Verificar GPU
# --------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando o dispositivo: {device}")

# --------------------------------------------------------------
# Função de acurácia
# --------------------------------------------------------------
def calcular_acuracia(predicoes, labels):
    _, pred = torch.max(predicoes, 1)
    acertos = (pred == labels).sum().item()
    return acertos / labels.size(0)

# --------------------------------------------------------------
# Processo de validação (sem gradiente)
# --------------------------------------------------------------
def validar(modelo, criterio, valid_loader):
    modelo.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for imagens, labels in valid_loader:
            imagens = imagens.to(device)
            labels = labels.to(device)

            saida = modelo(imagens)

            loss = criterio(saida, labels)
            acc = calcular_acuracia(saida, labels)

            total_loss += loss.item()
            total_acc += acc

    modelo.train()
    return total_loss, total_acc / len(valid_loader)

# --------------------------------------------------------------
# Treinamento completo
# --------------------------------------------------------------
def treinar_modelo(epochs=10, batch_size=32, learning_rate=0.001):
    print("Iniciando treinamento do Detetor de Fadiga...")

    # 1) Carregar DataLoaders (treino e validação)
    train_loader, valid_loader = criar_dataloaders(batch_size=batch_size)

    # 2) Criar modelo
    modelo = DetetorDeFadigaCNN().to(device)

    # 3) Função de perda e otimizador
    criterio = nn.CrossEntropyLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=learning_rate)

    # Histórico de métricas
    train_loss_hist = []
    train_acc_hist = []
    valid_loss_hist = []
    valid_acc_hist = []

    # 4) Loop de treinamento
    for epoca in range(epochs):
        total_loss = 0
        total_acc = 0

        for imagens, labels in train_loader:
            imagens = imagens.to(device)
            labels = labels.to(device)

            otimizador.zero_grad()

            saida = modelo(imagens)
            loss = criterio(saida, labels)
            acc = calcular_acuracia(saida, labels)

            loss.backward()
            otimizador.step()

            total_loss += loss.item()
            total_acc += acc

        # Loss e acurácia média da época
        media_loss = total_loss / len(train_loader)
        media_acc = total_acc / len(train_loader)

        # Validação
        valid_loss, valid_acc = validar(modelo, criterio, valid_loader)

        # Guardar métricas
        train_loss_hist.append(media_loss)
        train_acc_hist.append(media_acc)
        valid_loss_hist.append(valid_loss)
        valid_acc_hist.append(valid_acc)

        print(f"Época {epoca+1}/{epochs} | "
              f"Train Loss: {media_loss:.4f} | Train Acc: {media_acc:.4f} | "
              f"Val Loss: {valid_loss:.4f} | Val Acc: {valid_acc:.4f}")

    # Criar pasta models/
    os.makedirs("models", exist_ok=True)

    # ----------------------------------------------------------
    # Gráficos
    # ----------------------------------------------------------
    # LOSS
    plt.figure()
    plt.plot(train_loss_hist, label="Treino")
    plt.plot(valid_loss_hist, label="Validação")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.title("Loss - Treino vs Validação")
    plt.legend()
    plt.grid(True)
    plt.savefig("models/grafico_loss.png")
    plt.close()

    # ACURÁCIA
    plt.figure()
    plt.plot(train_acc_hist, label="Treino")
    plt.plot(valid_acc_hist, label="Validação")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.title("Acurácia - Treino vs Validação")
    plt.legend()
    plt.grid(True)
    plt.savefig("models/grafico_accuracy.png")
    plt.close()

    # ----------------------------------------------------------
    # Salvar modelo
    # ----------------------------------------------------------
    caminho_modelo = "models/modelo.pth"
    torch.save(modelo.state_dict(), caminho_modelo)

    print("\nGráficos salvos em /models/")
    print(f"Modelo salvo em: {caminho_modelo}")


# --------------------------------------------------------------
if __name__ == "__main__":
    treinar_modelo(epochs=10, batch_size=32, learning_rate=0.001)
