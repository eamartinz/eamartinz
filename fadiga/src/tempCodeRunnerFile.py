# ARQUIVO 4
# src/evaluate.py

# ===============================================================
#  Objetivo:
#     - Carregar o modelo treinado (SimpleCNN)
#     - Carregar uma imagem de teste
#     - Aplicar as mesmas transformações usadas no dataset
#     - Executar a inferência e mostrar o resultado
# ===============================================================

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

# Importa o modelo
from model1 import FadigaCNN

# ---------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------

# Caminho para o modelo treinado (.pth)
MODEL_PATH = "models/model.pth"

# Caminho da imagem de teste (troque este arquivo para testar outros)
TEST_IMAGE = "data/evaluate/teste4.jpg"

# Lista de classes possíveis
CLASSES = ["Closed_Eyes", "No_yawn", "Open_Eyes", "Yawn"]

# Mapeamos se o modelo deve classificar como ATENÇÃO ou FADIGA
# Aqui fazemos uma simplificação:
#   - Closed_Eyes → Fadiga
#   - Yawn        → Fadiga
#   - Open_Eyes   → Atenção
#   - No_yawn     → Atenção
MAPEAMENTO_FADIGA = {
    "Closed_Eyes": "Fadiga",
    "Yawn": "Fadiga",
    "Open_Eyes": "Atenção",
    "No_yawn": "Atenção"
}

# ---------------------------------------------------------------
# VERIFICAR DISPONIBILIDADE DE GPU
# ---------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


# ---------------------------------------------------------------
# TRANSFORMAÇÃO IGUAL À DO DATASET
# ---------------------------------------------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# ---------------------------------------------------------------
# FUNÇÃO PARA CARREGAR A IMAGEM
# ---------------------------------------------------------------
def carregar_imagem(caminho_img):
    """
    Abre a imagem, aplica transformações e retorna um tensor [1, 1, 64, 64].
    """
    img = Image.open(caminho_img)

    img_t = transform(img)
    img_t = img_t.unsqueeze(0)  # adiciona dimensão batch -> [1, 1, 64, 64]

    return img, img_t


# ---------------------------------------------------------------
# FUNÇÃO PRINCIPAL
# ---------------------------------------------------------------
def main():

    if not os.path.exists(MODEL_PATH):
        print(f"ERRO: Modelo não encontrado em: {MODEL_PATH}")
        sys.exit(1)

    if not os.path.exists(TEST_IMAGE):
        print(f"ERRO: Imagem para teste não encontrada: {TEST_IMAGE}")
        sys.exit(1)

    # 1) Carregar modelo
    print("Carregando modelo...")
    # model = FadigaCNN(num_classes=len(CLASSES))
    model = FadigaCNN(num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 2) Carregar imagem
    img_original, img_tensor = carregar_imagem(TEST_IMAGE)
    img_tensor = img_tensor.to(device)

    # 3) Inferência
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_index = torch.argmax(probabilities).item()

    predicted_class = CLASSES[predicted_index]
    estado = MAPEAMENTO_FADIGA[predicted_class]

    # 4) Imprimir resultados numéricos
    print("\n========== RESULTADO ==========")
    print(f"Classe detectada: {predicted_class}")
    print(f"Estado: {estado}")
    print("Probabilidades:")
    for i, c in enumerate(CLASSES):
        print(f"  {c}: {probabilities[0][i]:.4f}")

    # 5) Mostrar imagem com label
    plt.imshow(img_original)
    plt.title(f"Predição: {predicted_class} ({estado})")
    plt.axis("off")
    plt.show()


# ---------------------------------------------------------------
# EXECUÇÃO DIRETA
# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
