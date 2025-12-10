#!/usr/bin/env python3
"""
Eyes_Validation.py

Script para validar imagens estáticas utilizando um modelo treinado de detecção
de olhos (abertos / fechados). Configure os caminhos e a lista de imagens na
seção CONFIGURAÇÕES DO USUÁRIO.

Saída: para cada imagem será impresso algo como:

-------------------------------------------
Imagem testada: _27.jpg
Classe detectada: Olhos abertos
Confiança: 0.9543
Probabilidades:
  Olhos fechados: 0.0457
  Olhos abertos: 0.9543

"""

import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Tenta importar a classe de modelo local (se existir)
try:
    from model_eyes import EyesCNN
except Exception:
    EyesCNN = None

# ======================================================
# CONFIGURAÇÕES DO USUÁRIO
# ======================================================
MODEL_PATH = "models/eyes_model.pth"     # Modelo treinado (state_dict salvo por train_eyes.py)
TEST_DIR = "data/Test_images/"           # Diretório das imagens
TEST_IMAGES = [
    "_27.jpg",
    "_33.jpg",
    "_34.jpg",
    "_35.jpg"
]

CLASSES = ["Olhos fechados", "Olhos abertos"]
IMAGE_SIZE = 64   # deve ser igual ao usado no treino
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# Função para carregar imagem
# ======================================================
def load_image(image_path):
    """
    Carrega e transforma uma imagem para inferência.
    Está configurada para imagens em escala de cinza (L) como no exemplo.
    """
    # OBS: o dataset usado em `train_eyes.py` aplica: Resize -> Grayscale -> ToTensor
    # Portanto aqui NÃO aplicamos normalização para manter consistência com o treino.
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("L")  # converte para escala de cinza
    return transform(img).unsqueeze(0)  # adiciona dimensão de batch


# ======================================================
# Função de predição
# ======================================================
def predict_image(model, image_path):
    model.eval()
    image = load_image(image_path).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_idx = int(torch.argmax(outputs, dim=1).item())

    return predicted_idx, probs


# ======================================================
# Função para carregar o modelo (state_dict ou modelo completo)
# ======================================================
def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"[ERRO] Modelo não encontrado em: {model_path}")
        sys.exit(1)

    # Tentar carregar como state_dict usando EyesCNN (se disponível)
    # O script de treino (`train_eyes.py`) salva apenas o state_dict em
    # `models/eyes_model.pth`. Aqui assumimos esse formato e carregamos
    # exclusivamente via EyesCNN + load_state_dict.
    if EyesCNN is None:
        print("[ERRO] Classe EyesCNN não disponível. Certifique-se de que `model_eyes.py` existe e exporta `EyesCNN`.")
        sys.exit(1)

    try:
        model = EyesCNN(num_classes=2).to(DEVICE)
        state = torch.load(model_path, map_location=DEVICE)
        # Caso o usuário tenha salvo um checkpoint com chaves, extrair model_state_dict
        if isinstance(state, dict) and 'model_state_dict' in state:
            state_dict = state['model_state_dict']
        else:
            state_dict = state

        if not isinstance(state_dict, dict):
            raise ValueError("O arquivo de modelo não contém um state_dict válido.")

        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"[ERRO] Falha ao carregar state_dict para EyesCNN: {e}")
        print("Certifique-se de que o modelo foi salvo com `torch.save(model.state_dict(), PATH)` em `train_eyes.py`.")
        sys.exit(1)


def main():
    # Carregar o modelo
    model = load_model(MODEL_PATH)

    print("\n========== VALIDAÇÃO DE IMAGENS ESTÁTICAS ==========")
    print(f"Usando modelo: {MODEL_PATH}")
    print(f"Device: {DEVICE}\n")

    for img_name in TEST_IMAGES:
        img_path = os.path.join(TEST_DIR, img_name)

        if not os.path.exists(img_path):
            print(f"\n[ERRO] Imagem não encontrada: {img_path}")
            continue

        predicted_idx, proba = predict_image(model, img_path)

        print("\n-------------------------------------------")
        print(f"Imagem testada: {img_name}")
        print(f"Classe detectada: {CLASSES[predicted_idx]}")
        print(f"Confiança: {proba[predicted_idx]:.4f}")
        print("Probabilidades:")
        for i, p in enumerate(proba):
            print(f"  {CLASSES[i]}: {p:.4f}")

    print("\n===========================================\n")


if __name__ == "__main__":
    main()
