# ARQUIVO 08
# src/evaluate_mouth.py

# evaluate_mouth.py
# Testa o modelo de detecção de Yawm / No_Yawn

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from model_mouth import MouthCNN  # modelo adaptado para boca

# ======================================================
# CONFIGURAÇÕES DO USUÁRIO (EDITE APENAS AQUI)
# ======================================================
MODEL_PATH = "models/mouth_model.pth"      # Modelo treinado
TEST_DIR = "data/Test_images/"       # Diretório das imagens
TEST_IMAGES = [
    "teste1.jpg",
    "teste2.jpg",
    "teste3.jpg",
    "teste4.jpg",
    "teste5.jpg",
    "teste6.jpg",
    "teste7.jpg"
]

CLASSES = ["No_Yawn", "Yawn"]
IMAGE_SIZE = 64   # Mesmo usado no treino
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# Função para carregar imagem
# ======================================================
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # igual ao treino
    ])

    img = Image.open(image_path).convert("L")  # escala de cinza
    return transform(img).unsqueeze(0)         # adiciona batch


# ======================================================
# Função de predição
# ======================================================
def predict_image(model, image_path):
    model.eval()
    image = load_image(image_path).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_idx = torch.argmax(outputs, dim=1).item()

    return predicted_idx, probabilities


# ======================================================
# EXECUÇÃO
# ======================================================
if __name__ == "__main__":

    # Carrega o modelo treinado
    model = MouthCNN(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("\n========== TESTE DO MODELO DE BOCA (Yawn / No_Yawn) ==========\n")
    print(f"Usando modelo: {MODEL_PATH}")

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

    print("\n==============================================================\n")
