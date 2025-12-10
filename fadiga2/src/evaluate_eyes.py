# ARQUIVO 04
# src/evaluate_eyes.py

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from model_eyes import EyesCNN  # ajuste se o nome do modelo for outro

# ======================================================
# CONFIGURAÇÕES DO USUÁRIO
# ======================================================
MODEL_PATH = "models/eyes_model.pth"     # Modelo treinado
TEST_DIR = "data/Test_images/"                # Diretório das imagens
TEST_IMAGES = [
    "_27.jpg",
    "_33.jpg",
    "_34.jpg",
    "_35.jpg"
]

CLASSES = ["Closed_Eyes", "Open_Eyes"]
IMAGE_SIZE = 64   # deve ser igual ao usado no treino (isso me deu problema antes)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# Função para carregar imagem
# ======================================================
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Mesma normalização usada no treino
    ])

    img = Image.open(image_path).convert("L")  # Converte para escala de cinza
    return transform(img).unsqueeze(0)         # Adiciona dimensão batch


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

    # Carrega o modelo
    model = EyesCNN(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("\n========== TESTE DO MODELO DE OLHOS ==========\n")
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
    print("\n===========================================\n")
