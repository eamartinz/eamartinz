# Arquivo 5
# src/webcam_inferencia.py

import cv2
import torch
import torchvision.transforms as transforms
import time

from model1 import FadigaCNN  # importa o modelo

# ------------------------------------------------------------
# CONFIGURAÇÕES DO SCRIPT
# ------------------------------------------------------------
MODEL_PATH = "models/model.pth"  # caminho para o modelo treinado

# As classes do dataset na ordem correta
CLASSES = ["Closed_Eyes", "No_yawn", "Open_Eyes", "Yawn"]

# Seleciona o device GPU se estiver disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ------------------------------------------------------------
# CARREGAR O MODELO TREINADO
# ------------------------------------------------------------
model = FadigaCNN(num_classes=len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()  # modo de inferência
print("Modelo carregado com sucesso!")

# ------------------------------------------------------------
# TRANSFORMAÇÕES USADAS NO TREINAMENTO
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# ------------------------------------------------------------
# FUNÇÃO PARA CLASSIFICAR UM FRAME
# ------------------------------------------------------------
def classificar_frame(frame):
    # Converte de BGR (OpenCV) para RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Converter para GRAYSCALE (1 canal)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Transformações do treinamento
    img = transform(img).unsqueeze(0).to(device)

    # Inferência
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    
    classe = CLASSES[predicted.item()]
    return classe

# ------------------------------------------------------------
# INICIAR A WEBCAM
# ------------------------------------------------------------
cap = cv2.VideoCapture(0)  # webcam padrão

if not cap.isOpened():
    print("Erro ao acessar a webcam.")
    exit()

print("Pressione Q para sair.")

# ------------------------------------------------------------
# LOOP PRINCIPAL DA WEBCAM
# ------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame.")
        break

    # Classifica o frame atual
    classe = classificar_frame(frame)

    # Desenha texto na imagem
    cv2.putText(frame, f"Classe: {classe}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Mostra o frame
    cv2.imshow("Deteccao de Fadiga - Webcam", frame)

    # Se pressionar Q → fecha
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------------------------------------
# ENCERRAMENTO
# ------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
print("Encerrado.")
