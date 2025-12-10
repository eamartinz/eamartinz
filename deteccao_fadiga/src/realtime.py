# Inferência em tempo real pela imagem da webcam
# src/realtime.py
"""
Script para usar a rede treinada e fazer inferência em tempo real a partir da webcam.
"""

import cv2
import torch
import torchvision.transforms as T
from model import EyeCNN

# Transformações para a imagem capturada pela webcam
transform = T.Compose([
    T.ToPILImage(),
    T.Grayscale(),
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EyeCNN().to(device)
model.load_state_dict(torch.load("models/eye_cnn.pth", map_location=device))
model.eval()

# Carrega os classificadores Haar para detecção de rosto e olhos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Inicia a captura de vídeo pela webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao abrir a webcam")
    exit()

print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi)
        for (ex, ey, ew, eh) in eyes:
            eye_img = face_roi[ey:ey+eh, ex:ex+ew]
            # aplica as transformações e prediz o estado do olho
            input_tensor = transform(eye_img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_tensor)
                pred = torch.argmax(out, dim=1).item()
            label = "ABERTO" if pred == 1 else "FECHADO"
            color = (0, 255, 0) if pred == 1 else (0,0,255)
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), color, 2)
            cv2.putText(frame, label, (x+ex, y+ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Deteccao de Fadiga', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Fim do arquivo realtime.py
