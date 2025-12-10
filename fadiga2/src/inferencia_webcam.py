# ARQUIVO 9
# src/inferencia_webcam.py
#
 
import cv2
import dlib
import torch
import numpy as np
import torchvision.transforms as transforms
from model_eyes import EyesCNN
from model_mouth import MouthCNN
from PIL import Image

# ============================================================
# CONFIGURAÇÕES DO USUÁRIO
# ============================================================
EYES_MODEL_PATH = "models/eyes_model.pth"
MOUTH_MODEL_PATH = "models/mouth_model.pth"

IMAGE_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES_EYES = ["Closed_Eyes", "Open_Eyes"]
CLASSES_MOUTH = ["No_Yawn", "Yawn"]

# Detector facial
detector = dlib.get_frontal_face_detector()

# Previsor de marcos faciais (68 pontos)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# ============================================================
# FUNÇÃO DE PRÉ-PROCESSAMENTO
# ============================================================
def preprocess_roi(roi):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    pil_img = Image.fromarray(roi)
    return transform(pil_img).unsqueeze(0).to(DEVICE)


# ============================================================
# FUNÇÃO DE PREDIÇÃO
# ============================================================
def predict(model, roi):
    with torch.no_grad():
        output = model(roi)
        _, pred = torch.max(output, 1)
        return pred.item()


# ============================================================
# CARREGAR MODELOS
# ============================================================
eyes_model = EyesCNN(num_classes=2).to(DEVICE)
eyes_model.load_state_dict(torch.load(EYES_MODEL_PATH, map_location=DEVICE))
eyes_model.eval()

mouth_model = MouthCNN(num_classes=2).to(DEVICE)
mouth_model.load_state_dict(torch.load(MOUTH_MODEL_PATH, map_location=DEVICE))
mouth_model.eval()

print("\nModelos carregados. Iniciando webcam...\n")

# ============================================================
# LOOP PRINCIPAL WEBCAM
# ============================================================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Pega marcos faciais
        shape = predictor(gray, face)
        points = np.array([[p.x, p.y] for p in shape.parts()])

        # ------------------------------------------------------------
        # ROI: Olhos (pontos 36–41 e 42–47)
        # ------------------------------------------------------------
        left_eye_pts = points[36:42]
        right_eye_pts = points[42:48]

        # Define bounding boxes para olhos
        lx1, ly1 = np.min(left_eye_pts, axis=0)
        lx2, ly2 = np.max(left_eye_pts, axis=0)

        rx1, ry1 = np.min(right_eye_pts, axis=0)
        rx2, ry2 = np.max(right_eye_pts, axis=0)

        left_eye = frame[ly1:ly2, lx1:lx2]
        right_eye = frame[ry1:ry2, rx1:rx2]

        # Combine os olhos (média simples)
        try:
            left_eye_roi = preprocess_roi(left_eye)
            right_eye_roi = preprocess_roi(right_eye)

            left_pred = predict(eyes_model, left_eye_roi)
            right_pred = predict(eyes_model, right_eye_roi)

            # Se qualquer olho estiver fechado → fechado
            eyes_state = "Closed_Eyes" if (left_pred == 0 or right_pred == 0) else "Open_Eyes"

        except:
            eyes_state = "Unknown"

        # ------------------------------------------------------------
        # ROI: Boca (pontos 48–67)
        # ------------------------------------------------------------
        mouth_pts = points[48:68]
        mx1, my1 = np.min(mouth_pts, axis=0)
        mx2, my2 = np.max(mouth_pts, axis=0)

        mouth = frame[my1:my2, mx1:mx2]

        try:
            mouth_roi = preprocess_roi(mouth)
            mouth_pred = predict(mouth_model, mouth_roi)
            mouth_state = CLASSES_MOUTH[mouth_pred]
        except:
            mouth_state = "Unknown"

        # ------------------------------------------------------------
        # DECISÃO FINAL
        # ------------------------------------------------------------
        if eyes_state == "Closed_Eyes" or mouth_state == "Yawn":
            status = "FADIGA"
            color = (0, 0, 255)  # vermelho
        else:
            status = "ATENÇÃO"
            color = (0, 255, 0)  # verde

        # Exibir texto na tela
        cv2.putText(frame, f"Olhos: {eyes_state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(frame, f"Boca: {mouth_state}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(frame, f"Estado: {status}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

    cv2.imshow("Deteccao de Fadiga", frame)

    # Pressione Q para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
