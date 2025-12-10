#!/usr/bin/env python3
"""
Evaluate_drowsiness.py

Script em tempo real para detectar fadiga usando webcam.
- Usa YOLO para detectar olhos e boca
- Classifica olhos como "Abertos" ou "Fechados"
- Classifica boca como "Yawn" ou "No Yawn"
- Implementa lógica de detecção de fadiga:
  * Olhos fechados + Yawn = FADIGA imediato
  * Olhos abertos + Yawn por >3s = FADIGA
  * Olhos fechados + No Yawn por >5s = FADIGA
  * Outras situações = ATENÇÃO
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pathlib import Path
from collections import deque
import time
import sys


class EyeClassifierCNN(nn.Module):
    """Modelo para classificar olhos (Abertos vs Fechados)"""
    def __init__(self, num_classes=2):
        super(EyeClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class MouthClassifierCNN(nn.Module):
    """Modelo para classificar boca (Yawn vs No Yawn)"""
    def __init__(self, num_classes=2):
        super(MouthClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class DrowsinessDetector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        
        # Carregar modelos de classificação
        self.eye_model = EyeClassifierCNN(num_classes=2).to(self.device)
        self.mouth_model = MouthClassifierCNN(num_classes=2).to(self.device)
        
        # Carregar pesos
        eye_path = Path('models/eyes_model_best.pth')
        mouth_path = Path('models/mouth_model_best.pth')
        
        if not eye_path.exists():
            print(f"Erro: {eye_path} não encontrado")
            raise FileNotFoundError(f"Modelo de olhos não encontrado: {eye_path}")
        if not mouth_path.exists():
            print(f"Erro: {mouth_path} não encontrado")
            raise FileNotFoundError(f"Modelo de boca não encontrado: {mouth_path}")
        
        try:
            self.eye_model.load_state_dict(torch.load(str(eye_path), map_location=self.device))
            self.mouth_model.load_state_dict(torch.load(str(mouth_path), map_location=self.device))
        except Exception as e:
            print(f"Erro ao carregar modelos: {e}")
            raise
        
        self.eye_model.eval()
        self.mouth_model.eval()
        
        # Carregar Haar cascades do OpenCV (mais simples e leve)
        haar_base = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(str(Path(haar_base) / 'haarcascade_frontalface_default.xml'))
        # olhos
        self.eye_cascade = cv2.CascadeClassifier(str(Path(haar_base) / 'haarcascade_eye.xml'))
        # boca (pode não existir em algumas instalações; será None se não carregar)
        mouth_path = Path(haar_base) / 'haarcascade_mcs_mouth.xml'
        if mouth_path.exists():
            self.mouth_cascade = cv2.CascadeClassifier(str(mouth_path))
        else:
            self.mouth_cascade = None
        
        # Transformação de imagem
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Estado de fadiga
        self.eye_closed_start = None
        self.yawn_start = None
        self.attention_start = None
        self.last_status = "ATENÇÃO"
        
        # Histórico para suavização
        self.eye_history = deque(maxlen=5)
        self.mouth_history = deque(maxlen=5)

    def classify_eye(self, eye_crop):
        """Classifica se olho está aberto ou fechado"""
        if eye_crop is None or eye_crop.size == 0:
            return None, None
        
        try:
            from PIL import Image
            if isinstance(eye_crop, str):
                img = Image.open(eye_crop).convert('RGB')
            else:
                img = Image.fromarray(eye_crop).convert('RGB')
            
            inp = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.eye_model(inp)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            pred = int(probs.argmax())
            confidence = float(probs[pred])
            
            # 0 = Fechado, 1 = Aberto
            state = "Fechado" if pred == 0 else "Aberto"
            return state, confidence
        except Exception as e:
            print(f"Erro ao classificar olho: {e}")
            return None, None

    def classify_mouth(self, mouth_crop):
        """Classifica se boca está bocejando ou não"""
        if mouth_crop is None or mouth_crop.size == 0:
            return None, None
        
        try:
            from PIL import Image
            if isinstance(mouth_crop, str):
                img = Image.open(mouth_crop).convert('RGB')
            else:
                img = Image.fromarray(mouth_crop).convert('RGB')
            
            inp = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.mouth_model(inp)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            pred = int(probs.argmax())
            confidence = float(probs[pred])
            
            # 0 = No Yawn, 1 = Yawn
            state = "No Yawn" if pred == 0 else "Yawn"
            return state, confidence
        except Exception as e:
            print(f"Erro ao classificar boca: {e}")
            return None, None

    def detect_drowsiness(self, eye_state, mouth_state):
        """
        Implementa lógica de detecção de fadiga:
        - Olhos fechados + Yawn = FADIGA imediato
        - Olhos abertos + Yawn por >3s = FADIGA
        - Olhos fechados + No Yawn por >5s = FADIGA
        - Outras situações = ATENÇÃO
        """
        current_time = time.time()
        status = "ATENÇÃO"
        
        # Condição 1: Olhos fechados + Yawn = FADIGA imediato
        if eye_state == "Fechado" and mouth_state == "Yawn":
            status = "FADIGA"
            self.eye_closed_start = None
            self.yawn_start = None
            return status

        # Condição 2: Olhos abertos + Yawn por >3s = FADIGA
        if eye_state == "Aberto" and mouth_state == "Yawn":
            if self.yawn_start is None:
                self.yawn_start = current_time
            elif current_time - self.yawn_start >= 3.0:
                status = "FADIGA"
            else:
                status = "ATENÇÃO"
        else:
            self.yawn_start = None
        
        # Condição 3: Olhos fechados + No Yawn por >5s = FADIGA
        if eye_state == "Fechado" and mouth_state == "No Yawn":
            if self.eye_closed_start is None:
                self.eye_closed_start = current_time
            elif current_time - self.eye_closed_start >= 5.0:
                status = "FADIGA"
            else:
                status = "ATENÇÃO"
        else:
            self.eye_closed_start = None
        
        # Outras situações = ATENÇÃO
        if status != "FADIGA":
            status = "ATENÇÃO"
        
        self.last_status = status
        return status

    def run(self, camera_id=0):
        """Executa detecção em tempo real"""
        print("Iniciando webcam...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Erro: Não foi possível abrir a webcam")
            return
        
        # Configurar câmera para melhor performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Pressione 'q' para sair")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detectar face/olhos/boca com Haar cascades (mais simples)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            eye_state = None
            mouth_state = None

            for (fx, fy, fw, fh) in faces:
                # desenhar retângulo da face
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (200, 200, 0), 2)

                # Região superior para olhos
                eye_region_gray = gray[fy:fy + int(fh/2), fx:fx + fw]
                eye_region_color = frame[fy:fy + int(fh/2), fx:fx + fw]

                eyes = self.eye_cascade.detectMultiScale(eye_region_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
                # usar a primeira detecção de olho para classificação (simples)
                if len(eyes) > 0:
                    ex, ey, ew, eh = eyes[0]
                    # ajustar coordenadas relativas para a imagem completa
                    eye_crop = eye_region_color[ey:ey+eh, ex:ex+ew]
                    eye_state, eye_conf = self.classify_eye(eye_crop)
                    cv2.rectangle(frame, (fx+ex, fy+ey), (fx+ex+ew, fy+ey+eh), (0, 255, 0), 2)
                    if eye_state:
                        cv2.putText(frame, f"Eye: {eye_state} ({eye_conf:.2f})", (fx+ex, fy+ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # Região inferior para boca
                mouth_region_gray = gray[fy + int(fh/2):fy+fh, fx:fx + fw]
                mouth_region_color = frame[fy + int(fh/2):fy+fh, fx:fx + fw]

                mouth_detected = False
                if self.mouth_cascade is not None:
                    mouths = self.mouth_cascade.detectMultiScale(mouth_region_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    if len(mouths) > 0:
                        mx, my, mw, mh = mouths[0]
                        mouth_crop = mouth_region_color[my:my+mh, mx:mx+mw]
                        mouth_state, mouth_conf = self.classify_mouth(mouth_crop)
                        cv2.rectangle(frame, (fx+mx, fy+int(fh/2)+my), (fx+mx+mw, fy+int(fh/2)+my+mh), (255,0,0), 2)
                        if mouth_state:
                            cv2.putText(frame, f"Mouth: {mouth_state} ({mouth_conf:.2f})", (fx+mx, fy+int(fh/2)+my-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                        mouth_detected = True

                # fallback: se cascade de boca não existir ou não detectar, use a região central inferior
                if not mouth_detected:
                    # definir região central inferior como provável boca
                    mx, my, mw, mh = int(fw*0.2), int(fh*0.6), int(fw*0.6), int(fh*0.25)
                    mouth_crop = frame[fy+my:fy+my+mh, fx+mx:fx+mx+mw]
                    if mouth_crop.size != 0:
                        mouth_state, mouth_conf = self.classify_mouth(mouth_crop)
                        cv2.rectangle(frame, (fx+mx, fy+my), (fx+mx+mw, fy+my+mh), (255,0,0), 2)
                        if mouth_state:
                            cv2.putText(frame, f"Mouth: {mouth_state} ({mouth_conf:.2f})", (fx+mx, fy+my-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            
            # Detectar fadiga (só quando tivermos uma classificação para olho e boca)
            if eye_state is not None and mouth_state is not None:
                status = self.detect_drowsiness(eye_state, mouth_state)
            else:
                status = "ATENÇÃO"
            
            # Exibir status
            color = (0, 0, 255) if status == "FADIGA" else (0, 255, 0)  # Vermelho ou Verde
            cv2.putText(frame, f"Status: {status}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            
            if eye_state:
                cv2.putText(frame, f"Olho: {eye_state}", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if mouth_state:
                cv2.putText(frame, f"Boca: {mouth_state}", (20, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Mostrar frame
            cv2.imshow('Drowsiness Detection', frame)
            
            # Sair com 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam fechada")


def main():
    try:
        detector = DrowsinessDetector()
        detector.run(camera_id=0)
    except Exception as e:
        print(f"Erro na execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
