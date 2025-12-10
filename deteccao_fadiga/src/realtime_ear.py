# src/realtime_ear.py

import cv2
import time
import numpy as np
import mediapipe as mp

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True,
                             max_num_faces=1,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

# Índices de landmarks para olhos (para olho esquerdo/direito) — conforme MediaPipe
LEFT_EYE = [33, 133, 159, 145, 153, 144]   # ou conjunto apropriado
RIGHT_EYE = [362, 263, 386, 374, 380, 373]

def eye_aspect_ratio(eye_landmarks, landmarks, w, h):
    # eye_landmarks: lista de índices
    # landmarks: face_landmarks.landmark
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_landmarks]
    # para simplificar: usar dois distâncias verticais e uma horizontal
    # aqui como exemplo: p2-p6, p3-p5, p1-p4 como na definição original
    p1, p2, p3, p4, p5, p6 = coords
    # distâncias Euclidianas
    def dist(a,b): return np.linalg.norm(np.array(a)-np.array(b))
    vertical1 = dist(p2, p6)
    vertical2 = dist(p3, p5)
    horizontal = dist(p1, p4)
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# parâmetros
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20  # por exemplo, ~0.5–1s dependendo da taxa de frames
counter = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao abrir webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = eye_aspect_ratio(LEFT_EYE, landmarks, w, h)
        right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks, w, h)
        ear = (left_ear + right_ear) / 2.0

        if ear < EAR_THRESHOLD:
            counter += 1
        else:
            counter = 0

        # se olhos fechados por muitos frames → fadiga
        if counter >= CONSEC_FRAMES:
            cv2.putText(frame, "FADIGA!", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    cv2.imshow("Detecção de Fadiga (EAR)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
