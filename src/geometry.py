import cv2
import mediapipe as mp
import numpy as np
import math
from typing import Dict, Any
from src.config import ICAOThresholds

class FaceGeometryController:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # 3D координаты стандартного лица
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye
            (225.0, 170.0, -135.0),      # Right eye
            (-150.0, -150.0, -125.0),    # Left mouth
            (150.0, -150.0, -125.0)      # Right mouth
        ], dtype=np.float64)

        self.keypoints_indices = [1, 152, 33, 263, 61, 291]
        
        # Индексы для EAR (Глаза)
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]
        self.right_eye_indices = [362, 385, 387, 263, 373, 380]
        
        # Индексы для MAR (Рот) - точки верхних и нижних губ
        self.mouth_indices = [61, 37, 267, 291, 314, 84]

    def _calculate_ratio(self, landmarks, w, h, indices):
        """Универсальный расчет Aspect Ratio (для EAR и MAR)"""
        pts = []
        for idx in indices:
            lm = landmarks.landmark[idx]
            pts.append(np.array([lm.x * w, lm.y * h]))
        
        # Вертикальные расстояния
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        # Горизонтальное расстояние
        hor = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * hor)

    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        h, w, _ = image.shape
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return {"error": "NO_FACE"}

        landmarks = results.multi_face_landmarks[0]
        
        # 1. Расчет углов Эйлера (PnP)
        image_points = np.array([[landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h] 
                                 for idx in self.keypoints_indices], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        
        success, rot_vec, _ = cv2.solvePnP(self.model_points, image_points, camera_matrix, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)

        yaw, pitch, roll = 0.0, 0.0, 0.0
        if success:
            rmat, _ = cv2.Rodrigues(rot_vec)
            sy = math.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
            if sy > 1e-6:
                pitch = math.degrees(math.atan2(rmat[2,1], rmat[2,2]))
                yaw = math.degrees(math.atan2(-rmat[2,0], sy))
                roll = math.degrees(math.atan2(rmat[1,0], rmat[0,0]))
            
            # Коррекция осей под твой стандарт
            if pitch < -90: pitch = -(pitch + 180)
            elif pitch > 90: pitch = 180 - pitch
            yaw, roll = -yaw, -roll

        # 2. Расчет открытости глаз (EAR)
        ear_l = self._calculate_ratio(landmarks, w, h, self.left_eye_indices)
        ear_r = self._calculate_ratio(landmarks, w, h, self.right_eye_indices)
        avg_ear = (ear_l + ear_r) / 2.0

        # 3. Расчет закрытости рта (MAR)
        mar = self._calculate_ratio(landmarks, w, h, self.mouth_indices)

        # 4. Расчет масштаба лица (Vertical Coverage)
        # Расстояние от макушки (10) до подбородка (152) в процентах от высоты кадра
        face_height_pct = abs(landmarks.landmark[152].y - landmarks.landmark[10].y)

        return {
            "angles": {"yaw": yaw, "pitch": pitch, "roll": roll},
            "ear": avg_ear,
            "mar": mar,
            "face_height": face_height_pct,
            "landmarks": landmarks
        }