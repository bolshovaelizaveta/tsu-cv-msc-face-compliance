import cv2
import mediapipe as mp
import numpy as np
import math
from typing import Tuple, Optional, Dict, Any
from src.config import ICAOThresholds

class FaceGeometryController:
    """
    Класс для извлечения геометрических характеристик лица.
    Реализует расчет углов Эйлера (Yaw, Pitch, Roll) и алгоритм PnP
    для вычисления углов поворота головы.
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=2, # Если найдет два - выкидываем ошибку 
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # 3D координаты стандартного лица. Координаты взяты из антропометрических баз
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Кончик носа (Nose tip)
            (0.0, -330.0, -65.0),        # Подбородок (Chin)
            (-225.0, 170.0, -135.0),     # Левый глаз, внешний угол
            (225.0, 170.0, -135.0),      # Правый глаз, внешний угол
            (-150.0, -150.0, -125.0),    # Левый угол рта
            (150.0, -150.0, -125.0)      # Правый угол рта
        ], dtype=np.float64)

        # Индексы соответствующих точек в MediaPipe (468 points map)
        # 1 - nose tip, 199 - chin, 33 - left eye, 263 - right eye, 61 - left mouth, 291 - right mouth
        self.keypoints_indices = [1, 152, 33, 263, 61, 291]

        # Индексы для расчета соотношения сторон глаза
        # [p1, p2, p3, p4, p5, p6] для левого и правого глаза
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]
        self.right_eye_indices = [362, 385, 387, 263, 373, 380]

    def _calculate_ear(self, landmarks, w, h, indices):
        """Вспомогательный метод для расчета коэффициента открытости глаз"""
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
        """
        Полный анализ геометрии.
        Returns:
            dict с ключами: 'angles', 'ear', 'face_count', 'landmarks', 'error'
        """
        h, w, _ = image.shape
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        # Проверка наличия лиц
        if not results.multi_face_landmarks:
            return {"error": "NO_FACE", "face_count": 0}
        
        # Проверка количества лиц (если больше 1 — сразу отказ)
        face_count = len(results.multi_face_landmarks)
        if face_count > 1:
            return {"error": "MULTIPLE_FACES", "face_count": face_count}

        # Берем первое лицо
        landmarks = results.multi_face_landmarks[0]
        
        # PnP (Углы поворота)
        image_points = []
        for idx in self.keypoints_indices:
            lm = landmarks.landmark[idx]
            image_points.append([lm.x * w, lm.y * h])
        image_points = np.array(image_points, dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))

        success, rot_vec, trans_vec = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        yaw, pitch, roll = 0.0, 0.0, 0.0
        if success:
            rmat, _ = cv2.Rodrigues(rot_vec)
            sy = math.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
            singular = sy < 1e-6
            if not singular:
                x = math.atan2(rmat[2,1], rmat[2,2])
                y = math.atan2(-rmat[2,0], sy)
                z = math.atan2(rmat[1,0], rmat[0,0])
            else:
                x = math.atan2(-rmat[1,2], rmat[1,1])
                y = math.atan2(-rmat[2,0], sy)
                z = 0

            pitch = math.degrees(x)
            yaw = math.degrees(y)
            roll = math.degrees(z)

            # Коррекция осей 
            if pitch < -90: pitch = -(pitch + 180)
            elif pitch > 90: pitch = 180 - pitch
            yaw = -yaw 
            roll = -roll

        # Глаза
        ear_left = self._calculate_ear(landmarks, w, h, self.left_eye_indices)
        ear_right = self._calculate_ear(landmarks, w, h, self.right_eye_indices)
        avg_ear = (ear_left + ear_right) / 2.0

        return {
            "error": None,
            "face_count": 1,
            "angles": {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)},
            "ear": float(avg_ear),
            "landmarks": landmarks
        }