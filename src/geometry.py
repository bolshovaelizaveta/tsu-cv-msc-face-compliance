import cv2
import mediapipe as mp
import numpy as np
import math
from typing import Tuple, Optional, Dict
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
            max_num_faces=1,
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

    def get_head_pose(self, image: np.ndarray) -> Tuple[Optional[Dict[str, float]], Optional[any]]:
        """
        Вычисляет углы Yaw, Pitch, Roll.
        
        Args:
            image: Изображение в формате BGR (OpenCV).
            
        Returns:
            dict: Словарь с углами {'yaw': float, 'pitch': float, 'roll': float} или None.
            landmarks: Нормализованные ландмарки для отрисовки.
        """
        h, w, _ = image.shape
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return None, None

        landmarks = results.multi_face_landmarks[0]
        image_points = []

        for idx in self.keypoints_indices:
            lm = landmarks.landmark[idx]
            image_points.append([lm.x * w, lm.y * h])

        image_points = np.array(image_points, dtype=np.float64)

        # Матрица камеры для фото
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1)) 

        # Решение PnP
        success, rot_vec, trans_vec = cv2.solvePnP(
            self.model_points, 
            image_points, 
            camera_matrix, 
            dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None, None

        # Получаем матрицу вращения
        rmat, _ = cv2.Rodrigues(rot_vec)

        # Прямое извлечение углов Эйлера из матрицы вращения
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

        # Переводим в градусы
        pitch = math.degrees(x)
        yaw = math.degrees(y)
        roll = math.degrees(z)

        # Коррекция осей 
        
        # Исправляем переворот Pitch на 180 градусов
        if pitch < -90:
            pitch = -(pitch + 180)
        elif pitch > 90:
            pitch = 180 - pitch
            
        # Инвертируем Yaw, чтобы поворот головы влево давал минус, а вправо плюс
        yaw = -yaw 

        # Инвертируем Roll для естественного отображения наклона
        roll = -roll

        return {
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll": float(roll)
        }, landmarks