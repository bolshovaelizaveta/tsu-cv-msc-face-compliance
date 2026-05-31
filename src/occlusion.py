import os
import cv2
import numpy as np
from ultralytics import YOLO

class FaceOcclusionController:
    """
    Контроллер для проверки перекрытий лица (маски, очки).
    Использует обученную модель YOLOv8m-cls.
    """
    def __init__(self, model_path="models/yolov8m-occlusion.pt"): 
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        self.model = YOLO(model_path)
        
        # 0: clean, 1: clear_glasses, 2: occluded
        self.names = self.model.names

    def analyze(self, face_crop: np.ndarray) -> dict:
        """
        Принимает квадратный кроп лица (numpy array BGR).
        Возвращает словарь с классом и уверенностью.
        """
        results = self.model(face_crop, verbose=False)[0]
        
        # Получаем индекс класса с максимальной вероятностью
        top1_index = results.probs.top1
        confidence = float(results.probs.top1conf)
        predicted_class = self.names[top1_index]

        return {
            "class": predicted_class,
            "confidence": confidence
        }

def check_glare(face_crop: np.ndarray, landmarks) -> bool:
    """
    Быстрая проверка на блики в области глаз.
    Вызывается только если человек в прозрачных очках.
    """
    h, w = face_crop.shape[:2]
    
    # Индексы центров глаз в MediaPipe
    left_eye_center = 468
    right_eye_center = 473
    
    try:
        # Получаем координаты
        lx, ly = int(landmarks.landmark[left_eye_center].x * w), int(landmarks.landmark[left_eye_center].y * h)
        rx, ry = int(landmarks.landmark[right_eye_center].x * w), int(landmarks.landmark[right_eye_center].y * h)
        
        # Вырезаем небольшие квадраты вокруг глаз (область стекол очков)
        box_size = int(w * 0.15) # 15% от ширины лица
        
        # Переводим в градации серого для поиска пересветов
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # Проверяем левый глаз
        left_roi = gray[max(0, ly-box_size):min(h, ly+box_size), max(0, lx-box_size):min(w, lx+box_size)]
        # Проверяем правый глаз
        right_roi = gray[max(0, ry-box_size):min(h, ry+box_size), max(0, rx-box_size):min(w, rx+box_size)]
        
        # Ищем чисто белые пиксели (блики) с яркостью > 230
        _, left_thresh = cv2.threshold(left_roi, 230, 255, cv2.THRESH_BINARY)
        _, right_thresh = cv2.threshold(right_roi, 230, 255, cv2.THRESH_BINARY)
        
        # Если площадь пересвета больше 2% от области глаза -> это сильный блик
        left_glare = (cv2.countNonZero(left_thresh) / (left_roi.size + 1e-6)) > 0.005
        right_glare = (cv2.countNonZero(right_thresh) / (right_roi.size + 1e-6)) > 0.005
        
        return left_glare or right_glare
    except Exception:
        return False 