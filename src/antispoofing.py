import cv2
import numpy as np

class FaceAntiSpoofingController:
    """
    Эвристический контроллер для защиты от презентационных атак.
    Отказ от нестабильных Deep Learning моделей в пользу интерпретируемых правил
    и контроля наличия рук (через MediaPipe в main.py).
    """
    def __init__(self, model_path=None):
        pass

    def analyze(self, image: np.ndarray, bbox: list, threshold: float = 0.0) -> dict:
        """
        Анализирует изображение на наличие признаков пересъемки с экрана (clipping).
        """
        # Вырезаем область лица без полей
        x, y, w, h = bbox
        crop = image[max(0, y):y+h, max(0, x):x+w]
        
        if crop.size == 0:
            return {"is_real": True, "attack_type": "none", "score": 1.0}

        # Переводим в Grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Детекция "экранного клиппинга": экраны смартфонов при съемке веб-камерой
        # дают характерные зоны абсолютно белого цвета (яркость > 252)
        white_pixels = np.sum(gray > 252)
        clipping_ratio = white_pixels / gray.size

        # Порог: если более 25% лица - это чисто белый шум, вероятно это экран
        is_screen = clipping_ratio > 0.25
        
        # Итоговый вердикт
        is_real = not is_screen
        score = 1.0 if is_real else 0.0
        
        return {
            "is_real": is_real,
            "attack_type": "screen_replay" if is_screen else "none",
            "score": score
        }