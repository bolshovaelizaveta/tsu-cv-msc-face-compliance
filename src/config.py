from dataclasses import dataclass
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class ICAOThresholds:
    # Сбалансированные пороги для читательского билета
    YAW_MAX: float = 12.5    
    PITCH_MAX: float = 12.5  
    ROLL_MAX: float = 12.5   
    
    # Глаза и рот (допускаем улыбку и прищур)
    EYE_OPEN_MIN: float = 0.18     
    MOUTH_CLOSED_MAX: float = 0.55 # Допускает легкую улыбку, но не широко открытый рот

# Оценка направления взгляда
    GAZE_MIN: float = 0.65
    GAZE_MAX: float = 1.35   # Горизонтальный максимум
    GAZE_Y_MAX: float = 1.35 # Вертикальный максимум

    # Масштаб лица 
    FACE_HEIGHT_MIN: float = 0.35 
    FACE_HEIGHT_MAX: float = 0.60

    # Порог качества MagFace
    MIN_QUALITY_SCORE: float = 7.0 # Проходит и на домашних веб-камерах