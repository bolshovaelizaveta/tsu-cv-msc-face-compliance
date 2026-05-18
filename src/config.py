from dataclasses import dataclass
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class ICAOThresholds:
    # Сбалансированные пороги для читательского билета
    # 15-20 градусов — это визуальный анфас, комфортный для человека
    YAW_MAX: float = 20.0    
    PITCH_MAX: float = 20.0  
    ROLL_MAX: float = 15.0   
    
    # Глаза и рот (допускаем улыбку и прищур)
    EYE_OPEN_MIN: float = 0.15     
    MOUTH_CLOSED_MAX: float = 0.35 # Допускает легкую улыбку, но не широко открытый рот

    # Масштаб лица 
    FACE_HEIGHT_MIN: float = 0.25 
    FACE_HEIGHT_MAX: float = 0.65

    # Порог качества MagFace
    MIN_QUALITY_SCORE: float = 7.0 # Проходит и на домашних веб-камерах