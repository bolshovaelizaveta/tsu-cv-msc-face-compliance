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

    # Пропорции лица по ICAO
    # Расстояние от нижней границы до глаз (50-70% от высоты кадра)
    EYE_DIST_BOTTOM_MIN: float = 0.20 
    EYE_DIST_BOTTOM_MAX: float = 0.85 
    # Отношение ширины лица к ширине кадра
    FACE_WIDTH_RATIO_MIN: float = 0.15  
    FACE_WIDTH_RATIO_MAX: float = 0.85  

    # Порог качества MagFace
    MIN_QUALITY_SCORE: float = 7.0 # Проходит и на домашних веб-камерах

    # Антиспуфинг
    SPOOF_MOIRE_THRESH: float = 4.5    # Порог для частотного спектра (экраны)
    SPOOF_LIVENESS_MIN: float = 0.85   # 85% уверенности, что это живой человек