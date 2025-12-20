from dataclasses import dataclass
import os

# Пути
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "magface_iresnet50.onnx")


@dataclass
class ICAOThresholds:
    YAW_MAX: float = 20.0    # Поворот головы влево-вправо
    PITCH_MAX: float = 20.0  # Наклон головы вверх-вниз
    ROLL_MAX: float = 20.0   # Наклон головы к плечам
    
    # Порог качества MagFace
    MIN_QUALITY_SCORE: float = 9.0