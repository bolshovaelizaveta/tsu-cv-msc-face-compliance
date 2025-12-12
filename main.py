from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import time
import math  
import io
import PIL.Image
import PIL.ImageOps

from src.geometry import FaceGeometryController
from src.quality import FaceQualityController
from src.config import ICAOThresholds

app = FastAPI(
    title="TSU Face Compliance API",
    description="Сервис автоматической проверки фотографий по стандартам ICAO 9303 (ВКР Большова Е.А.)",
    version="0.1.0"
)

# Разрешаем запросы с любых фронтендов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
geo_processor = None
quality_processor = None

@app.on_event("startup")
async def startup_event():
    global geo_processor, quality_processor
    print("Инициализация моделей...")
    geo_processor = FaceGeometryController()
    quality_processor = FaceQualityController()
    print("Система готова к работе.")

def fix_exif_rotation(image_bytes: bytes) -> np.ndarray:
    """
    Исправляет ориентацию изображения на основе метаданных EXIF.
    Pillow корректно разворачивает фото перед конвертацией в массив numpy.
    """
    try:
        img = PIL.Image.open(io.BytesIO(image_bytes))
        img = PIL.ImageOps.exif_transpose(img)
        # Конвертируем в RGB если фото в другом формате (например, RGBA)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Переводим в BGR формат для OpenCV
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None

def safe_float(value: float) -> float:
    # Превращает infinite/nan в 0.0
    if math.isinf(value) or math.isnan(value):
        return 0.0
    return round(float(value), 2)

@app.post("/validate")
async def validate_photo(file: UploadFile = File(...)):
    start_time = time.time()
    
    # Чтение файла
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Файл должен быть изображением")
    
    contents = await file.read()
    
    # Используем фикс EXIF вместо прямого cv2.imdecode
    img = fix_exif_rotation(contents)
    
    if img is None:
        raise HTTPException(400, "Не удалось декодировать изображение или прочитать метаданные")

    # Геометрия
    angles, landmarks = geo_processor.get_head_pose(img)
    
    result = {
        "filename": file.filename,
        "is_compliant": False,
        "errors": [],
        "metrics": {},
        "latency_ms": 0
    }

    if not angles:
        result["errors"].append("Лицо не обнаружено")
        result["latency_ms"] = int((time.time() - start_time) * 1000)
        return result

    # Проверка геометрии
    # Используем safe_float для защиты от бесконечности
    yaw = safe_float(angles['yaw'])
    pitch = safe_float(angles['pitch'])
    roll = safe_float(angles['roll'])

    geo_ok = True
    if abs(yaw) > ICAOThresholds.YAW_MAX:
        result["errors"].append(f"Недопустимый поворот головы (Yaw): {yaw}°")
        geo_ok = False
    if abs(pitch) > ICAOThresholds.PITCH_MAX:
        result["errors"].append(f"Недопустимый наклон головы (Pitch): {pitch}°")
        geo_ok = False
    if abs(roll) > ICAOThresholds.ROLL_MAX:
        result["errors"].append(f"Недопустимый наклон к плечу (Roll): {roll}°")
        geo_ok = False

    # Качество
    raw_quality = quality_processor.get_quality_score(img, landmarks)
    quality_score = safe_float(raw_quality)
    
    quality_ok = True
    if quality_score < ICAOThresholds.MIN_QUALITY_SCORE:
        result["errors"].append(f"Низкое качество изображения (Score: {quality_score} < {ICAOThresholds.MIN_QUALITY_SCORE})")
        quality_ok = False

    # Итог
    result["is_compliant"] = geo_ok and quality_ok
    result["metrics"] = {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "quality_score": quality_score
    }
    
    result["latency_ms"] = int((time.time() - start_time) * 1000)
    
    return result