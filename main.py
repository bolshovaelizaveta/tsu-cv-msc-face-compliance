from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import time
import math  
import io
import PIL.Image
import PIL.ImageOps

from fastapi.responses import FileResponse
from src.geometry import FaceGeometryController
from src.quality import FaceQualityController, check_photometry
from src.config import ICAOThresholds

app = FastAPI(
    title="TSU Face Compliance API",
    description="Сервис автоматической проверки фотографий по стандартам ICAO 9303 (ВКР Большова Е.А.)",
    version="1.0.0"
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
    
    # Формируем ответ
    result = {
        "filename": file.filename,
        "is_compliant": False,
        "errors": [],
        "metrics": {
            "yaw": 0.0, "pitch": 0.0, "roll": 0.0, 
            "ear": 0.0, "quality_score": 0.0
        },
        "latency_ms": 0
    }

    # Проверка освещения
    photo_ok, photo_msg = check_photometry(img)
    if not photo_ok:
        result["errors"].append(f"Ошибка освещения: {photo_msg}")
        result["latency_ms"] = int((time.time() - start_time) * 1000)
        return result
    
    # Геометрия
    geo_result = geo_processor.analyze(img)

    if geo_result.get("error"):
        result["errors"].append(geo_result["error"]) # NO_FACE, MULTIPLE_FACES
        result["latency_ms"] = int((time.time() - start_time) * 1000)
        return result

    # Если геометрия найдена
    angles = geo_result["angles"]
    ear = geo_result["ear"]
    landmarks = geo_result["landmarks"]
    
    yaw = safe_float(angles['yaw'])
    pitch = safe_float(angles['pitch'])
    roll = safe_float(angles['roll'])
    ear = safe_float(ear)

    geo_ok = True
    if abs(yaw) > ICAOThresholds.YAW_MAX:
        result["errors"].append(f"Недопустимый поворот головы (Yaw): {yaw}°")
        geo_ok = False
    if abs(pitch) > ICAOThresholds.PITCH_MAX:
        result["errors"].append(f"Недопустимый наклон головы (Pitch): {pitch}°")
        geo_ok = False
    if ear < 0.15: 
        result["errors"].append("Глаза закрыты")
        geo_ok = False
    if abs(roll) > ICAOThresholds.ROLL_MAX:
        result["errors"].append(f"Недопустимый наклон к плечу (Roll): {roll}°")
        geo_ok = False

    # Качество
    quality_score = 0.0
    quality_ok = True
    
    if geo_ok:
        raw_quality = quality_processor.get_quality_score(img, landmarks)
        quality_score = safe_float(raw_quality)
        
        if quality_score < ICAOThresholds.MIN_QUALITY_SCORE:
            result["errors"].append(f"Низкое качество (Score: {quality_score})")
            quality_ok = False

    # Итог
    result["is_compliant"] = geo_ok and quality_ok
    result["metrics"] = {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "ear": ear,
        "quality_score": quality_score
    }
    result["latency_ms"] = int((time.time() - start_time) * 1000)

    return result

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/analyze_live")
async def analyze_live(file: UploadFile = File(...)):
    """
    Облегченный метод для живых подсказок в самом интерфейсе.
    Не запускает MagFace для экономии ресурсов.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"hints": []}

    hints = []
    
    # Проверка света и резкости
    photo_ok, photo_msg = check_photometry(img)
    if not photo_ok:
        if photo_msg == "TOO_DARK": hints.append("Слишком темно, включите свет")
        if photo_msg == "BLURRY": hints.append("Изображение размыто, протрите камеру")
        if photo_msg == "LOW_CONTRAST": hints.append("Низкий контраст")

    # Геометрия
    geo_result = geo_processor.analyze(img)
    if geo_result.get("error"):
        if geo_result["error"] == "NO_FACE": hints.append("Лицо не обнаружено")
        if geo_result["error"] == "MULTIPLE_FACES": hints.append("В кадре должно быть только один человек")
    else:
        angles = geo_result["angles"]
        ear = geo_result["ear"]
        
        if abs(angles['yaw']) > ICAOThresholds.YAW_MAX: hints.append("Поверните голову прямо")
        if angles['pitch'] > ICAOThresholds.PITCH_MAX: hints.append("Опустите голову чуть ниже")
        if angles['pitch'] < -ICAOThresholds.PITCH_MAX: hints.append("Поднимите голову чуть выше")
        if ear < 0.15: hints.append("Не закрывайте глаза")

    return {"hints": hints}