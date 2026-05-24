from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import time
import math  
import io
import base64
import PIL.Image
import PIL.ImageOps

from fastapi.responses import FileResponse
from src.config import ICAOThresholds
from src.quality import FaceQualityController, check_photometry
from src.geometry import FaceGeometryController
from src.detector_v2 import FaceDetectorV2
from src.segmentation_v2 import FaceSegmentationV2
from src.occlusion import FaceOcclusionController, check_glare

app = FastAPI(
    title="TSU Face Compliance API",
    description="Проверка корректности фотографии лица по стандартам ICAO 9303 (ВКР Большова Е.А.)",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector_processor = None
matting_processor = None
geo_processor = None
occlusion_processor = None
quality_processor = None

@app.on_event("startup")
async def startup_event():
    global detector_processor, matting_processor, geo_processor, occlusion_processor, quality_processor
    print("Инициализация каскада нейросетей...")
    detector_processor = FaceDetectorV2()
    matting_processor = FaceSegmentationV2()
    geo_processor = FaceGeometryController()
    occlusion_processor = FaceOcclusionController()
    quality_processor = FaceQualityController()
    print("Система готова к работе")

def fix_exif_rotation(image_bytes: bytes) -> np.ndarray:
    try:
        img = PIL.Image.open(io.BytesIO(image_bytes))
        img = PIL.ImageOps.exif_transpose(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None

def safe_float(value: float) -> float:
    if math.isinf(value) or math.isnan(value):
        return 0.0
    return round(float(value), 2)

def encode_image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/validate")
async def validate_photo(file: UploadFile = File(...)):
    start_time = time.perf_counter()
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Файл должен быть изображением")
    
    contents = await file.read()
    img = fix_exif_rotation(contents)
    
    if img is None:
        raise HTTPException(400, "Не удалось декодировать изображение")

    result = {
        "filename": file.filename,
        "is_compliant": False,
        "errors": [],
        "metrics": {},
        "latency_ms": 0,
        "processed_image_base64": None
    }
    
    # 0. Проверка минимального разрешения 
    h, w = img.shape[:2]
    min_side = min(h, w)
    max_side = max(h, w)

    # Технический минимум 
    if min_side < 300:
        result["errors"].append(f"Изображение слишком маленькое ({w}x{h}). Минимум 300px.")
        result["latency_ms"] = int((time.perf_counter() - start_time) * 1000)
        return result

    # 2. Апскейл до требований заказчика (640x480)
    if min_side < 480 or max_side < 640:
        scale_w = 640 / w
        scale_h = 480 / h
        scale = max(scale_w, scale_h)
        
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        print(f"DEBUG: Выполнен адаптивный апскейл с {w}x{h} до {img.shape[1]}x{img.shape[0]}")

    # Этап 1: Фотометрия
    photo_ok, photo_msg = check_photometry(img)
    if not photo_ok:
        result["errors"].append(f"Освещение: {photo_msg}")
        result["latency_ms"] = int((time.perf_counter() - start_time) * 1000)
        return result
    
    # Этап 2: Локализация (YOLO-Face)
    wide_crop, face_center = detector_processor.get_passport_crop(img, side_coeff=2.3)
    tight_crop, _ = detector_processor.get_passport_crop(img, side_coeff=1.2)
    
    if wide_crop is None or tight_crop is None:
        result["errors"].append("Лицо не обнаружено или находится слишком далеко")
        result["latency_ms"] = int((time.perf_counter() - start_time) * 1000)
        return result

    # Этап 3: Геометрия (MediaPipe) 
    geo_result = geo_processor.analyze(wide_crop)
    if geo_result.get("error"):
        result["errors"].append(geo_result["error"])
        result["latency_ms"] = int((time.perf_counter() - start_time) * 1000)
        return result

    angles = geo_result["angles"]
    ear, mar, landmarks = geo_result["ear"], geo_result["mar"], geo_result["landmarks"]
    yaw, pitch, roll = safe_float(angles['yaw']), safe_float(angles['pitch']), safe_float(angles['roll'])

    if abs(yaw) > ICAOThresholds.YAW_MAX: result["errors"].append(f"Поворот головы (Yaw): {yaw}°")
    if abs(pitch) > ICAOThresholds.PITCH_MAX: result["errors"].append(f"Наклон головы (Pitch): {pitch}°")
    if abs(roll) > ICAOThresholds.ROLL_MAX: result["errors"].append(f"Наклон к плечу (Roll): {roll}°")
    if ear < ICAOThresholds.EYE_OPEN_MIN: result["errors"].append("Глаза закрыты")
    if mar > ICAOThresholds.MOUTH_CLOSED_MAX: result["errors"].append("Рот открыт")

    if result["errors"]:
        result["latency_ms"] = int((time.perf_counter() - start_time) * 1000)
        return result

    # Этап 4: Классификация окклюзий (YOLO-cls) 
    occ_result = occlusion_processor.analyze(tight_crop)
    occ_class = occ_result['class']
    occ_confidence = safe_float(occ_result.get('confidence', 1.0))

    if occ_class == "occluded":
        result["errors"].append("Обнаружены перекрытия лица")
    elif occ_class == "headwear":
        result["errors"].append("Обнаружен головной убор. Если это религиозный атрибут, требуется ручная проверка.")
    elif occ_class == "clear_glasses":
        if check_glare(tight_crop, landmarks):
            result["errors"].append("Обнаружены сильные блики на очках")

    if result["errors"]:
        result["latency_ms"] = int((time.perf_counter() - start_time) * 1000)
        return result

    # Этап 5: Качество (MagFace) 
    quality_score = safe_float(quality_processor.get_quality_score(wide_crop, landmarks))
    if quality_score < ICAOThresholds.MIN_QUALITY_SCORE:
        result["errors"].append(f"Низкое биометрическое качество (Score: {quality_score})")
        result["latency_ms"] = int((time.perf_counter() - start_time) * 1000)
        return result

    # Этап 6: Нормализация фона (MODNet)
    final_img = matting_processor.remove_background(wide_crop, face_center)
    final_img = cv2.resize(final_img, (600, 600), interpolation=cv2.INTER_LANCZOS4)

    # Успешное завершение
    result["is_compliant"] = True
    result["metrics"] = {
        "yaw": yaw, "pitch": pitch, "roll": roll,
        "quality_score": quality_score,
        "occlusion_class": occ_class,
        "occlusion_confidence": occ_confidence
    }
    result["processed_image_base64"] = encode_image_to_base64(final_img)
    result["latency_ms"] = int((time.perf_counter() - start_time) * 1000)

    return result

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/analyze_live")
async def analyze_live(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"hints": []}

    hints = []
    
    # 1. Фотометрия
    photo_ok, photo_msg = check_photometry(img)
    if not photo_ok:
        if photo_msg == "TOO_DARK": hints.append("Слишком темно, включите свет")
        if photo_msg == "BLURRY": hints.append("Изображение размыто")

    # 2. Локализация (YOLO-Face)
    wide_crop, _ = detector_processor.get_passport_crop(img, side_coeff=2.3)
    tight_crop, _ = detector_processor.get_passport_crop(img, side_coeff=1.2)
    
    if wide_crop is None or tight_crop is None:
        hints.append("Лицо не обнаружено")
        return {"hints": hints}

    # 3. Геометрия (MediaPipe)
    geo_result = geo_processor.analyze(wide_crop)
    if geo_result.get("error"):
        if geo_result["error"] == "MULTIPLE_FACES": 
            hints.append("В кадре должен быть один человек")
    else:
        angles = geo_result["angles"]
        if abs(angles['yaw']) > ICAOThresholds.YAW_MAX: hints.append("Поверните голову прямо")
        if angles['pitch'] > ICAOThresholds.PITCH_MAX: hints.append("Опустите голову чуть ниже")
        if angles['pitch'] < -ICAOThresholds.PITCH_MAX: hints.append("Поднимите голову чуть выше")
        if geo_result['ear'] < ICAOThresholds.EYE_OPEN_MIN: hints.append("Не закрывайте глаза")

    # 4. Окклюзии (YOLO-cls)
    occ_result = occlusion_processor.analyze(tight_crop)
    if occ_result['class'] == "occluded":
        hints.append("Что-то мешает для фотографии, уберите, пожалуйста")
    elif occ_result['class'] == "headwear":
        hints.append("Пожалуйста, снимите головной убор")
    elif occ_result['class'] == "clear_glasses":
        hints.append("Очки обнаружены: убедитесь, что нет бликов")

    return {"hints": hints}