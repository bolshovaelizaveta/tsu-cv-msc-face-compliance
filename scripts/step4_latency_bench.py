import os
import sys
import time
import numpy as np
import cv2
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.detector_v2 import FaceDetectorV2
from src.geometry import FaceGeometryController
from src.occlusion import FaceOcclusionController
from src.quality import FaceQualityController
from src.segmentation_v2 import FaceSegmentationV2

def benchmark():
    print("[*] Загрузка всех моделей на CPU...")
    # Принудительно отключаем GPU если он есть для чистоты теста CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    detector = FaceDetectorV2(os.path.join(PROJECT_ROOT, "models", "yolov8n-face.pt"))
    geometry = FaceGeometryController()
    occlusion = FaceOcclusionController(os.path.join(PROJECT_ROOT, "models", "yolov8m-occlusion.pt"))
    quality = FaceQualityController()
    segmentation = FaceSegmentationV2(os.path.join(PROJECT_ROOT, "models", "modnet_photographic_portrait_matting.onnx"))

    # Загружаем тестовое изображение
    test_img_path = os.path.join(PROJECT_ROOT, "data", "system_test", "good_photos")
    img_name = [f for f in os.listdir(test_img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][0]
    img = cv2.imread(os.path.join(test_img_path, img_name))
    
    iters = 50
    warmup = 5
    
    timings = {
        "YOLO-Face": [],
        "MediaPipe-Geom": [],
        "YOLO-Occlusion": [],
        "MagFace-Quality": [],
        "MODNet-Matting": []
    }

    print(f"Начинаем бенчмарк ({iters} итераций + {warmup} разогрев)...")

    for i in tqdm(range(iters + warmup)):
        # 1. Detector
        start = time.perf_counter()
        crop, center = detector.get_passport_crop(img)
        if i >= warmup: timings["YOLO-Face"].append(time.perf_counter() - start)

        # 2. Geometry
        start = time.perf_counter()
        geom_res = geometry.analyze(img)
        if i >= warmup: timings["MediaPipe-Geom"].append(time.perf_counter() - start)

        # 3. Occlusion
        start = time.perf_counter()
        _ = occlusion.analyze(crop)
        if i >= warmup: timings["YOLO-Occlusion"].append(time.perf_counter() - start)

        # 4. Quality
        start = time.perf_counter()
        _ = quality.get_quality_score(img, geom_res["landmarks"])
        if i >= warmup: timings["MagFace-Quality"].append(time.perf_counter() - start)

        # 5. Segmentation
        start = time.perf_counter()
        _ = segmentation.remove_background(crop, center)
        if i >= warmup: timings["MODNet-Matting"].append(time.perf_counter() - start)

    print("\n" + "="*60)
    print(f"{'Модуль пайплайна':<25} | {'Среднее (мс)':<15} | {'Std Dev (мс)':<10}")
    print("-"*60)
    
    total_mean = 0
    for name, values in timings.items():
        m_ms = np.mean(values) * 1000
        std_ms = np.std(values) * 1000
        total_mean += m_ms
        print(f"{name:<25} | {m_ms:<15.2f} | {std_ms:<10.2f}")
    
    print("-"*60)
    print(f"{'ИТОГО ВЕСЬ ПАЙПЛАЙН':<25} | {total_mean:<15.2f} | ms")
    print("="*60)
    print(f"Производительность системы: {1000/total_mean:.2f} FPS")

if __name__ == "__main__":
    benchmark()