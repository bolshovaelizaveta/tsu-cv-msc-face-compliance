import os
import time
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

from src.geometry import FaceGeometryController
from src.quality import FaceQualityController, check_photometry
from src.config import ICAOThresholds

def run_benchmark_final(test_dir, output_csv):
    geo = FaceGeometryController()
    quality = FaceQualityController()
    
    results = []
    files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Запуск теста на {len(files)} изображениях...")
    
    for filename in tqdm(files):
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path)
        if img is None: continue

        # Ground Truth
        # ok_ -> 1, fail_ -> 0
        expected_compliant = 1 if filename.startswith("ok_") else 0
        
        start_time = time.perf_counter()
        
        status = "REJECTED"
        error_reason = "None"
        yaw, pitch, roll = 0.0, 0.0, 0.0
        q_score = 0.0
        ear = 0.0

        # Фотометрия (Свет) 
        photo_ok, photo_msg = check_photometry(img)
        if not photo_ok:
            error_reason = photo_msg 
        else:
            # Геометрия 
            geo_result = geo.analyze(img)
            
            if geo_result.get("error"):
                error_reason = geo_result["error"] # NO_FACE, MULTIPLE_FACES
            else:
                # Данные геометрии
                yaw = geo_result["angles"]["yaw"]
                pitch = geo_result["angles"]["pitch"]
                roll = geo_result["angles"]["roll"]
                ear = geo_result["ear"]
                landmarks = geo_result["landmarks"]

                # Проверка порогов геометрии
                geo_ok = (abs(yaw) <= ICAOThresholds.YAW_MAX and 
                          abs(pitch) <= ICAOThresholds.PITCH_MAX and 
                          abs(roll) <= ICAOThresholds.ROLL_MAX)
                
                # Проверка глаз 
                eyes_ok = ear > 0.15 # Порог открытых глаз

                if not geo_ok:
                    error_reason = "BAD_ANGLE"
                elif not eyes_ok:
                    error_reason = "EYES_CLOSED"
                else:
                    # Качество (MagFace) 
                    # Запускаем нейросеть только если всё остальное ОК
                    q_score = quality.get_quality_score(img, landmarks)
                    
                    if q_score >= ICAOThresholds.MIN_QUALITY_SCORE:
                        status = "COMPLIANT"
                    else:
                        error_reason = "LOW_QUALITY"

        latency_ms = (time.perf_counter() - start_time) * 1000
        
        actual_compliant = 1 if status == "COMPLIANT" else 0
        
        results.append({
            "filename": filename,
            "gt": expected_compliant,
            "pred": actual_compliant,
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "quality": round(q_score, 2),
            "latency": round(latency_ms, 2),
            "reason": error_reason
        })

    # Сохранение и отчет
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # Расчет метрик
    tp = len(df[(df['gt'] == 1) & (df['pred'] == 1)])
    tn = len(df[(df['gt'] == 0) & (df['pred'] == 0)])
    fp = len(df[(df['gt'] == 0) & (df['pred'] == 1)])
    fn = len(df[(df['gt'] == 1) & (df['pred'] == 0)])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(df)

    print(f"\nРезультаты (Датасет: {len(df)} фото)")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Latency:   {df['latency'].mean():.2f} ms")
    
    # Топ причин отказа для оригиналов (анализ ошибок)
    fails = df[(df['gt'] == 1) & (df['pred'] == 0)]
    if not fails.empty:
        print("\nПочему не прошли фотографии для регистрации (Top-3):")
        print(fails['reason'].value_counts().head(3))

if __name__ == "__main__":
    run_benchmark_final("data/benchmark/distorted", "docs/benchmark_final.csv")