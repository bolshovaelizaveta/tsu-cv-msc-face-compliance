import os
import time
import cv2
import numpy as np
from src.geometry import FaceGeometryController
from src.quality import FaceQualityController
from src.config import ICAOThresholds

def run_benchmarks():
    # Инициализация
    geo = FaceGeometryController()
    quality = FaceQualityController()
    
    # Разметка: 1 - Compliant, 0 - Non-compliant
    gt_labels = {
        "compliant_1.jpg": 1, "compliant_2.jpg": 1, "compliant_3.jpg": 1,
        "compliant_4.jpg": 1, "compliant_5.jpg": 1,
        "fail_yaw.jpg": 0, "fail_pitch.jpg": 0, "fail_roll.jpg": 0,
        "fail_quality.jpg": 0, "fail_noface.jpg": 0
    }

    test_dir = "data/test_samples"
    if not os.path.exists(test_dir):
        print(f"Ошибка: Создайте папку {test_dir} и добавьте фотографии")
        return

    results = []
    latencies = []

    tp, tn, fp, fn = 0, 0, 0, 0

    print(f"{'Filename':<20} | {'Status':<10} | {'Score':<7} | {'Time (ms)':<10}")
    print("-" * 55)

    for filename, expected in gt_labels.items():
        img_path = os.path.join(test_dir, filename)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        
        start_time = time.time()
        
        # Геометрия
        angles, landmarks = geo.get_head_pose(img)
        
        is_compliant = False
        q_score = 0.0
        
        if angles and landmarks:
            # Качество
            q_score = quality.get_quality_score(img, landmarks)
            
            # Проверка условий
            geo_ok = (abs(angles['yaw']) <= ICAOThresholds.YAW_MAX and 
                      abs(angles['pitch']) <= ICAOThresholds.PITCH_MAX and 
                      abs(angles['roll']) <= ICAOThresholds.ROLL_MAX)
            quality_ok = q_score >= ICAOThresholds.MIN_QUALITY_SCORE
            
            if geo_ok and quality_ok:
                is_compliant = True

        latency = (time.time() - start_time) * 1000
        latencies.append(latency)
        
        # Считаем метрики
        actual = 1 if is_compliant else 0
        if actual == 1 and expected == 1: tp += 1
        elif actual == 0 and expected == 0: tn += 1
        elif actual == 1 and expected == 0: fp += 1
        elif actual == 0 and expected == 1: fn += 1

        status_str = "PASS" if is_compliant else "REJECT"
        print(f"{filename:<20} | {status_str:<10} | {q_score:<7.2f} | {latency:<10.2f}")

    # Расчет финальных метрик
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_latency = np.mean(latencies)
    fps = 1000 / avg_latency

    print("\n" + "="*30)
    print(f"Итоговые метрики(N={len(latencies)}):")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1-Score:  {f1:.2f}")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"System FPS:  {fps:.1f}")
    print("="*30)

if __name__ == "__main__":
    run_benchmarks()