import os
import cv2
import shutil
from tqdm import tqdm

from src.geometry import FaceGeometryController
from src.config import ICAOThresholds

def clean_and_sort(source_dir, rejected_dir):
    geo = FaceGeometryController()
    os.makedirs(rejected_dir, exist_ok=True)
    
    files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Анализ {len(files)} фото. Не подходящие перенесутся в {rejected_dir}")
    
    for filename in tqdm(files):
        file_path = os.path.join(source_dir, filename)
        img = cv2.imread(file_path)
        if img is None: continue
            
        result = geo.analyze(img)
        
        # Если есть ошибка (нет лица, много лиц, глаза, углы)
        is_bad = False
        if result.get("error") or \
           abs(result["angles"]["yaw"]) > ICAOThresholds.YAW_MAX or \
           abs(result["angles"]["pitch"]) > ICAOThresholds.PITCH_MAX or \
           result["ear"] < 0.15:
            is_bad = True
                
        if is_bad:
            # Перемещаем 
            shutil.move(file_path, os.path.join(rejected_dir, filename))

if __name__ == "__main__":
    clean_and_sort("data/benchmark/source", "data/benchmark/rejected_by_cleaner")