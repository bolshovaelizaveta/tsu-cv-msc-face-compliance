import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

def train_yolo_classifier():
    print("Инициализация модели YOLOv8 (Classification)...")
    model = YOLO("yolov8m-cls.pt")

    dataset_path = os.path.abspath("data/yolo_occlusion_dataset")

    print("\nЗапуск обучения...")
    
    results = model.train(
        data=dataset_path,
        
        # 1. Основные параметры 
        epochs=50,                  # 50 эпох
        batch=8,                   # Батч-сайз 8
        optimizer='SGD',            # Оптимизатор SGD
        cos_lr=True,                # Косинусный Learning Rate (true)
        amp=True,                   # Automatic Mixed Precision 
        freeze=8,                   # Заморозка первых 8 слоев 
        
        # 2. Learning Rate 
        lr0=0.01,                   
        
        # 3. Аугментации 
        degrees=15.0,               # Наклоны
        hsv_s=0.5,                  # Насыщенность
        hsv_v=0.5,                  # Яркость
        scale=0.5,                  # Масштабирование
        erasing=0.4,                # Зернистость
        
        # 4. Системные параметры
        imgsz=224,                  
        device="mps",               
        project="models/train_runs",
        name="occlusion_yolov8m",
        exist_ok=True
    )
    
    print("\nОбучение завершено!")
    print("Веса лежат здесь: models/occlusion_yolov8m.pt")

if __name__ == "__main__":
    train_yolo_classifier()