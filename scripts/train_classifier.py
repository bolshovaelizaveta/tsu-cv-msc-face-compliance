import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

def train_yolo_classifier(freeze_layers, run_name):
    print(f"\nИнициализация модели YOLOv8 (Запуск: {run_name})...")
    model = YOLO("yolov8m-cls.pt")

    dataset_path = os.path.abspath("data/yolo_occlusion_dataset")

    print(f"\nЗапуск обучения (Заморожено слоев: {freeze_layers})...")
    
    results = model.train(
        data=dataset_path,
        
        # 1. Основные параметры 
        epochs=50,                  # 50 эпох
        batch=8,                    # Батч-сайз 8
        optimizer='SGD',            # Оптимизатор SGD
        cos_lr=True,                # Косинусный Learning Rate (true)
        amp=True,                   # Automatic Mixed Precision 
        freeze=freeze_layers,       # Заморозка первых 8 слоев 
        
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
        name=run_name,             
        exist_ok=True
    )
    
    print(f"\nОбучение {run_name} завершено!")
    
    print(f"\nЗапуск контрольного тестирования на выборке TEST для {run_name}...")
    metrics = model.val(data=dataset_path, split='test')
    print(f"Точность на тесте для {run_name}: {metrics.top1:.4f}\n")

if __name__ == "__main__":
    
    # С заморозкой
    train_yolo_classifier(freeze_layers=8, run_name="occlusion_4classes_frozen_8")
    
    # Без заморозки
    train_yolo_classifier(freeze_layers=None, run_name="occlusion_4classes_unfrozen")