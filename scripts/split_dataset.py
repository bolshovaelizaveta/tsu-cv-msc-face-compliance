import os
import shutil
import random
from tqdm import tqdm

def split_for_yolo():
    source_dir = "data" 
    target_dir = "data/yolo_occlusion_dataset"
    
    classes = ["clean", "clear_glasses", "headwear", "occluded"]
    
    for cls in classes:
        os.makedirs(os.path.join(target_dir, "train", cls), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "val", cls), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "test", cls), exist_ok=True) 
        
        src_cls_dir = os.path.join(source_dir, cls)
        if not os.path.exists(src_cls_dir):
            print(f"Ошибка: Папка {src_cls_dir} не найдена!")
            continue
            
        files = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(files)
        
        # 70% train, 15% val, 15% test
        train_idx = int(len(files) * 0.70)
        val_idx = int(len(files) * 0.85)
        
        train_files = files[:train_idx]
        val_files = files[train_idx:val_idx]
        test_files = files[val_idx:]
        
        print(f"\nКопирование класса {cls}...")
        for f in tqdm(train_files, desc="Train"):
            shutil.copy(os.path.join(src_cls_dir, f), os.path.join(target_dir, "train", cls, f))
        for f in tqdm(val_files, desc="Val"):
            shutil.copy(os.path.join(src_cls_dir, f), os.path.join(target_dir, "val", cls, f))
        for f in tqdm(test_files, desc="Test"):
            shutil.copy(os.path.join(src_cls_dir, f), os.path.join(target_dir, "test", cls, f))
            
        print(f"Класс {cls} готов: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

if __name__ == "__main__":
    split_for_yolo()