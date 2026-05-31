import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from rembg import remove, new_session

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TEST_DIR = os.path.join(PROJECT_ROOT, "data", "matting_test")
IMAGES_DIR = os.path.join(TEST_DIR, "images")
MASKS_DIR = os.path.join(TEST_DIR, "masks")

def prepare_dataset():
    if not os.path.exists(IMAGES_DIR):
        raise FileNotFoundError(f"Папка не найдена: {IMAGES_DIR}")
        
    os.makedirs(MASKS_DIR, exist_ok=True)
    
    # Собираем все картинки
    files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files.sort()
    
    if len(files) == 0:
        print("В папке images пусто!")
        return

    print(f"Найдено {len(files)} изображений. Начинаем переименование и генерацию масок...")
    
    # Инициализируем тяжелую модель для идеальных масок (u2net)
    session = new_session("u2net")
    
    for i, filename in enumerate(tqdm(files, desc="Генерация GT масок")):
        old_path = os.path.join(IMAGES_DIR, filename)
        
        # 1. Переименовываем в формат 001.jpg
        ext = os.path.splitext(filename)[1].lower()
        new_name = f"{i+1:03d}{ext}"
        new_path = os.path.join(IMAGES_DIR, new_name)
        
        if old_path != new_path:
            os.rename(old_path, new_path)
            
        # 2. Генерируем маску
        img = cv2.imread(new_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Получаем RGBA результат (лицо + прозрачный фон)
        result_rgba = remove(img_rgb, session=session)
        
        # Извлекаем Alpha-канал (
        # 4-й канал: 0 - прозрачный (фон), 255 - непрозрачный (человек)
        alpha_mask = result_rgba[:, :, 3]
        
        # Сохраняем маску 
        mask_name = f"{i+1:03d}.png"
        mask_path = os.path.join(MASKS_DIR, mask_name)
        cv2.imwrite(mask_path, alpha_mask)

    print(f"\nМаски сохранены в {MASKS_DIR}")

if __name__ == "__main__":
    prepare_dataset()