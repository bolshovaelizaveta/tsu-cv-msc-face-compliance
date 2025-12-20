import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm

from src.config import ICAOThresholds

class DatasetDistorter:
    """
    Класс для генерации синтетических нарушений стандартов ICAO.
    Используется для создания тестовой выборки с известными дефектами.
    """
    @staticmethod
    def apply_blur(image, level=17):
        return cv2.GaussianBlur(image, (level, level), 0)

    @staticmethod
    def apply_rotation(image, angle=40):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def apply_low_light(image, factor=0.25):
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def prepare_benchmark_data(source_path, output_path):
    distorter = DatasetDistorter()
    files = [f for f in os.listdir(source_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Обработка {len(files)} исходных изображений...")
    
    for filename in tqdm(files):
        img_path = os.path.join(source_path, filename)
        img = cv2.imread(img_path)
        if img is None: continue

        # Сохраняем оригинал (Compliant)
        cv2.imwrite(os.path.join(output_path, f"ok_{filename}"), img)
        # Поворот (Гарантированный брак > 25 градусов)
        cv2.imwrite(os.path.join(output_path, f"fail_rot_{filename}"), distorter.apply_rotation(img))
        # Размытие (Гарантированный брак Quality < 7.0)
        cv2.imwrite(os.path.join(output_path, f"fail_blur_{filename}"), distorter.apply_blur(img))
        # Темнота (Гарантированный брак Photometry)
        cv2.imwrite(os.path.join(output_path, f"fail_dark_{filename}"), distorter.apply_low_light(img))


if __name__ == "__main__":
    import shutil
    if os.path.exists("data/benchmark/distorted"):
        shutil.rmtree("data/benchmark/distorted")
    os.makedirs("data/benchmark/distorted")
    
    prepare_benchmark_data("data/benchmark/source", "data/benchmark/distorted")