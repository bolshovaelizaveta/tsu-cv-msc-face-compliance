import cv2
import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.segmentation_v2 import FaceSegmentationV2

def generate_figure():
    images_dir = os.path.join(PROJECT_ROOT, "data", "matting_test", "images")
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"Ошибка: В папке {images_dir} не найдено изображений!")
        return

    target_file = image_files[4]
    img_path = os.path.join(images_dir, target_file)
    print(f"[*] Используем файл: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        print(f"Ошибка: Не удалось прочитать файл {img_path}. Проверьте целостность файла.")
        return

    h, w = img.shape[:2]
    
    model_path = os.path.join(PROJECT_ROOT, "models", "modnet_photographic_portrait_matting.onnx")
    seg = FaceSegmentationV2(model_path)
    
    # 1. Получаем результат 
    result_img = seg.remove_background(img, (w // 2, h // 2))
    
    # 2. Достаем маску 
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path)
    
    # Препроцессинг
    input_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_tensor = (input_img.astype(np.float32) - 127.5) / 128.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))[np.newaxis, :, :, :]
    
    # Инференс
    matte = sess.run(None, {sess.get_inputs()[0].name: input_tensor})[0][0, 0, :, :]
    matte = cv2.resize(matte, (w, h))
    
    # Визуализация маски
    matte_vis = (matte * 255).astype(np.uint8)
    matte_vis_3ch = cv2.cvtColor(matte_vis, cv2.COLOR_GRAY2BGR)

    # Склеиваем в один ряд: Оригинал | Маска | Результат
    divider = np.ones((h, 10, 3), dtype=np.uint8) * 255
    canvas = np.hstack([img, divider, matte_vis_3ch, divider, result_img])
    
    output_path = os.path.join(PROJECT_ROOT, "figure_6_stages.png")
    cv2.imwrite(output_path, canvas)
    print(f"[*] Рисунок успешно сохранен в корень проекта: {output_path}")

if __name__ == "__main__":
    generate_figure()