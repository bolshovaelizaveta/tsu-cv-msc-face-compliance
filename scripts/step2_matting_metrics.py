import os
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TEST_DIR = os.path.join(PROJECT_ROOT, "data", "matting_test")
IMAGES_DIR = os.path.join(TEST_DIR, "images")
MASKS_DIR = os.path.join(TEST_DIR, "masks")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "modnet_photographic_portrait_matting.onnx")

def calculate_metrics():
    print(f"Инициализация MODNet из {MODEL_PATH}...")
    
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 8
    session = ort.InferenceSession(MODEL_PATH, sess_options)
    input_name = session.get_inputs()[0].name

    files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files.sort()

    total_sad = 0.0
    total_mse = 0.0
    total_iou = 0.0
    valid_images = 0

    print("Запуск расчета БИНАРНЫХ метрик (SAD, MSE, IoU)...")

    for filename in tqdm(files, desc="Оценка сегментации"):
        img_path = os.path.join(IMAGES_DIR, filename)
        mask_name = os.path.splitext(filename)[0] + ".png"
        mask_path = os.path.join(MASKS_DIR, mask_name)

        if not os.path.exists(mask_path):
            continue

        # 1. Готовим GT маску 
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Все, что больше 127, становится 255 (четкий контур)
        _, gt_bin = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
        gt_alpha = gt_bin.astype(np.float32) / 255.0

        # 2. Инференс MODNet
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        input_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_tensor = (input_img.astype(np.float32) - 127.5) / 128.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))[np.newaxis, :, :, :]

        pred_matte = session.run(None, {input_name: input_tensor})[0][0, 0, :, :]
        pred_alpha_soft = cv2.resize(pred_matte, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Банаризация предсказания 
        pred_alpha = (pred_alpha_soft > 0.5).astype(np.float32)

        # 3. Расчет метрик
        sad = np.sum(np.abs(pred_alpha - gt_alpha)) / 1000.0
        mse = np.mean(np.square(pred_alpha - gt_alpha))
        
        # Расчет IoU (Intersection over Union)
        intersection = np.logical_and(pred_alpha, gt_alpha)
        union = np.logical_or(pred_alpha, gt_alpha)
        iou = np.sum(intersection) / np.sum(union)

        total_sad += sad
        total_mse += mse
        total_iou += iou
        valid_images += 1

    mean_sad = total_sad / valid_images
    mean_mse = total_mse / valid_images
    mean_iou = total_iou / valid_images

    print("\n" + "="*50)
    print("Результаты оценки сегментации")
    print("="*50)
    print(f"Количество изображений (N): {valid_images}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean SAD: {mean_sad:.4f}")
    print(f"Mean MSE: {mean_mse:.6f}")
    print("="*50)

if __name__ == "__main__":
    calculate_metrics()