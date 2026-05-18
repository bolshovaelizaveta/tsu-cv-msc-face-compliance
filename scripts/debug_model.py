import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys

sys.path.append(os.getcwd())

def debug_occlusion_model(image_path, model_path="models/yolov8m-occlusion.pt"):
    if not os.path.exists(image_path):
        print(f"Файл {image_path} не найден")
        return

    model = YOLO(model_path)
    img = cv2.imread(image_path)
    
    from src.detector_v2 import FaceDetectorV2
    detector = FaceDetectorV2()
    
    crop, _ = detector.get_passport_crop(img, side_coeff=1.2)
    
    if crop is None:
        print("YOLO-Face не нашла лицо на снимке")
        return

    # Инференс классификатора
    results = model(crop, verbose=False)[0]
    probs = results.probs.data.cpu().numpy()
    names = model.names

    print(f"\n--- Результаты для {os.path.basename(image_path)} ---")
    for i, prob in enumerate(probs):
        print(f"Класс: {names[i]:<15} | Вероятность: {prob:.4f}")
    
    top1_idx = results.probs.top1
    label = f"Result: {names[top1_idx]} ({probs[top1_idx]:.4f})"
    
    cv2.putText(crop, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("Result", crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    debug_occlusion_model("data/test_samples/clear_glasses.jpg")