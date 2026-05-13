import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import os
import sys

sys.path.append(os.getcwd())
from src.geometry import FaceGeometryController
from src.config import ICAOThresholds

class StrictICAOPipeline:
    def __init__(self, 
                 seg_model="models/selfie_multiclass.tflite",
                 face_model="yolov8n-face.pt"):
        
        if not os.path.exists(face_model): face_model = os.path.join("models", face_model)
        if not os.path.exists(face_model) or not os.path.exists(seg_model):
            print("ОШИБКА: Файлы моделей не найдены.")
            sys.exit()

        self.detector = YOLO(face_model)
        
        base_options = python.BaseOptions(model_asset_path=seg_model)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)
        self.geo_controller = FaceGeometryController()

    def get_square_crop(self, image):
        """Кроп лица"""
        res = self.detector(image, verbose=False)[0]
        if not res.boxes: return None
        
        boxes = res.boxes.data.cpu().numpy()
        main_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        fx1, fy1, fx2, fy2 = main_box[:4]
        fw, fh = fx2 - fx1, fy2 - fy1
        
        side = int(max(fw, fh) * 2.5)
        cx, cy = int(fx1 + fw/2), int(fy1 + fh/2)
        cy_adj = cy + int(side * 0.12)

        nx1, ny1 = cx - side // 2, cy_adj - side // 2
        nx2, ny2 = nx1 + side, ny1 + side

        ih, iw = image.shape[:2]
        pad_t, pad_b = max(0, -ny1), max(0, ny2 - ih)
        pad_l, pad_r = max(0, -nx1), max(0, nx2 - iw)

        img_p = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(255,255,255))
        return img_p[ny1+pad_t : ny2+pad_t, nx1+pad_l : nx2+pad_l]

    def remove_bg(self, crop):
        """Удаление фона"""
        h, w = crop.shape[:2]
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        res = self.segmenter.segment(mp_image)
        mask = np.where(res.category_mask.numpy_view() > 0, 255, 0).astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Изоляция 
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            center = np.array([w / 2, h / 2])
            main_label = 1 + np.argmin([np.linalg.norm(c - center) for c in centroids[1:]])
            mask = np.where(labels == main_label, 255, 0).astype(np.uint8)

        # Сглаживание краев
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        alpha = mask.astype(float) / 255.0
        alpha = np.stack([alpha] * 3, axis=-1)
        bg = np.full(crop.shape, 255, dtype=np.uint8)
        
        return (crop.astype(float) * alpha + bg.astype(float) * (1.0 - alpha)).astype(np.uint8)

    def process_file(self, path):
        img = cv2.imread(path)
        if img is None: return

        # 1. Вырезаем (с макушкой) и удаляем фон
        crop = self.get_square_crop(img)
        if crop is None: return
        final_img = self.remove_bg(crop)
        final_img = cv2.resize(final_img, (600, 600), interpolation=cv2.INTER_LANCZOS4)

        # 2. Геометрия
        geo_result = self.geo_controller.analyze(final_img)

        # 3. Валидация
        status = "COMPLIANT"
        color = (0, 255, 0)
        reasons = []

        if geo_result.get("error"):
            status = "REJECTED"
            color = (0, 0, 255)
            reasons.append(geo_result["error"])
        else:
            angles = geo_result["angles"]
            ear = geo_result["ear"]
            mar = geo_result["mar"]
            scale = geo_result["face_height"] * 100

            if abs(angles['yaw']) > ICAOThresholds.YAW_MAX: reasons.append(f"Yaw: {angles['yaw']:.1f}")
            if abs(angles['pitch']) > ICAOThresholds.PITCH_MAX: reasons.append(f"Pitch: {angles['pitch']:.1f}")
            if abs(angles['roll']) > ICAOThresholds.ROLL_MAX: reasons.append(f"Roll: {angles['roll']:.1f}")
            if ear < ICAOThresholds.EYE_OPEN_MIN: reasons.append(f"Eyes: {ear:.2f}")
            if mar > ICAOThresholds.MOUTH_CLOSED_MAX: reasons.append(f"Mouth: {mar:.2f}")
            if scale < (ICAOThresholds.FACE_HEIGHT_MIN*100) or scale > (ICAOThresholds.FACE_HEIGHT_MAX*100):
                reasons.append(f"Scale: {scale:.0f}%")

            if reasons:
                status = "REJECTED"
                color = (0, 0, 255)

            # Отрисовка
            cv2.putText(final_img, f"Yaw: {angles['yaw']:.1f}", (10, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.putText(final_img, f"Pitch: {angles['pitch']:.1f}", (10, 545), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.putText(final_img, f"Roll: {angles['roll']:.1f}", (10, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.putText(final_img, f"EAR: {ear:.2f}", (450, 545), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.putText(final_img, f"MAR: {mar:.2f}", (450, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        cv2.putText(final_img, status, (15, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
        if reasons:
            cv2.putText(final_img, " | ".join(reasons), (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        orig_disp = cv2.resize(img, (800, int(img.shape[0]*(800/img.shape[1]))), interpolation=cv2.INTER_LANCZOS4)
        
        cv2.imshow("1. Original", orig_disp)
        cv2.imshow("2. Geometry Validation", final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_photo = "data/test_samples/compliant_1.jpg" 
    pipeline = StrictICAOPipeline()
    pipeline.process_file(test_photo)

