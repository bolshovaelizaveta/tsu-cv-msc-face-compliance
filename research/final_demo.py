import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import os
import sys
import time

sys.path.append(os.getcwd())
from src.geometry import FaceGeometryController
from src.config import ICAOThresholds

class ReaderIDPipeline:
    """Пайплайн проверки фото для читательского билета РГБ."""
    
    def __init__(self, 
                 matting_model="models/modnet_photographic_portrait_matting.onnx",
                 face_model="yolov8n-face.pt"):
        
        # Поиск моделей
        if not os.path.exists(face_model): face_model = os.path.join("models", face_model)
        if not os.path.exists(matting_model):
            print(f"ОШИБКА: Модель {matting_model} не найдена.")
            sys.exit()

        print("Инициализация системы валидации...")
        
        # Загрузка нейросетей
        self.detector = YOLO(face_model)
        self.matting_session = ort.InferenceSession(matting_model, providers=['CPUExecutionProvider'])
        
        # Контроллер геометрии
        self.geo_controller = FaceGeometryController()

    def get_smart_crop(self, image):
        """Кроп под документы"""
        res = self.detector(image, verbose=False)[0]
        if not res.boxes: return None
        
        # Берем самое крупное лицо
        box = res.boxes.data[0].cpu().numpy()
        fx1, fy1, fx2, fy2 = box[:4]
        fw, fh = fx2 - fx1, fy2 - fy1
        
        # Коэффициент 2.3 для фото (лицо + плечи)
        side = int(max(fw, fh) * 2.3)
        cx, cy = int(fx1 + fw/2), int(fy1 + fh/2)
        
        # Сдвиг макушки
        cy_adj = cy + int(side * 0.08)

        nx1, ny1 = cx - side // 2, cy_adj - side // 2
        nx2, ny2 = nx1 + side, ny1 + side
        
        ih, iw = image.shape[:2]
        pad_t = max(0, -ny1); pad_b = max(0, ny2 - ih)
        pad_l = max(0, -nx1); pad_r = max(0, nx2 - iw)

        img_p = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(255,255,255))
        return img_p[ny1+pad_t : ny2+pad_t, nx1+pad_l : nx2+pad_l]

    def apply_matting_clean(self, crop):
        """Удаление фона"""
        h, w = crop.shape[:2]
        input_img = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_tensor = (input_img.astype(np.float32) - 127.5) / 128.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))[np.newaxis, :, :, :]

        matte = self.matting_session.run(None, {self.matting_session.get_inputs()[0].name: input_tensor})[0][0, 0, :, :]
        matte = cv2.resize(matte, (w, h))

        # Удаление соседей по кадру 
        matte_uint8 = (matte * 255).astype(np.uint8)
        _, bin_mask = cv2.threshold(matte_uint8, 10, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        target_label = labels[h//2, w//2]
        if target_label == 0 and num_labels > 1:
            target_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        if target_label > 0:
            matte = matte * (labels == target_label)

        # Обработка маски
        matte = np.clip((matte - 0.2) / 0.8, 0.0, 1.0)
        matte_3ch = np.stack([matte] * 3, axis=-1)

        bg = np.full(crop.shape, 255, dtype=np.uint8)
        return (crop.astype(float) * matte_3ch + bg.astype(float) * (1.0 - matte_3ch)).astype(np.uint8)

    def process_file(self, path):
        img = cv2.imread(path)
        if img is None: return

        # 1. Формирование изображения
        crop = self.get_smart_crop(img)
        if crop is None:
            print("Лицо не обнаружено")
            return
        
        final_img = self.apply_matting_clean(crop)
        final_img = cv2.resize(final_img, (600, 600), interpolation=cv2.INTER_LANCZOS4)

        # 2. Анализ геометрии 
        res = self.geo_controller.analyze(final_img)
        reasons = []

        if not res.get("error"):
            a = res['angles']
            # Проверки по сбалансированным порогам
            if abs(a['yaw']) > ICAOThresholds.YAW_MAX: reasons.append(f"Yaw:{a['yaw']:.0f}")
            if abs(a['pitch']) > ICAOThresholds.PITCH_MAX: reasons.append(f"Pitch:{a['pitch']:.0f}")
            if abs(a['roll']) > ICAOThresholds.ROLL_MAX: reasons.append(f"Roll:{a['roll']:.0f}")
            if res['ear'] < ICAOThresholds.EYE_OPEN_MIN: reasons.append("Eyes")
            if res['mar'] > ICAOThresholds.MOUTH_CLOSED_MAX: reasons.append("Mouth")
            
            fh_pct = res['face_height'] * 100
            if fh_pct < (ICAOThresholds.FACE_HEIGHT_MIN*100) or fh_pct > (ICAOThresholds.FACE_HEIGHT_MAX*100):
                reasons.append(f"Scale:{fh_pct:.0f}%")

        # 3. Вердикт и отрисовка
        status = "VALID PHOTO" if not reasons else "INVALID PHOTO"
        color = (0, 255, 0) if not reasons else (0, 0, 255)

        # Статус
        cv2.putText(final_img, status, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
        if reasons:
            cv2.putText(final_img, " | ".join(reasons), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Метрики внизу (техническая информация)
        info = f"Y:{res['angles']['yaw']:.1f} P:{res['angles']['pitch']:.1f} R:{res['angles']['roll']:.1f} | Scale:{res['face_height']*100:.0f}%"
        cv2.putText(final_img, info, (10, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,100,100), 1)

        # Показ двух окон
        orig_disp = cv2.resize(img, (800, int(img.shape[0]*(800/img.shape[1]))), interpolation=cv2.INTER_LANCZOS4)
        cv2.imshow("1. Original Image", orig_disp)
        cv2.imshow("2. Final ID Photo", final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_photo = "data/test_samples/compliant_1.jpg" 
    pipeline = ReaderIDPipeline()
    pipeline.process_file(test_photo)