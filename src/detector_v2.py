import cv2
import numpy as np
from ultralytics import YOLO
import os

class FaceDetectorV2:
    def __init__(self, model_path="models/yolov8n-face.pt"):
        if not os.path.exists(model_path):
            print(f"ERROR: {model_path} not found")
        self.detector = YOLO(model_path)

    def get_passport_crop(self, image, side_coeff=2.0):
        """
        Вырезает квадратный кроп лица с симметричными полями.
        """
        results = self.detector(image, verbose=False)[0]
        if not results.boxes:
            return None, None
        
        # Выбираем самое крупное лицо
        boxes = results.boxes.data.cpu().numpy()
        main_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        
        fx1, fy1, fx2, fy2 = main_box[:4]
        fw, fh = fx2 - fx1, fy2 - fy1
        
        side = int(max(fw, fh) * side_coeff)
        cx, cy = int(fx1 + fw/2), int(fy1 + fh/2)
        cy_adj = cy + int(side * 0.05) 

        nx1, ny1 = cx - side // 2, cy_adj - side // 2
        nx2, ny2 = nx1 + side, ny1 + side
        
        ih, iw = image.shape[:2]
        
        pad_t, pad_b = max(0, -ny1), max(0, ny2 - ih)
        pad_l, pad_r = max(0, -nx1), max(0, nx2 - iw)

        # Если кроп выходит за границы (слишком крупное фото)
        if (pad_t + pad_b + pad_l + pad_r) > (side * 0.2):
            nx1_c, ny1_c = max(0, nx1), max(0, ny1)
            nx2_c, ny2_c = min(iw, nx2), min(ih, ny2)
            crop = image[ny1_c:ny2_c, nx1_c:nx2_c]
            
            h_c, w_c = crop.shape[:2]
            s_c = max(h_c, w_c)
            
            t = (s_c - h_c) // 2
            b = s_c - h_c - t
            l = (s_c - w_c) // 2
            r = s_c - w_c - l
            
            final_crop = cv2.copyMakeBorder(crop, t, b, l, r, cv2.BORDER_CONSTANT, value=(255,255,255))
            return final_crop, (s_c//2, s_c//2)

        # Обычный случай
        img_p = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(255,255,255))
        crop = img_p[ny1+pad_t : ny2+pad_t, nx1+pad_l : nx2+pad_l]
        return crop, (side//2, side//2)