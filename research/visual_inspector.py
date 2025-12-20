import cv2
import numpy as np
import os
import pandas as pd
import random

from src.geometry import FaceGeometryController
from src.quality import FaceQualityController, check_photometry
from src.config import ICAOThresholds

class DiverseInspector:
    def __init__(self):
        self.geo = FaceGeometryController()
        self.quality = FaceQualityController()

    def create_card(self, filename, folder):
        path = os.path.join(folder, filename)
        raw = cv2.imread(path)
        if raw is None: return None
        
        img = cv2.resize(raw, (512, 512), interpolation=cv2.INTER_CUBIC)
        h, w = 512, 512
        
        # Прогон
        photo_ok, photo_msg = check_photometry(raw)
        res = self.geo.analyze(raw)
        
        color, status, desc = (0, 255, 0), "COMPLIANT", "ICAO OK"

        if not photo_ok:
            status, color, desc = "REJECTED", (0, 0, 255), f"LIGHT: {photo_msg}"
        elif res.get("error"):
            status, color, desc = "REJECTED", (0, 0, 255), f"FACE: {res['error']}"
        else:
            angles, ear = res["angles"], res["ear"]
            lms = res["landmarks"]
            
            # Рисуем аннотации
            nose = lms.landmark[1]
            cx, cy = int(nose.x*w), int(nose.y*h)
            cv2.line(img, (cx, cy), (int(cx + 80*np.cos(np.radians(angles['yaw']))), cy), (0, 0, 255), 3)
            cv2.line(img, (cx, cy), (cx, int(cy - 80*np.cos(np.radians(angles['pitch'])))), (0, 255, 0), 3)
            # Точки на глазах
            for idx in [33, 133, 362, 263]:
                pt = lms.landmark[idx]
                cv2.circle(img, (int(pt.x*w), int(pt.y*h)), 4, (255, 255, 0), -1)

            if abs(angles['yaw']) > ICAOThresholds.YAW_MAX or abs(angles['pitch']) > ICAOThresholds.PITCH_MAX:
                status, color, desc = "REJECTED", (0,0,255), f"ANGLE Y:{angles['yaw']:.1f} P:{angles['pitch']:.1f}"
            elif ear < 0.15:
                status, color, desc = "REJECTED", (0,0,255), f"EYES CLOSED ({ear:.2f})"
            else:
                score = self.quality.get_quality_score(raw, lms)
                if score < ICAOThresholds.MIN_QUALITY_SCORE:
                    status, color, desc = "REJECTED", (0,0,255), f"LOW QUALITY: {score:.1f}"
                else:
                    desc = f"SCORE: {score:.1f}"

        # Плашки
        cv2.rectangle(img, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.putText(img, status, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
        cv2.rectangle(img, (0, h-45), (w, h), (0, 0, 0), -1)
        cv2.putText(img, desc, (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.rectangle(img, (0, 0), (w, h), (80, 80, 80), 2)
        return img

def main():
    results_dir = "docs/results"
    csv_path = os.path.join(results_dir, "benchmark_final.csv")
    dist_dir = "data/benchmark/distorted"
    rej_dir = "data/benchmark/rejected_by_cleaner"
    
    if not os.path.exists(csv_path):
        print("CSV не найден.")
        return
        
    df = pd.read_csv(csv_path)
    inspector = DiverseInspector()
    
    # Состав дашборда
    cases = ['COMPLIANT', 'BAD_ANGLE', 'BLURRY', 'TOO_DARK', 'REAL_REJECT']
    selected_cards = []

    for case in cases:
        filename, folder = None, dist_dir
        
        if case == 'COMPLIANT':
            matches = df[df['pred'] == 1]
            if not matches.empty:
                filename = matches.sample(n=1).iloc[0]['filename'] 
        
        elif case == 'REAL_REJECT':
            # Берем случайный файл из тех, что отсеяли вручную 
            if os.path.exists(rej_dir):
                all_rej = [f for f in os.listdir(rej_dir) if f.endswith(('.jpg', '.png'))]
                if all_rej:
                    filename = random.choice(all_rej) 
                    folder = rej_dir
        
        else:
            # Ищем в синтетическом браке
            matches = df[df['reason'] == case]
            if not matches.empty:
                filename = matches.sample(n=1).iloc[0]['filename']

        if filename:
            print(f"Генерация кейса {case}: {filename}")
            card = inspector.create_card(filename, folder)
            if card is not None: selected_cards.append(card)

    if selected_cards:
        dashboard = np.hstack(selected_cards)
        out_path = os.path.join(results_dir, "diverse_dashboard.png")
        cv2.imwrite(out_path, dashboard)
        print(f"\nДашборд: {out_path}")

if __name__ == "__main__":
    main()