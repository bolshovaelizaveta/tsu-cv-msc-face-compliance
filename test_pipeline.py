import cv2
import time
import numpy as np
from src.geometry import FaceGeometryController
from src.quality import FaceQualityController
from src.config import ICAOThresholds

def main():
    print("Запуск WEBCAM")
    print("Загрузка геометрии (MediaPipe)...")
    geo_processor = FaceGeometryController()
    print("Загрузка нейросети (MagFace PyTorch)...") 
    try:
        quality_processor = FaceQualityController()
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    print("Открытие камеры...")
    cap = cv2.VideoCapture(0) 
    
    # Установка разрешения
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ОШИБКА: Не удалось открыть камеру.")
        return

    print("Готово! Нажми 'q' для выхода.")
    
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
            
        # Зеркалим для удобства
        frame = cv2.flip(frame, 1)

        # Геометрия
        angles, landmarks = geo_processor.get_head_pose(frame)
        
        status_text = "NO FACE"
        color = (0, 0, 255) 
        quality = 0.0
        
        # Считаем FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        if angles and landmarks:
            # Качество (MagFace)
            quality = quality_processor.get_quality_score(frame, landmarks)
            
            # Проверка порогов
            is_geo_ok = (abs(angles['yaw']) <= ICAOThresholds.YAW_MAX and 
                         abs(angles['pitch']) <= ICAOThresholds.PITCH_MAX and 
                         abs(angles['roll']) <= ICAOThresholds.ROLL_MAX)
            
            is_quality_ok = quality >= ICAOThresholds.MIN_QUALITY_SCORE
            
            if is_geo_ok and is_quality_ok:
                status_text = "ICAO COMPLIANT"
                color = (0, 255, 0) 
            else:
                status_text = "REJECTED"
                color = (0, 0, 255) 

            # Вывод метрик на экран
            # Y/P/R - углы, Q - качество (MagFace Score)
            info_line1 = f"Yaw:{angles['yaw']:.1f} P:{angles['pitch']:.1f} R:{angles['roll']:.1f}"
            info_line2 = f"Quality: {quality:.2f} (Min: {ICAOThresholds.MIN_QUALITY_SCORE})"
            
            cv2.putText(frame, info_line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, info_line2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Причины отказа
            reasons = []
            if not is_geo_ok: reasons.append("BAD ANGLE")
            if not is_quality_ok: reasons.append("LOW QUALITY")
            
            cv2.putText(frame, f"{status_text}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if reasons:
                cv2.putText(frame, " ".join(reasons), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1]-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow('TSU Face Compliance Demo', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()