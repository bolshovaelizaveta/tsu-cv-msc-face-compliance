import cv2
import time
import numpy as np
from src.geometry import FaceGeometryController
from src.quality import FaceQualityController, check_photometry
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

        # Проверка освещения (Фотометрия)
        photo_ok, photo_msg = check_photometry(frame)
        
        # Геометрия и глаза
        geo_result = geo_processor.analyze(frame)
        
        status_text = "CHECKING..."
        color = (0, 255, 255) 
        reasons = []
        
        # Считаем FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Логика принятия решения
        if not photo_ok:
            status_text = "REJECTED"
            color = (0, 0, 255)
            reasons.append(photo_msg) # TOO_DARK / TOO_BRIGHT
            
        elif geo_result.get("error"):
            status_text = "REJECTED"
            color = (0, 0, 255)
            reasons.append(geo_result["error"]) # NO_FACE / MULTIPLE_FACES
            
        else:
            # Лицо найдено, проверяем углы и глаза
            angles = geo_result["angles"]
            ear = geo_result["ear"]
            landmarks = geo_result["landmarks"]
            
            # Проверка углов
            is_geo_ok = (abs(angles['yaw']) <= ICAOThresholds.YAW_MAX and 
                         abs(angles['pitch']) <= ICAOThresholds.PITCH_MAX and 
                         abs(angles['roll']) <= ICAOThresholds.ROLL_MAX)
            
            # Проверка глаз 
            is_eyes_ok = ear > 0.15
            
            if not is_geo_ok:
                reasons.append("BAD ANGLE")
            if not is_eyes_ok:
                reasons.append("EYES CLOSED")

            # Качество (MagFace)
            quality_score = 0.0
            if is_geo_ok and is_eyes_ok:
                quality_score = quality_processor.get_quality_score(frame, landmarks)
                is_quality_ok = quality_score >= ICAOThresholds.MIN_QUALITY_SCORE
                
                if not is_quality_ok:
                    reasons.append(f"LOW QUALITY ({quality_score:.1f})")
                else:
                    status_text = "ICAO COMPLIANT"
                    color = (0, 255, 0) 
            else:
                status_text = "REJECTED"
                color = (0, 0, 255)

            # Отрисовка
            # Углы
            info_line1 = f"Y:{angles['yaw']:.1f} P:{angles['pitch']:.1f} R:{angles['roll']:.1f}"
            cv2.putText(frame, info_line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Глаза и Качество
            info_line2 = f"EAR:{ear:.2f} Q:{quality_score:.2f}"
            cv2.putText(frame, info_line2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Итоговый статус
        cv2.putText(frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Причины отказа
        if reasons:
            cv2.putText(frame, ", ".join(reasons), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1]-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow('TSU Face Compliance Demo', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()