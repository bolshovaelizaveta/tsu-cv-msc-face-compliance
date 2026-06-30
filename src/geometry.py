import cv2
import mediapipe as mp
import numpy as np
import math
from typing import Dict, Any

class FaceGeometryController:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # Детектор рук MediaPipe 
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

        # 3D координаты стандартного лица
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye
            (225.0, 170.0, -135.0),      # Right eye
            (-150.0, -150.0, -125.0),    # Left mouth
            (150.0, -150.0, -125.0)      # Right mouth
        ], dtype=np.float64)

        self.keypoints_indices = [1, 152, 33, 263, 61, 291]
        
        # Индексы для EAR (Глаза)
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]
        self.right_eye_indices = [362, 385, 387, 263, 373, 380]
        
        # Индексы для MAR (Рот)
        self.mouth_indices = [61, 37, 267, 291, 314, 84]

    def _calculate_ratio(self, landmarks, w, h, indices):
        pts = []
        for idx in indices:
            lm = landmarks.landmark[idx]
            pts.append(np.array([lm.x * w, lm.y * h]))
        
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        hor = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * hor)

    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        h, w, _ = image.shape
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Проверяем наличие рук в кадре 
        hands_results = self.hands_detector.process(img_rgb)
        hand_detected = True if hands_results.multi_hand_landmarks else False
        
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return {"error": "NO_FACE", "hand_detected": hand_detected}

        if len(results.multi_face_landmarks) > 1:
            return {"error": "MULTIPLE_FACES", "hand_detected": hand_detected}

        landmarks = results.multi_face_landmarks[0]
        
        # 1. Расчет углов Эйлера (PnP)
        image_points = np.array([[landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h] 
                                 for idx in self.keypoints_indices], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        
        success, rot_vec, _ = cv2.solvePnP(self.model_points, image_points, camera_matrix, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)

        yaw, pitch, roll = 0.0, 0.0, 0.0
        if success:
            rmat, _ = cv2.Rodrigues(rot_vec)
            sy = math.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
            if sy > 1e-6:
                pitch = math.degrees(math.atan2(rmat[2,1], rmat[2,2]))
                yaw = math.degrees(math.atan2(-rmat[2,0], sy))
                roll = math.degrees(math.atan2(rmat[1,0], rmat[0,0]))
            
            if pitch < -90: pitch = -(pitch + 180)
            elif pitch > 90: pitch = 180 - pitch
            yaw, roll = -yaw, -roll

        # 2. Расчет открытости глаз (EAR) и рта (MAR)
        ear_l = self._calculate_ratio(landmarks, w, h, self.left_eye_indices)
        ear_r = self._calculate_ratio(landmarks, w, h, self.right_eye_indices)
        avg_ear = (ear_l + ear_r) / 2.0
        mar = self._calculate_ratio(landmarks, w, h, self.mouth_indices)

        # 3. Масштаб лица и метрики ICAO 
        face_height_pct = abs(landmarks.landmark[152].y - landmarks.landmark[10].y)
        
        # Точки 454 (правая скула) и 234 (левая скула)
        face_width_pct = abs(landmarks.landmark[454].x - landmarks.landmark[234].x)
        
        # Точки 33 и 263 - уголки глаз. Считаем среднюю высоту Y. 
        # (1.0 - y) дает расстояние от нижнего края изображения.
        eye_y_avg = (landmarks.landmark[33].y + landmarks.landmark[263].y) / 2.0
        eye_dist_from_bottom = 1.0 - eye_y_avg

        # 4. Оценка направления взгляда
        def get_gaze_ratio(eye_left_idx, eye_right_idx, iris_idx):
            eye_left = np.array([landmarks.landmark[eye_left_idx].x * w, landmarks.landmark[eye_left_idx].y * h])
            eye_right = np.array([landmarks.landmark[eye_right_idx].x * w, landmarks.landmark[eye_right_idx].y * h])
            iris = np.array([landmarks.landmark[iris_idx].x * w, landmarks.landmark[iris_idx].y * h])
            return np.linalg.norm(iris - eye_left) / (np.linalg.norm(iris - eye_right) + 1e-6)

        gaze_score_x = (get_gaze_ratio(33, 133, 468) + get_gaze_ratio(362, 263, 473)) / 2.0
        
        def get_vertical_gaze_ratio(upper_idx, lower_idx, iris_idx):
            return abs(landmarks.landmark[iris_idx].y - landmarks.landmark[upper_idx].y) / (abs(landmarks.landmark[lower_idx].y - landmarks.landmark[upper_idx].y) + 1e-6)

        gaze_score_y = (get_vertical_gaze_ratio(159, 145, 468) + get_vertical_gaze_ratio(386, 374, 473)) / 2.0

        return {
            "angles": {"yaw": yaw, "pitch": pitch, "roll": roll},
            "ear": avg_ear,
            "mar": mar,
            "face_height": face_height_pct,
            "face_width": face_width_pct,
            "eye_dist_bottom": eye_dist_from_bottom,
            "gaze_score": gaze_score_x,
            "gaze_score_y": gaze_score_y, 
            "hand_detected": hand_detected, 
            "landmarks": landmarks
        }