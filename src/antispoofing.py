import os
import cv2
import numpy as np
import onnxruntime as ort

class FaceAntiSpoofingController:
    """
    Контроллер для защиты от презентационных атак (MiniFASNet-V2).
    """
    def __init__(self, model_path="models/antispoofing.onnx"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель антиспуфинга не найдена: {model_path}")
        
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

def analyze(self, crop_27x: np.ndarray) -> dict:
        # 1. Убираем шум (зернистость) камеры
        img_denoised = cv2.medianBlur(crop_27x, 3) 
        
        # 2. Resize и нормализация
        img_resized = cv2.resize(img_denoised, (80, 80), interpolation=cv2.INTER_LINEAR)
        img_float = img_resized.astype(np.float32) / 255.0
        
        input_tensor = np.transpose(img_float, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0) 
        
        outputs = self.session.run(None, {self.input_name: input_tensor})
        probs = self._softmax(outputs[0])[0]
        
        liveness_score = float(probs[0])
        print(f"DEBUG FAS | Live: {probs[0]:.3f} | Print: {probs[1]:.3f} | Replay: {probs[2]:.3f}")
        
        is_real = liveness_score >= 0.10
        
        attack_type = "none"
        if not is_real:
            attack_type = "print_attack" if probs[1] > probs[2] else "replay_attack"
                
        return {
            "is_real": is_real,
            "liveness_score": liveness_score,
            "attack_type": attack_type
        }