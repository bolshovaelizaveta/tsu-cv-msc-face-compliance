import os
import sys
import math
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

# Импорты твоих модулей
from src.config import ICAOThresholds
from src.detector_v2 import FaceDetectorV2
from src.geometry import FaceGeometryController
from src.occlusion import FaceOcclusionController, check_glare
from src.quality import FaceQualityController, check_photometry

TEST_DIR = os.path.join(PROJECT_ROOT, "data", "system_test")
GOOD_DIR = os.path.join(TEST_DIR, "good_photos")
BAD_DIR = os.path.join(TEST_DIR, "bad_photos")

OUTPUT_FRR_IMG = os.path.join(PROJECT_ROOT, "false_rejections.png")
OUTPUT_FAR_IMG = os.path.join(PROJECT_ROOT, "false_acceptances.png")


class PipelineValidator:
    def __init__(self):
        print("Загрузка нейросетей...")
        self.detector = FaceDetectorV2(os.path.join(PROJECT_ROOT, "models", "yolov8n-face.pt"))
        self.geometry = FaceGeometryController()
        self.occlusion = FaceOcclusionController(os.path.join(PROJECT_ROOT, "models", "yolov8m-occlusion.pt"))
        self.quality = FaceQualityController() 
        self.icao = ICAOThresholds()
        
    def validate(self, img_path):
        """Возвращает (is_passed: bool, reason: str)"""
        img = cv2.imread(img_path)
        if img is None:
            return False, "CORRUPTED_IMAGE"

        # 1. Базовая фотометрия
        photo_ok, photo_reason = check_photometry(img)
        if not photo_ok:
            return False, f"PHOTOMETRY: {photo_reason}"

        # 2. Геометрия (MediaPipe)
        geom_res = self.geometry.analyze(img)
        if "error" in geom_res:
            return False, f"GEOMETRY: {geom_res['error']}"
        
        # 3. Проверка порогов ICAO
        angles = geom_res["angles"]
        if abs(angles["yaw"]) > self.icao.YAW_MAX: return False, f"YAW_FAIL ({angles['yaw']:.1f})"
        if abs(angles["pitch"]) > self.icao.PITCH_MAX: return False, f"PITCH_FAIL ({angles['pitch']:.1f})"
        if abs(angles["roll"]) > self.icao.ROLL_MAX: return False, f"ROLL_FAIL ({angles['roll']:.1f})"
        
        if geom_res["ear"] < self.icao.EYE_OPEN_MIN: return False, f"EYES_CLOSED ({geom_res['ear']:.2f})"
        if geom_res["mar"] > self.icao.MOUTH_CLOSED_MAX: return False, f"MOUTH_OPEN ({geom_res['mar']:.2f})"
        
        # Оценка взгляда
        if not (self.icao.GAZE_MIN < geom_res["gaze_score"] < self.icao.GAZE_MAX):
            return False, f"GAZE_X_FAIL ({geom_res['gaze_score']:.2f})"

        # 4. Детекция и кроп лица (YOLO-Face)
        crop, _ = self.detector.get_passport_crop(img)
        if crop is None:
            return False, "CROP_FAILED"

        # 5. Окклюзии (YOLO-cls)
        occ_res = self.occlusion.analyze(crop)
        cls_name = occ_res["class"]
        
        if cls_name in ["headwear", "occluded"]:
            return False, f"OCCLUSION: {cls_name}"
            
        if cls_name == "clear_glasses":
            has_glare = check_glare(crop, geom_res["landmarks"])
            if has_glare:
                return False, "GLASSES_GLARE"

        # 6. Качество изображения (MagFace)
        q_score = self.quality.get_quality_score(img, geom_res["landmarks"])
        if q_score < self.icao.MIN_QUALITY_SCORE:
            return False, f"LOW_QUALITY ({q_score:.1f})"

        return True, "PASSED_ALL"

def draw_error_grid(error_list, output_path, title_prefix):
    """Рисует сетку изображений с подписями ошибок"""
    if not error_list:
        return
        
    cols = 4
    rows = math.ceil(len(error_list) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    
    if len(error_list) > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
        
    for i, ax in enumerate(axes):
        if i < len(error_list):
            item = error_list[i]
            img = cv2.imread(item['path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(f"{title_prefix}\n{item['reason']}", fontsize=10, color='red')
            ax.axis('off')
        else:
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

def calculate_system_metrics():
    validator = PipelineValidator()
    
    # 1. Считаем FRR (False Rejection Rate) - хорошие фото
    good_files = [f for f in os.listdir(GOOD_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    false_rejections = []
    
    print("\nАнализ идеальных фотографий (FRR)...")
    for fname in tqdm(good_files):
        path = os.path.join(GOOD_DIR, fname)
        is_passed, reason = validator.validate(path)
        if not is_passed:
            false_rejections.append({'path': path, 'reason': reason})
            
    # 2. Считаем FAR (False Acceptance Rate) - плохие фото
    bad_files = [f for f in os.listdir(BAD_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    false_acceptances = []
    
    print("\n[*] Анализ плохих фотографий (FAR)...")
    for fname in tqdm(bad_files):
        path = os.path.join(BAD_DIR, fname)
        is_passed, reason = validator.validate(path)
        if is_passed: 
            false_acceptances.append({'path': path, 'reason': reason})

    # Расчет метрик
    total_good = len(good_files)
    total_bad = len(bad_files)
    
    frr = (len(false_rejections) / total_good) * 100 if total_good > 0 else 0
    far = (len(false_acceptances) / total_bad) * 100 if total_bad > 0 else 0
    
    print("\n" + "="*50)
    print("МЕТРИКИ (FAR / FRR)")
    print("="*50)
    print(f"Всего тестовых фото: {total_good}")
    print(f"Ложных отказов (False Rejections): {len(false_rejections)}")
    print(f"-> FRR (False Rejection Rate): {frr:.2f}%\n")
    
    print(f"Всего тестовых плохих фото: {total_bad}")
    print(f"Ложных пропусков (False Acceptances): {len(false_acceptances)}")
    print(f"-> FAR (False Acceptance Rate): {far:.2f}%")
    print("="*50)

    # Сохраняем визуализации
    if false_rejections:
        draw_error_grid(false_rejections, OUTPUT_FRR_IMG, "ОШИБКА ОТКАЗА")
        print(f"\nИзображения ложных отказов сохранены: {OUTPUT_FRR_IMG}")
        
    if false_acceptances:
        draw_error_grid(false_acceptances, OUTPUT_FAR_IMG, "ОШИБКА ПРОПУСКА")
        print(f"[*] Изображения ложных пропусков сохранены: {OUTPUT_FAR_IMG}")

if __name__ == "__main__":
    calculate_system_metrics()