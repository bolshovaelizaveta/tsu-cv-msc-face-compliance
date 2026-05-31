import os
import math
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolov8m-occlusion.pt")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "yolo_occlusion_dataset", "test")
OUTPUT_ERRORS_PATH = os.path.join(PROJECT_ROOT, "misclassified_grid.png")

EXPECTED_CLASSES = ["clean", "clear_glasses", "headwear", "occluded"]

def analyze_errors():
    print(f"Инициализация модели из {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    errors = [] 
    
    print("Поиск неверно классифицированных изображений...")
    for cls_name in EXPECTED_CLASSES:
        cls_dir = os.path.join(TEST_DIR, cls_name)
        if not os.path.exists(cls_dir):
            continue
            
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(images, desc=f"Проверка {cls_name}"):
            img_path = os.path.join(cls_dir, img_name)
            
            results = model(img_path, verbose=False)[0]
            pred_idx = results.probs.top1
            pred_class = model.names[pred_idx]
            conf = float(results.probs.top1conf)
            
            if pred_class != cls_name:
                errors.append({
                    'path': img_path,
                    'true': cls_name,
                    'pred': pred_class,
                    'conf': conf
                })
                
    print(f"\n[*] Найдено ошибок: {len(errors)}")
    if len(errors) == 0:
        print("Ошибок нет!")
        return
        
    print("Отрисовка сетки с ошибками...")
    # По 5 картинок в ряд
    cols = 5
    rows = math.ceil(len(errors) / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    if len(errors) > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        if i < len(errors):
            err = errors[i]
            img = Image.open(err['path'])
            ax.imshow(img)
            
            # T - True (Истина), P - Predicted (Предсказание)
            title = f"T: {err['true']}\nP: {err['pred']}\nConf: {err['conf']:.2f}"
            ax.set_title(title, fontsize=10, color='darkred')
            ax.axis('off')
        else:
            ax.axis('off') 
            
    plt.tight_layout()
    plt.savefig(OUTPUT_ERRORS_PATH, dpi=300, bbox_inches='tight')
    print(f"Сетка ошибок сохранена в корень проекта: {OUTPUT_ERRORS_PATH}")

if __name__ == "__main__":
    analyze_errors()