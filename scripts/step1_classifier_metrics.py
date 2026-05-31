import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolov8m-occlusion.pt")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "yolo_occlusion_dataset", "test")
OUTPUT_CM_PATH = os.path.join(PROJECT_ROOT, "confusion_matrix.png")

EXPECTED_CLASSES = ["clean", "clear_glasses", "headwear", "occluded"]

def evaluate_classifier():
    print(f"Инициализация модели из {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")
    
    model = YOLO(MODEL_PATH)
    
    y_true = []
    y_pred = []
    
    print("Запуск инференса на тестовой выборке...")
    
    for cls_name in EXPECTED_CLASSES:
        cls_dir = os.path.join(TEST_DIR, cls_name)
        
        if not os.path.exists(cls_dir):
            print(f"[!] Внимание: папка {cls_dir} не найдена. Пропуск.")
            continue
            
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(images, desc=f"Класс: {cls_name.ljust(15)}"):
            img_path = os.path.join(cls_dir, img_name)
            
            # Инференс модели 
            results = model(img_path, verbose=False)[0]
            
            # Получаем предсказанный класс
            top1_index = results.probs.top1
            pred_class_name = model.names[top1_index]
            
            y_true.append(cls_name)
            y_pred.append(pred_class_name)

    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ YOLOv8m-cls (Окклюзии)")
    print("="*50)
    
    # 1. Top-1 Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\nTop-1 Accuracy: {acc:.4f}")
    
    # 2. Precision, Recall, F1-Score
    print("\nClassification Report (Precision, Recall, F1-Score):")
    report = classification_report(y_true, y_pred, target_names=EXPECTED_CLASSES, digits=4)
    print(report)
    
    # 3. Confusion Matrix
    print("[*] Построение Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=EXPECTED_CLASSES)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EXPECTED_CLASSES, yticklabels=EXPECTED_CLASSES)
    plt.title('Confusion Matrix - Face Occlusion Classification')
    plt.ylabel('True Label (Ground Truth)')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_CM_PATH, dpi=300)
    print(f"Матрица ошибок успешно сохранена в файл: {OUTPUT_CM_PATH}")

if __name__ == "__main__":
    evaluate_classifier()