import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def save_confusion_matrix(results_dir):
    csv_path = os.path.join(results_dir, "benchmark_final.csv")
    df = pd.read_csv(csv_path)
    
    # Расчет матрицы
    cm = confusion_matrix(df['gt'], df['pred'])
    
    plt.figure(figsize=(10, 8))
    sns.set_context("paper", font_scale=1.8)
    
    # Рисуем тепловую карту
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Rejected', 'Compliant'], 
                yticklabels=['Rejected', 'Compliant'],
                cbar=False, annot_kws={"size": 25})
    
    plt.title('Матрица классификации (Confusion Matrix)', pad=20)
    plt.xlabel('Предсказание системы', labelpad=20)
    plt.ylabel('Эталонная разметка', labelpad=20)
    
    plt.tight_layout()
    out_path = os.path.join(results_dir, "fig_confusion_matrix.png")
    plt.savefig(out_path, dpi=300)
    print(f"Матрица сохранена в {out_path}")
    plt.show()

if __name__ == "__main__":
    save_confusion_matrix("docs/results")
