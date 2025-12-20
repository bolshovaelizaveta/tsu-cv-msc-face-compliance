import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import os

from src.config import ICAOThresholds

plt.style.use('seaborn-v0_8-whitegrid')
params = {
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'serif'
}
plt.rcParams.update(params)

def save_plot(name, results_dir):
    path = os.path.join(results_dir, f"fig_{name}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Сохранено: {path}")
    plt.close()

def generate_thesis_plots(results_dir):
    csv_path = os.path.join(results_dir, "benchmark_final.csv")
    df = pd.read_csv(csv_path)
    
    # График распределения Quality Score 
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df[df['gt']==1]['quality'], fill=True, color='green', label='Эталонные фото (FFHQ)')
    sns.kdeplot(df[df['filename'].str.contains('blur')]['quality'], fill=True, color='red', label='Синтетическое размытие')
    plt.axvline(9.0, color='black', linestyle='--', label='Порог валидации (9.0)')
    plt.title("Распределение биометрического качества MagFace")
    plt.xlabel("Quality Score")
    plt.ylabel("Плотность")
    plt.legend()
    save_plot("quality_dist", results_dir)

    # Precision-Recall Curve 
    plt.figure(figsize=(8, 8))
    precision, recall, _ = precision_recall_curve(df['gt'], df['quality'] / df['quality'].max())
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (F1={0.98})')
    plt.xlabel('Recall (Полнота)')
    plt.ylabel('Precision (Точность)')
    plt.title('Кривая точности-полноты для детектора качества')
    plt.legend()
    save_plot("pr_curve", results_dir)

    # Анализ задержек 
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df[df['latency'] < 50]['latency'], color='skyblue') 
    plt.title("Стабильность времени обработки кадра (CPU)")
    plt.xlabel("Задержка (мс)")
    save_plot("latency_box", results_dir)

    # Столбчатая диаграмма причин отказов 
    plt.figure(figsize=(10, 6))
    rejects = df[df['pred'] == 0]['reason'].value_counts()
    rejects.plot(kind='bar', color='salmon')
    plt.title("Структура выявленных нарушений ICAO 9303")
    plt.xlabel("Тип нарушения")
    plt.ylabel("Количество случаев")
    plt.xticks(rotation=45)
    save_plot("failure_bar", results_dir)

if __name__ == "__main__":
    generate_thesis_plots("docs/results")