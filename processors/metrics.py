import numpy as np
import matplotlib.pyplot as plt

from . import VectorIndexer
from . import VectorSearch

class MetricsCalculator:
    def __init__(self, results: list, ground_truth: list):
        self.results = results
        self.ground_truth = ground_truth
        
        self.found_paths = [res['path'] for res in self.results]
        
        found_set = set(self.found_paths)
        truth_set = set(self.ground_truth)
        
        self.tp = len(found_set.intersection(truth_set)) 
        self.fp = len(found_set.difference(truth_set))  
        self.fn = len(truth_set.difference(found_set))
        
        self.total_relevant = len(self.ground_truth)

    def calculate_set_metrics(self) -> dict:
        try:
            precision = self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            precision = 0.0

        try:
            recall = self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            recall = 0.0
            
        if precision + recall > 0:
            f_measure = 2 * (precision * recall) / (precision + recall)
        else:
            f_measure = 0.0
            
        return {'Precision': precision, 'Recall': recall, 'F-measure': f_measure}
            
    def calculate_rank_metrics(self) -> dict:
        metrics = {}

        # Precision@k
        for n in [5, 10]:
            top_n_paths = self.found_paths[:n]
            relevant_in_top_n = len(set(top_n_paths).intersection(set(self.ground_truth)))
            metrics[f'Precision@{n}'] = relevant_in_top_n / n
            
        # R-Precision
        r = self.total_relevant
        if r > 0:
            top_r_paths = self.found_paths[:r]
            relevant_in_top_r = len(set(top_r_paths).intersection(set(self.ground_truth)))
            metrics['R-Precision'] = relevant_in_top_r / r
        else:
            metrics['R-Precision'] = 0.0
                  
        # Average Precision
        if self.total_relevant == 0:
            metrics['Average Precision'] = 0.0
        else:
            precisions_at_hits = []
            relevant_found_count = 0
            for position, path in enumerate(self.found_paths, 1):
                if path in self.ground_truth:
                    relevant_found_count += 1
                    precision_at_k = relevant_found_count / position
                    precisions_at_hits.append(precision_at_k)
            
            if not precisions_at_hits:
                metrics['Average Precision'] = 0.0
            else:
                metrics['Average Precision'] = sum(precisions_at_hits) / self.total_relevant

        return metrics

    def get_interpolated_11_points(self) -> list:
        if self.total_relevant == 0:
            return [0.0] * 11
            
        recall_precision_points = []
        relevant_count = 0
        for i, res in enumerate(self.results):
            if res['path'] in self.ground_truth:
                relevant_count += 1
                recall = relevant_count / self.total_relevant
                precision = relevant_count / (i + 1)
                recall_precision_points.append((recall, precision))

        interpolated_precisions = []
        recall_levels = np.linspace(0.0, 1.0, 11)
        
        for r_level in recall_levels:
            prec_at_level = [p for r, p in recall_precision_points if r >= r_level]
            if not prec_at_level:
                interpolated_precisions.append(0.0)
            else:
                interpolated_precisions.append(max(prec_at_level))
        
        return interpolated_precisions

    def plot_precision_recall_curve(self, save_path="precision_recall_curve.png"):
        recall_levels = np.linspace(0.0, 1.0, 11)
        interpolated_precisions = self.get_interpolated_11_points()
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall_levels, interpolated_precisions, marker='o', linestyle='--')
        plt.xlabel("Полнота (Recall)")
        plt.ylabel("Точность (Precision)")
        plt.title("11-точечный график Полнота-Точность")
        plt.grid(True)
        plt.xticks(recall_levels)
        plt.yticks(np.linspace(0.0, 1.0, 11))
        plt.ylim(0, 1.05)
        plt.xlim(0, 1.0)
        
        plt.savefig(save_path)
        print(f"График сохранен в файл: {save_path}")

# evaluate.py

# ... (класс MetricsCalculator остается без изменений) ...
import os
def run_evaluation():
    # --- Конфигурация ---
    # Указываем пути к файлам индекса, как и в main.py
    INDEX_PATH = "processors/vector_index.faiss"
    MAP_PATH = "processors/path_map.pkl"

    # <<< НОВОЕ: Сначала проверяем, существуют ли файлы индекса
    if not os.path.exists(INDEX_PATH) or not os.path.exists(MAP_PATH):
        print(f"Ошибка: файлы индекса не найдены по путям '{INDEX_PATH}' и '{MAP_PATH}'.")
        print("Сначала запустите основное приложение (main.py), чтобы создать индекс.")
        return

    # --- Загрузка индекса ---
    print("Загрузка индекса...")
    try:
        # Пути для сканирования здесь не важны, т.к. мы только загружаем готовый индекс
        indexer = VectorIndexer(watch_paths=[], index_path=INDEX_PATH, map_path=MAP_PATH)
        if indexer.index.ntotal == 0:
            print("Ошибка: индекс найден, но он пуст. Проиндексируйте файлы в основном приложении.")
            return
    except Exception as e:
        print(f"Произошла ошибка при загрузке индекса: {e}")
        return
        
    search_engine = VectorSearch(indexer=indexer)
    
    # ... (остальная часть функции остается без изменений) ...
    query = input("Введите тестовый запрос: ")
    print("\nВведите эталонные ПУТИ К ФАЙЛАМ (например, local_fs/doc1.txt):")
    print("(каждый с новой строки, пустая строка для завершения)")
    ground_truth = []
    while True:
        line = input()
        if not line:
            break
        ground_truth.append(line.strip()) # Добавим .strip() для удаления случайных пробелов

    if not ground_truth:
        print("Не указаны эталонные документы. Оценка невозможна.")
        return
    
    results = search_engine.search(query, top_k=5)
    
    print("\n--- Результаты, полученные системой ---")
    if not results:
        print("Система ничего не нашла.")
    else:
        for res in results[:10]:
            print(f"Path: {res['path']}, Score: {res['score']:.4f}")
    
    metrics_calculator = MetricsCalculator(results=results, ground_truth=ground_truth)
    
    set_metrics = metrics_calculator.calculate_set_metrics()
    rank_metrics = metrics_calculator.calculate_rank_metrics()
    all_metrics = {**set_metrics, **rank_metrics}
    
    print("\n--- Таблица с метриками ---")
    for name, value in all_metrics.items():
        print(f"{name:<20}: {value:.4f}")

    metrics_calculator.plot_precision_recall_curve()


if __name__ == '__main__':
    run_evaluation()