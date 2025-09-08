import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from processors import Search

class MetricsCalculator:
    def __init__(self, results: pd.DataFrame, ground_truth: list):
        self.results = results
        self.ground_truth = ground_truth
        
        found_set = set(list(self.results['TITLE']))
        truth_set = set(self.ground_truth)
        
        self.a = len(found_set.intersection(truth_set))  # True Positives (TP)
        self.b = len(found_set.difference(truth_set))      # False Positives (FP)
        self.c = len(truth_set.difference(found_set))      # False Negatives (FN)
        
        self.total_relevant = len(self.ground_truth)

    def calculate_set_metrics(self) -> dict:
        try:
            precision = self.a / (self.a + self.b)
        except ZeroDivisionError:
            precision = 0.0

        try:
            recall = self.a / (self.a + self.c)
        except ZeroDivisionError:
            recall = 0.0
            
        if precision + recall > 0:
            f_measure = 2 * (precision * recall) / (precision + recall)
        else:
            f_measure = 0.0
            
        return {
            'Precision': precision,
            'Recall': recall,
            'F-measure': f_measure
        }
        
    def calculate_rank_metrics(self) -> dict:
        metrics = {}
        for n in [5, 10]:
            top_n = self.results.head(n)
            relevant_in_top_n = len(set(top_n['TITLE']).intersection(set(self.ground_truth)))
            metrics[f'Precision@{n}'] = relevant_in_top_n / n
            
        r = self.total_relevant
        if r > 0:
            top_r = self.results.head(r)
            relevant_in_top_r = len(set(top_r['TITLE']).intersection(set(self.ground_truth)))
            metrics['R-Precision'] = relevant_in_top_r / r
        else:
            metrics['R-Precision'] = 0.0
            
        if self.total_relevant == 0:
            metrics['Average Precision'] = 0.0
        else:
            precisions = []
            relevant_count = 0
            for i, row in self.results.iterrows():
                if row['TITLE'] in self.ground_truth:
                    relevant_count += 1
                    precisions.append(relevant_count / (i + 1))
            
            if not precisions:
                 metrics['Average Precision'] = 0.0
            else:
                 metrics['Average Precision'] = sum(precisions) / self.total_relevant

        return metrics

    def get_interpolated_11_points(self) -> list:
        if self.total_relevant == 0:
            return [0.0] * 11
            
        recall_precision_points = []
        relevant_count = 0
        for i, row in self.results.iterrows():
            if row['TITLE'] in self.ground_truth:
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
        plt.title("11-точечный график Полнота-Точность (TREC-style)")
        plt.grid(True)
        plt.xticks(recall_levels)
        plt.yticks(np.linspace(0.0, 1.0, 11))
        plt.ylim(0, 1.05)
        plt.xlim(0, 1.0)
        
        plt.savefig(save_path)
        print(f"График сохранен в файл: {save_path}")


def run_evaluation_submenu():
    if not os.path.exists("processors/data.csv"):
        print("Ошибка: файл 'data.csv' не найден. Сначала создайте индекс.")
        return
        
    search_engine = Search(database="processors/data.csv")
    query = input("Введите тестовый запрос: ")
    print("Введите эталонные заголовки релевантных статей (каждый с новой строки, пустая строка для завершения):")
    ground_truth = []
    while True:
        line = input()
        if not line:
            break
        ground_truth.append(line)

    if not ground_truth:
        print("Не указаны эталонные документы. Оценка невозможна.")
        return
    
    results = search_engine.search(query)
    print("\nРезультаты, полученные системой:")
    print(results)
    
    metrics_calculator = MetricsCalculator(results=results, ground_truth=ground_truth)
    
    set_metrics = metrics_calculator.calculate_set_metrics()
    rank_metrics = metrics_calculator.calculate_rank_metrics()
    all_metrics = {**set_metrics, **rank_metrics}
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index', columns=['Value'])
    
    print("\n--- Таблица с метриками ---")
    print(metrics_df)

    metrics_calculator.plot_precision_recall_curve()


if __name__ == '__main__':
    run_evaluation_submenu()