import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_ranks(models_results: List[Dict[str, Union[float, int]]], model_names: List[str]) -> pd.DataFrame:
    """
    Рассчитывает общие ранги моделей на основе всех метрик.
    """
    metrics_data = []
    
    for model_idx, results in enumerate(models_results):
        model_metrics = []
        for metric in ['precision', 'recall', 'ndcg']:
            for k in [1, 3, 5, 10, 20]:
                key = f'{metric}@{k}'
                if key in results:
                    model_metrics.append(results[key])
        metrics_data.append(np.mean(model_metrics))
    
    ranks = stats.rankdata(-np.array(metrics_data))
    
    rank_df = pd.DataFrame({
        'Модель': model_names,
        'Средний ранг': ranks,
        'Среднее значение метрик': metrics_data
    })
    rank_df.set_index('Модель', inplace=True)
    
    return rank_df

def plot_metrics_comparison(models_results: List[Dict[str, Union[float, int]]], 
                          model_names: List[str],
                          k_values: List[int] = [1, 3, 5, 10, 20]):
    """
    Создает три графика для сравнения моделей: Mean Recall@K, Mean Precision@K и Mean NDCG@K
    """
    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    metrics = ['recall', 'precision', 'ndcg']
    titles = ['Mean Recall @ K', 'Mean Precision @ K', 'Mean NDCG @ K']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f', '#9b59b6']
    markers = ['o', 's', '^', 'D', 'v']
    
    for model_name, results in zip(model_names, models_results):
        missing_metrics = []
        for metric in metrics:
            for k in k_values:
                key = f'{metric}@{k}'
                if key not in results:
                    missing_metrics.append(key)
        if missing_metrics:
            print(f"Предупреждение: В модели {model_name} отсутствуют метрики: {', '.join(missing_metrics)}")
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        for i, (results, model_name) in enumerate(zip(models_results, model_names)):
            values = []
            for k in k_values:
                key = f'{metric}@{k}'
                value = results.get(key, None)
                values.append(value)
            
            if any(v is not None for v in values):
                values = [np.nan if v is None else v for v in values]
                ax.plot(k_values, values, '-', 
                       color=colors[i % len(colors)],
                       linewidth=2,
                       label=model_name)
                ax.plot(k_values, values, markers[i % len(markers)],
                       color=colors[i % len(colors)],
                       markersize=8,
                       markeredgewidth=2,
                       markeredgecolor='white')
        
        ax.set_xlabel('K (Number of Recommendations)', fontsize=12)
        ax.set_ylabel(f'Mean {metric.capitalize()}', fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1.05)
        ax.legend(title='Model', title_fontsize=12, fontsize=10, 
                 loc='lower right' if metric == 'recall' else 'upper right',
                 frameon=True)
        ax.set_xticks(k_values)
        
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig


def print_comparison_report(models_results, model_names):
    """
    Выводит отчет о сравнении моделей: ранги и значения метрик
    """
    stats_df = pd.DataFrame(index=['mean', 'median', 'std'])
    
    model_avg_ranks = {model: [] for model in model_names}
    
    for model_type in ['precision', 'recall', 'ndcg']:
        model_ranks = []
        for k in [5, 10, 20]:
            values = [result.get(f'{model_type}@{k}', 0) for result in models_results]
            ranks = stats.rankdata(-np.array(values))
            model_ranks.extend(ranks)
            
            for i, rank in enumerate(ranks):
                model_avg_ranks[model_names[i]].append(rank)
        
        if model_ranks:
            model_ranks = np.array(model_ranks)
            stats_df[f'{model_type}_rank'] = [
                np.mean(model_ranks),
                np.median(model_ranks),
                np.std(model_ranks)
            ]
    
    metrics_df = pd.DataFrame(index=model_names)
    for i, result in enumerate(models_results):
        metrics_df.loc[model_names[i], 'ROC AUC'] = result.get('roc_auc', 0)
        metrics_df.loc[model_names[i], 'Avg Precision'] = result.get('average_precision', 0)
        
        recall_values = []
        ndcg_values = []
        for k in [1, 3, 5, 10, 20]:
            recall_values.append(result.get(f'recall@{k}', 0))
            ndcg_values.append(result.get(f'ndcg@{k}', 0))
        
        metrics_df.loc[model_names[i], 'Avg Recall'] = np.mean(recall_values)
        metrics_df.loc[model_names[i], 'Avg NDCG'] = np.mean(ndcg_values)
    
    avg_ranks = {model: np.mean(ranks) for model, ranks in model_avg_ranks.items()}
    rank_df = pd.DataFrame.from_dict(avg_ranks, orient='index', columns=['Average Rank'])
    rank_df = rank_df.sort_values('Average Rank')
    

    print("\nСтатистика рангов:")
    print(stats_df[['precision_rank', 'recall_rank', 'ndcg_rank']].round(6))
    
    print("\nРанги моделей:")
    print(rank_df.round(3))
    
    print("\nЗначения метрик:")
    print(metrics_df.round(4))