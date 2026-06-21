import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, probplot
from typing import Dict

"""
Статистические тесты для проверки нормальности данных

1. Тест Шапиро-Уилка для маргинальных распределений X и Y
2. Q-Q plots для визуальной проверки нормальности
"""

def shapiro_wilk_test(df: pd.DataFrame) -> pd.DataFrame:

    """
    Выполняет тест Шапиро-Уилка для X и Y

    Нулевая гипотеза H0: данные взяты из нормального распределения
    Альтернативная гипотеза H1: Если p-value < 0.05, нормальность отвергается
    """

    results = []

    for col in ['X', 'Y']:
        statistic, p_value = shapiro(df[col])

        if p_value < 0.05: 
            conclusion = "Отвергаем H0, работает гипотеза H1: распределение является не нормальным исходя из вероятности ошибки"
        else: 
            conclusion = " Гипотеза H0 верная: данные распределены нормально"


        results.append({
            'Переменная': col,
            'Статистика теста': statistic,
            'p-value': p_value,
            'Заключение (α=0.05)': conclusion
        })

    return pd.DataFrame(results)

def shapiro_wilk_test_summary(df: pd.DataFrame) -> Dict:
    """
    Краткая сводка теста Шапиро-Уилка
    """
    
    summary = {}

    for col in ['X', 'Y']:
        statistic, p_value = shapiro(df[col])
        summary[col] = {
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value >= 0.05
        }

    return summary


def qq_plot(df: pd.DataFrame, figsize=(12, 5), dpi: int = 100) -> plt.Figure:

    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi * 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Q-Q plot для X
    probplot(df['X'], dist="norm", plot=ax1)
    ax1.set_title('Q-Q plot для X', fontsize=12, weight='bold')
    ax1.set_xlabel('Теоретические квантили', fontsize=10)
    ax1.set_ylabel('Выборочные квантили', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Q-Q plot для Y
    probplot(df['Y'], dist="norm", plot=ax2)
    ax2.set_title('Q-Q plot для Y', fontsize=12, weight='bold')
    ax2.set_xlabel('Теоретические квантили', fontsize=10)
    ax2.set_ylabel('Выборочные квантили', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Q-Q plots для проверки нормальности', fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()

    return fig


def quick_statistical_tests(df: pd.DataFrame) -> Dict:
  
    shapiro_results = shapiro_wilk_test(df)
    shapiro_summary = shapiro_wilk_test_summary(df)
    qq_fig = qq_plot(df)

    return {
        'shapiro_results': shapiro_results,
        'shapiro_summary': shapiro_summary,
        'qq_plot': qq_fig
    }