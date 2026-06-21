import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

"""
Метод моментов (ММ) для двумерного нормального распределения

Для нормального распределения оценки метода моментов совпадают с ММП:
- μ_x = (1/n) * Σ x_i
- μ_y = (1/n) * Σ y_i
- σ_x² = (1/n) * Σ (x_i - μ_x)²
- σ_y² = (1/n) * Σ (y_i - μ_y)²
- ρ = (Σ (x_i - μ_x)(y_i - μ_y)) / (n * σ_x * σ_y)

"""


def mm_estimates(df: pd.DataFrame) -> dict:

    n = len(df)

    # Оценки средних
    mu_x = df['X'].mean()
    mu_y = df['Y'].mean()
    
    # Центрированные данные
    x_c = df['X'] - mu_x
    y_c = df['Y'] - mu_y

    var_x = np.sum(x_c ** 2) / n
    var_y = np.sum(y_c ** 2) / n

    sigma_x = np.sqrt(var_x)
    sigma_y = np.sqrt(var_y)

    cov_xy = np.sum(x_c * y_c) / n
    rho = cov_xy / (sigma_x * sigma_y) if sigma_x * sigma_y > 0 else 0

    cov_matrix = np.array([
        [var_x, cov_xy],
        [cov_xy, var_y]
    ])

    return {
        'mu_x': mu_x,
        'mu_y': mu_y,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'rho': rho,
        'cov_matrix': cov_matrix,
        'n_samples': n
    }


def mm_confidence_intervals(df: pd.DataFrame, confidence_level: float = 0.95) -> pd.DataFrame:

    n = len(df)
    estimates = mm_estimates(df)
    alpha = 1 - confidence_level # уровень значимости 
    z = norm.ppf(1 - alpha / 2)

    # Стандартные ошибки
    se_mu_x = estimates['sigma_x'] / np.sqrt(n)
    se_mu_y = estimates['sigma_y'] / np.sqrt(n)
    se_sigma_x = estimates['sigma_x'] / np.sqrt(2 * n)
    se_sigma_y = estimates['sigma_y'] / np.sqrt(2 * n)

    results = []

    results.append({
        'Параметр': 'μ_x',
        'Оценка': estimates['mu_x'],
        'Нижняя': estimates['mu_x'] - z * se_mu_x,
        'Верхняя': estimates['mu_x'] + z * se_mu_x
    })

    results.append({
        'Параметр': 'μ_y',
        'Оценка': estimates['mu_y'],
        'Нижняя': estimates['mu_y'] - z * se_mu_y,
        'Верхняя': estimates['mu_y'] + z * se_mu_y
    })

    results.append({
        'Параметр': 'σ_x',
        'Оценка': estimates['sigma_x'],
        'Нижняя': max(0, estimates['sigma_x'] - z * se_sigma_x),
        'Верхняя': estimates['sigma_x'] + z * se_sigma_x
    })

    results.append({
        'Параметр': 'σ_y',
        'Оценка': estimates['sigma_y'],
        'Нижняя': max(0, estimates['sigma_y'] - z * se_sigma_y),
        'Верхняя': estimates['sigma_y'] + z * se_sigma_y
    })

    return pd.DataFrame(results)

def mm_rho_confidence_intervals(df: pd.DataFrame, confidence_level: float = 0.95) -> pd.DataFrame:

    """
    Вычисляет доверительный интервал для ρ методом моментов
    с использованием Fisher Z-преобразования
    """

    n = len(df)
    estimates = mm_estimates(df)
    alpha = 1 - confidence_level
    z = norm.ppf(1 - alpha / 2)
    
    r = estimates['rho']
    
    # Fisher Z-преобразование
    z_r = 0.5 * np.log((1 + r) / (1 - r))
    se_z = 1 / np.sqrt(n - 3) if n > 3 else 1.0
    
    # Доверительный интервал для Z
    ci_z_low = z_r - z * se_z
    ci_z_up = z_r + z * se_z
    
    # Обратное преобразование
    ci_low = (np.exp(2 * ci_z_low) - 1) / (np.exp(2 * ci_z_low) + 1)
    ci_up = (np.exp(2 * ci_z_up) - 1) / (np.exp(2 * ci_z_up) + 1)
    
    return pd.DataFrame([{
        'Параметр': 'ρ',
        'Оценка (MM)': r,
        'Нижняя': max(-1, ci_low),
        'Верхняя': min(1, ci_up)
    }])

def mm_standard_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет стандартные ошибки оценок ММ
    
    Для нормального распределения стандартные ошибки совпадают с ММП
    """

    n = len(df)
    estimates = mm_estimates(df)

    se_mu_x = estimates['sigma_x'] / np.sqrt(n)
    se_mu_y = estimates['sigma_y'] / np.sqrt(n)
    se_sigma_x = estimates['sigma_x'] / np.sqrt(2 * n)
    se_sigma_y = estimates['sigma_y'] / np.sqrt(2 * n)
    se_rho = (1 - estimates['rho'] ** 2) / np.sqrt(n)
    
    return pd.DataFrame([
        {'Параметр': 'μ_x', 'Оценка': estimates['mu_x'], 'SE (MM)': se_mu_x},
        {'Параметр': 'μ_y', 'Оценка': estimates['mu_y'], 'SE (MM)': se_mu_y},
        {'Параметр': 'σ_x', 'Оценка': estimates['sigma_x'], 'SE (MM)': se_sigma_x},
        {'Параметр': 'σ_y', 'Оценка': estimates['sigma_y'], 'SE (MM)': se_sigma_y},
        {'Параметр': 'ρ', 'Оценка': estimates['rho'], 'SE (MM)': se_rho}
    ])


def mm_estimation_precision(df: pd.DataFrame, confidence_level: float = 0.95) -> pd.DataFrame:
    """
    Оценивает точность метода моментов для каждого параметра
    """
    

    n = len(df)
    estimates = mm_estimates(df)

    alpha = 1 - confidence_level
    z = norm.ppf(1 - alpha / 2)

    params = ['μ_x', 'μ_y', 'σ_x', 'σ_y', 'ρ']
    values = [estimates['mu_x'], estimates['mu_y'], 
              estimates['sigma_x'], estimates['sigma_y'], 
              estimates['rho']]
    
    se_mu_x = estimates['sigma_x'] / np.sqrt(n)
    se_mu_y = estimates['sigma_y'] / np.sqrt(n)
    se_sigma_x = estimates['sigma_x'] / np.sqrt(2 * n)
    se_sigma_y = estimates['sigma_y'] / np.sqrt(2 * n)
    se_rho = (1 - estimates['rho'] ** 2) / np.sqrt(n)

    se_values = [se_mu_x, se_mu_y, se_sigma_x, se_sigma_y, se_rho]
    
    results = []

    for param, value, se in zip(params, values, se_values):
        if param.startswith('σ') and value > 0: # ищем сигму, также она должна быть больше нуля 
            rel_error = se / value 
        else: 
            rel_error = np.nan 

        # Ширина доверительного интервала
        ci_width = 2 * z * se 

        # Категория точности
        if not np.isnan(rel_error):
            if rel_error < 0.05:
                precision_cat = "🟢 Высокая"
            elif rel_error < 0.15:
                precision_cat = "🟡 Средняя"
            else:
                precision_cat = "🔴 Низкая"
        else:
            precision_cat = "⚪ N/A"


        results.append({
            'param_mm': param,
            'estimate_mm': value,
            'SE_mm': se,
            'Relative_errors_mm': rel_error * 100 if not np.isnan(rel_error) else np.nan,
            'CI_mm': ci_width,
            'Accurancy_mm': precision_cat
        })

        return pd.DataFrame(results)
    

def plot_mm_standrad_errors(df: pd.DataFrame, figsize=(6, 3)) -> plt.Figure:
    """
    Визуализирует стандартные ошибки оценок метода моментов
    """

    fig, ax = plt.subplots(figsize=figsize)
    se_df = mm_standard_errors(df)

    params = ['μ_x', 'μ_y', 'σ_x', 'σ_y', 'ρ']
    se_values = se_df['SE (MM)'].values

    colors = ['#2ecc71' if se < 0.1 else '#f1c40f' if se < 0.3 else '#e74c3c' 
              for se in se_values]
    
    bars = ax.bar(params, se_values, color=colors, alpha=0.7, edgecolor='black', zorder=2)
    ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=10, color='#2c3e50', weight='bold')


    ax.set_ylim(0, max(se_values) * 1.15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')
    
    ax.set_xlabel('Параметр', fontsize=11, color='#2c3e50')
    ax.set_ylabel('Стандартная ошибка (SE)', fontsize=11, color='#2c3e50')
    ax.set_title('Стандартные ошибки оценок (Метод моментов)', 
                 fontsize=12, pad=15, color='#2c3e50', weight='bold')
    
    plt.tight_layout()
    return fig


def quick_mm(df: pd.DataFrame, confidence_level):

    mm_estimates_function = mm_estimates(df)
    mm_confidence_intervals_function = mm_confidence_intervals(df, confidence_level)
    mm_rho_confidence_intervals_function = mm_rho_confidence_intervals(df, confidence_level)
    mm_standard_errors_function = mm_standard_errors(df)
    mm_estimation_precision_function = mm_estimation_precision(df, confidence_level)


    return mm_estimates_function, mm_confidence_intervals_function, mm_rho_confidence_intervals_function, mm_standard_errors_function, mm_estimation_precision_function