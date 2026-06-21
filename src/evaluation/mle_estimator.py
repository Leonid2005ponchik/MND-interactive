import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

def mle_estimates(df: pd.DataFrame) -> dict:

    # функция вычисляет ММП оценки параметров двумерного нормального распределения

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
    
    return {
        'mu_x': mu_x,
        'mu_y': mu_y,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'rho': rho,
        'cov_matrix': np.array([[var_x, cov_xy], [cov_xy, var_y]])
    }

def confidence_intervals(df: pd.DataFrame, confidence_level: float = 0.95) -> pd.DataFrame:

    n = len(df)
    estimates = mle_estimates(df)
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


def fisher_function(df: pd.DataFrame, confidence_level: float = 0.95) -> pd.DataFrame:

    n = len(df)
    estimates = mle_estimates(df)
    alpha = 1 - confidence_level # уровень значимости 
    z = norm.ppf(1 - alpha / 2)


    r = estimates['rho']
    z_r = 0.5 * np.log((1 + r) / (1 - r))
    se_z = 1 / np.sqrt(n - 3) if n > 3 else 1.0
    ci_low = (np.exp(2 * (z_r - z * se_z)) - 1) / (np.exp(2 * (z_r - z * se_z)) + 1)
    ci_up = (np.exp(2 * (z_r + z * se_z)) - 1) / (np.exp(2 * (z_r + z * se_z)) + 1)

    result = []

    result.append({
        'Параметр': 'ρ',
        'Оценка': r,
        'Нижняя': max(-1, ci_low),
        'Верхняя': min(1, ci_up)
    })

    return pd.DataFrame(result)

def fisher_information_matrix(df: pd.DataFrame) -> pd.DataFrame: 
    """
    Вычисляет наблюдаемую информационную матрицу Фишера
    для параметров (μ_x, μ_y, σ_x, σ_y, ρ)
    """

    n = len(df)
    estimates = mle_estimates(df)

    sigma_x, sigma_y = estimates['sigma_x'], estimates['sigma_y']
    rho = estimates['rho']

    I = np.zeros((5, 5)) # Матрица 5x5 для параметров

    I[0, 0] = n / (sigma_x ** 2 * (1 - rho ** 2))
    I[0, 1] = n * rho / (sigma_x * sigma_y * (1 - rho ** 2))
    I[1, 0] = n * rho / (sigma_x * sigma_y * (1 - rho ** 2))
    I[1, 1] = n / (sigma_y ** 2 * (1 - rho ** 2))

    result = pd.DataFrame(
    I[:2, :2],  # берем только блок для средних
    index=['μ_x', 'μ_y'],
    columns=['μ_x', 'μ_y']
    )

    return pd.DataFrame(result)

def fisher_information_sigma(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет информационную матрицу Фишера для блока ковариации (σ_x, σ_y, ρ)
    
    Returns:
        pd.DataFrame: матрица 3x3 для параметров σ_x, σ_y, ρ
    """
    n = len(df)
    estimates = mle_estimates(df)
    
    sigma_x = estimates['sigma_x']
    sigma_y = estimates['sigma_y']
    rho = estimates['rho']

    # Защита от численных проблем при |ρ| → 1
    epsilon = 1e-10
    rho_safe = np.clip(rho, -1 + epsilon, 1 - epsilon)

    I_sigma = np.array([
        #        σ_x                    σ_y                    ρ
        [2 * n / (sigma_x**2),                          # σ_x, σ_x
         2 * n * rho_safe**2 / (sigma_x * sigma_y),    # σ_x, σ_y
         -2 * n * rho_safe / (sigma_x * np.sqrt(1 - rho_safe**2))],  # σ_x, ρ
        
        [2 * n * rho_safe**2 / (sigma_x * sigma_y),    # σ_y, σ_x
         2 * n / (sigma_y**2),                         # σ_y, σ_y
         -2 * n * rho_safe / (sigma_y * np.sqrt(1 - rho_safe**2))],  # σ_y, ρ
        
        [-2 * n * rho_safe / (sigma_x * np.sqrt(1 - rho_safe**2)),   # ρ, σ_x
         -2 * n * rho_safe / (sigma_y * np.sqrt(1 - rho_safe**2)),   # ρ, σ_y
         n * (1 + rho_safe**2) / (1 - rho_safe**2)**2]               # ρ, ρ
    ])

    return pd.DataFrame(
        I_sigma,
        index=['σ_x', 'σ_y', 'ρ'],
        columns=['σ_x', 'σ_y', 'ρ']
    )

def fisher_information_inv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет обратную информационную матрицу Фишера (ковариационную матрицу оценок)
    
    Returns:
        pd.DataFrame: обратная матрица 5x5
    """

    I_full = np.zeros((5, 5))
    I_mu = fisher_information_matrix(df).values  # 2x2
    I_sigma = fisher_information_sigma(df).values  # 3x3

    I_full[:2, :2] = I_mu      # блок для средних
    I_full[2:, 2:] = I_sigma   # блок для ковариации

    try:
        I_inv = np.linalg.inv(I_full)
    except np.linalg.LinAlgError:
        # Если матрица вырождена (|ρ|=1)
        I_inv = np.full((5, 5), np.nan)

    return pd.DataFrame(
        I_inv,
        index=['μ_x', 'μ_y', 'σ_x', 'σ_y', 'ρ'],
        columns=['μ_x', 'μ_y', 'σ_x', 'σ_y', 'ρ']
    )   

def fisher_standard_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Извлекает стандартные ошибки из обратной матрицы Фишера
    """
    
    estimates = mle_estimates(df)
    I_inv = fisher_information_inv(df).values

    if np.isnan(I_inv).any():
        return pd.DataFrame([{
            'Параметр': 'Ошибка',
            'SE': np.nan,
            'Примечание': 'Матрица вырождена (|ρ|=1)'
        }])
    
    params = ['μ_x', 'μ_y', 'σ_x', 'σ_y', 'ρ']
    values = [estimates['mu_x'], estimates['mu_y'], 
              estimates['sigma_x'], estimates['sigma_y'], 
              estimates['rho']]
    
    results = []
    for i, (param, value) in enumerate(zip(params, values)):
        se = np.sqrt(max(0, I_inv[i, i]))
        results.append({
            'Параметр': param,
            'Оценка': value,
            'SE (Fisher)': se
        })
    
    return pd.DataFrame(results)


def estimation_precision(df: pd.DataFrame, confidence_level) -> pd.DataFrame:
    results = []

    estimates = mle_estimates(df)
    I_inv = fisher_information_inv(df).values

    params = ['μ_x', 'μ_y', 'σ_x', 'σ_y', 'ρ']
    values = [estimates['mu_x'], estimates['mu_y'], 
              estimates['sigma_x'], estimates['sigma_y'], 
              estimates['rho']]
    
    z = norm.ppf(1 - (1 - confidence_level) / 2)
    
    for i, (param, value) in enumerate(zip(params, values)):
        se = np.sqrt(max(0, I_inv[i, i]))

        if param.startswith('σ') and value > 0:
            rel_error = se / value
        else:
            rel_error = np.nan

        ci_width = 2 * z * se

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
            'params': param,
            'estimation': value,
            'SE': se,
            'Relative_error': rel_error * 100 if not np.isnan(rel_error) else np.nan,
            'CI_width': ci_width,
            'Accuracy': precision_cat
        })

        

    return pd.DataFrame(results)


def plot_standard_errors(df: pd.DataFrame, figsize=(6, 3)) -> plt.Figure: 
    """
    Визуализирует стандартные ошибки оценок
    """



    fig, ax = plt.subplots(figsize=figsize)

    se_df = fisher_standard_errors(df)

    params = ['μ_x', 'μ_y', 'σ_x', 'σ_y', 'ρ']
    se_values = se_df['SE (Fisher)'].values
    
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
    ax.set_title('Стандартные ошибки оценок параметров', fontsize=12, pad=15, color='#2c3e50', weight='bold')

    plt.tight_layout()
    return fig

def quick_mle(df: pd.DataFrame, confidence_level):
    """Быстрый расчёт ММП: оценки + доверительные интервалы"""
    estimates = mle_estimates(df)
    ci = confidence_intervals(df)
    fisher = fisher_function(df)
    fisher_matrix = fisher_information_matrix(df) # нужно вставить 
    fisher_sigma = fisher_information_sigma(df) # Нужно вставить 
    fisher_inv = fisher_information_inv(df)
    fisher_errors = fisher_standard_errors(df)
    fisher_estimation_precision = estimation_precision(df, confidence_level)

    return estimates, ci, fisher, fisher_matrix, fisher_sigma, fisher_inv, fisher_errors, fisher_estimation_precision