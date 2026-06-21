import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from typing import Dict
from src.evaluation.mle_estimator import mle_estimates # для упрощения модуля bootstrap 


def bootstrap_nonparametric(df: pd.DataFrame, B: int = 100, random_seed: int = None) -> Dict:
    """
    Генерация B псевдовыборок с возвращением
    Для каждой вычисляются оценки (μ, σ, ρ)
    Возвращает распределение оценок
    """

    if random_seed is not None: 
        np.random.seed(random_seed)

    n = len(df)

    mu_x_bootstrap = np.zeros(B) # массив из нулей 
    mu_y_bootstrap = np.zeros(B)
    sigma_x_bootstrap = np.zeros(B)
    sigma_y_bootstrap = np.zeros(B)
    rho_bootstrap = np.zeros(B)

    for i in range(B): # итерации bootstrap

        indices = np.random.choice(n, size=n, replace=True) # случайным образом генерируем выборку 
        boot_sample = df.iloc[indices] # выбираем 
        estimates = mle_estimates(boot_sample)


        # Добавляем в наши пустные массивы оценки по итерациям B из модуля MLE 
        mu_x_bootstrap[i] = estimates['mu_x']
        mu_y_bootstrap[i] = estimates['mu_y']
        sigma_x_bootstrap[i] = estimates['sigma_x']
        sigma_y_bootstrap[i] = estimates['sigma_y']
        rho_bootstrap[i] = estimates['rho']

    return {
        'mu_x': mu_x_bootstrap,
        'mu_y': mu_y_bootstrap,
        'sigma_x': sigma_x_bootstrap,
        'sigma_y': sigma_y_bootstrap,
        'rho': rho_bootstrap,
        'B': B,                    
        'method': 'nonparametric'
    }
        



def bootstrap_parametric(df: pd.DataFrame, B: int = 100, random_seed: int = None) -> Dict:
    """
    Parametric bootstrap: генерация из оценённого распределения N(μ̂, Σ̂)
    """

    if random_seed is not None: 
        np.random.seed(random_seed) 

    estimates = mle_estimates(df)

    mu_x = estimates['mu_x']
    mu_y = estimates['mu_y']
    sigma_x = estimates['sigma_x']
    sigma_y = estimates['sigma_y']
    rho = estimates['rho']
    cov_matrix = estimates['cov_matrix']

    L = np.linalg.cholesky(cov_matrix) # применяем разложение Холецкого 

    # Инициализация массивов для хранения оценок
    mu_x_bootstrap = np.zeros(B)
    mu_y_bootstrap = np.zeros(B)
    sigma_x_bootstrap = np.zeros(B)
    sigma_y_bootstrap = np.zeros(B)
    rho_bootstrap = np.zeros(B)

    for i in range(B): # итерации bootstrap

        z = np.random.normal(0, 1, size=(len(df), 2))
        boot_sample = (L @ z.T).T + np.array([mu_x, mu_y])
        boot_df = pd.DataFrame(boot_sample, columns=['X', 'Y'])
        
        # Оценка параметров на псевдовыборке
        boot_estimates = mle_estimates(boot_df)

        # Добавляем в наши пустные массивы оценки по итерациям B из модуля MLE 
        mu_x_bootstrap[i] = boot_estimates['mu_x']
        mu_y_bootstrap[i] = boot_estimates['mu_y']
        sigma_x_bootstrap[i] = boot_estimates['sigma_x']
        sigma_y_bootstrap[i] = boot_estimates['sigma_y']
        rho_bootstrap[i] = boot_estimates['rho']

    return {
        'mu_x': mu_x_bootstrap,
        'mu_y': mu_y_bootstrap,
        'sigma_x': sigma_x_bootstrap,
        'sigma_y': sigma_y_bootstrap,
        'rho': rho_bootstrap,
        'B': B,                    
        'method': 'parametric'
    }


def plot_bootstrap(bootstrap_distributions: Dict,
                          confidence_level: float = 0.95,
                          figsize=(16, 8), dpi: int = 100) -> plt.Figure:
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi * 2
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['axes.labelsize'] = 9
    
    # Параметры для отображения
    params = ['mu_x', 'mu_y', 'sigma_x', 'sigma_y', 'rho']
    param_labels = ['μ_x', 'μ_y', 'σ_x', 'σ_y', 'ρ']
    
    # Увеличиваем figsize для лучшей читаемости
    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
    axes = axes.flatten()
    
    fig.delaxes(axes[5])

    alpha = 1 - confidence_level
    lower_p = (alpha / 2) * 100
    upper_p = (1 - alpha / 2) * 100

    for idx, (param, label) in enumerate(zip(params, param_labels)):
        ax = axes[idx]
        data = bootstrap_distributions[param]

        ax.hist(data, bins=25, density=True, alpha=0.6, 
                color='#3498db', edgecolor='white', linewidth=0.5)
        
        # Доверительные интервалы (вертикальные линии)
        ci_low = np.percentile(data, lower_p)
        ci_high = np.percentile(data, upper_p)
        
        ax.axvline(ci_low, color='#e74c3c', linestyle='--', linewidth=1.5)
        ax.axvline(ci_high, color='#e74c3c', linestyle='--', linewidth=1.5)
        ax.axvline(np.median(data), color='#2c3e50', linestyle='-', linewidth=1.5)

        ax.axvspan(ci_low, ci_high, alpha=0.15, color='#e74c3c')
        
        # Заголовок с интервалом
        ax.set_title(f'{label}\n[{ci_low:.2f}, {ci_high:.2f}]', 
                     fontsize=11, weight='bold')
        ax.set_xlabel('Значение', fontsize=9)
        ax.set_ylabel('Частота', fontsize=9)
        ax.grid(True, alpha=0.2)
        
        # Убираем лишние рамки
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    

    method = bootstrap_distributions.get('method', 'unknown')
    B = bootstrap_distributions.get('B', 100)
    fig.suptitle(f'Bootstrap распределения ({method}, B={B})', fontsize=14, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93]) 

    return fig

def quick_bootstrap(df: pd.DataFrame, B: int = 100, random_seed: int = None) -> Dict:
    return {
        'nonparametric': bootstrap_nonparametric(df, B, random_seed),
        'parametric': bootstrap_parametric(df, B, random_seed)
    }