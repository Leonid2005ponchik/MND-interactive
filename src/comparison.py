"""
Сравнительный анализ методов оценивания для двумерного нормального распределения

Критерии сравнения:
1. Смещение (bias): E[θ̂] - θ
2. Дисперсия (variance): Var(θ̂)
3. Среднеквадратичная ошибка (MSE): Bias² + Var
4. Доверительные интервалы: длина и coverage probability
5. Вычислительная сложность: время выполнения
"""


import numpy as np 
import pandas as pd 
import time 
from typing import Dict, Tuple, List
from scipy.stats import norm

# Импортируем все методы оценивания
from src.evaluation.mle_estimator import mle_estimates
from src.evaluation.mm_estimator import mm_estimates
from src.evaluation.bootstrap import bootstrap_nonparametric, bootstrap_parametric
from src.data_generator import MultivariateNormalGenerator


def compute_bias(estimates: Dict, true_params: Dict) -> Dict:
    """
    Вычисляет смещение для каждого параметра (bias): E[θ̂] - θ
    """
    
    
    return {
        'mu_x': estimates['mu_x'] - true_params['mu_x'],
        'mu_y': estimates['mu_y'] - true_params['mu_y'],
        'sigma_x': estimates['sigma_x'] - true_params['sigma_x'],
        'sigma_y': estimates['sigma_y'] - true_params['sigma_y'],
        'rho': estimates['rho'] - true_params['rho']
    }

def compute_variance(estimates_list: List[Dict]) -> Dict:
    """
    Вычисляет дисперсию оценок по множеству повторений
    """

    n = len(estimates_list)

    # защита от ошибок если список пустой 

    if n == 0:
        return {k: np.nan for k in ['mu_x', 'mu_y', 'sigma_x', 'sigma_y', 'rho']}
    
    # Извлекаем значения по каждому параметру
    mu_x_vals = [e['mu_x'] for e in estimates_list]
    mu_y_vals = [e['mu_y'] for e in estimates_list]
    sigma_x_vals = [e['sigma_x'] for e in estimates_list]
    sigma_y_vals = [e['sigma_y'] for e in estimates_list]
    rho_vals = [e['rho'] for e in estimates_list]
    
    return {
        'mu_x': np.var(mu_x_vals, ddof=1),
        'mu_y': np.var(mu_y_vals, ddof=1),
        'sigma_x': np.var(sigma_x_vals, ddof=1),
        'sigma_y': np.var(sigma_y_vals, ddof=1),
        'rho': np.var(rho_vals, ddof=1)
    }

def compute_mse(estimates: Dict, true_params: Dict, variance: Dict) -> Dict:

    bias = compute_bias(estimates, true_params)
    
    return {
        'mu_x': bias['mu_x']**2 + variance['mu_x'],
        'mu_y': bias['mu_y']**2 + variance['mu_y'],
        'sigma_x': bias['sigma_x']**2 + variance['sigma_x'],
        'sigma_y': bias['sigma_y']**2 + variance['sigma_y'],
        'rho': bias['rho']**2 + variance['rho']
    }


def compute_ci_length(ci_df: pd.DataFrame) -> Dict:
    """
    Вычисляет длину доверительных интервалов
    """
    
    lengths = {}
    for _, row in ci_df.iterrows():
        param = row['Параметр']
        lengths[param] = row['Верхняя'] - row['Нижняя']
    
    return lengths


def run_comparison_simulation(true_params: Dict,n_samples: int, n_experiments: int = 100,
                                bootstrap_B: int = 100,
                                confidence_level: float = 0.95,
                                random_seed: int = 42
                                ) -> Dict:
    
    # симуляция для сравнения методов
    """
    Параметры:
    true_params : dict
        Истинные параметры распределения
    n_samples : int
        Размер выборки
    n_experiments : int
        Количество повторений эксперимента
    bootstrap_B : int
        Количество bootstrap итераций
    confidence_level : float
        Уровень доверия
    random_seed : int
        Зерно для воспроизводимости
    """

    np.random.seed(random_seed)


    # Инициализация списков для хранения оценок
    mle_estimates_list = []
    mm_estimates_list = []
    bootstrap_np_estimates_list = []
    bootstrap_p_estimates_list = []

    # Время выполнения
    times = {
        'mle': [],
        'mm': [],
        'bootstrap_nonparametric': [],
        'bootstrap_parametric': []
    }

    print(f"Запуск симуляции: {n_experiments} экспериментов, n={n_samples}")

    for exp in range(n_experiments): # Логгер прогресса, чтобы было видно долгие вычисления 
        if exp % 10 == 0:
            print(f"  Эксперимент {exp+1}/{n_experiments}")

        generator = MultivariateNormalGenerator(
            mu_x=true_params['mu_x'],
            mu_y=true_params['mu_y'],
            sigma_x=true_params['sigma_x'],
            sigma_y=true_params['sigma_y'],
            rho=true_params['rho']
        )

        df = generator.generate(
            n_samples=n_samples,
            random_seed=random_seed + exp
        )

        start = time.time()
        mle_est = mle_estimates(df)
        times['mle'].append(time.time() - start)
        mle_estimates_list.append(mle_est)

        start = time.time()
        mm_est = mm_estimates(df)
        times['mm'].append(time.time() - start)
        mm_estimates_list.append(mm_est)

        start = time.time()
        boot_np = bootstrap_nonparametric(df, B=bootstrap_B, random_seed=random_seed + exp)
        boot_np_est = {
            'mu_x': np.mean(boot_np['mu_x']),
            'mu_y': np.mean(boot_np['mu_y']),
            'sigma_x': np.mean(boot_np['sigma_x']),
            'sigma_y': np.mean(boot_np['sigma_y']),
            'rho': np.mean(boot_np['rho'])
        }
        times['bootstrap_nonparametric'].append(time.time() - start)
        bootstrap_np_estimates_list.append(boot_np_est)

        start = time.time()
        boot_p = bootstrap_parametric(df, B=bootstrap_B, random_seed=random_seed + exp)
        boot_p_est = {
            'mu_x': np.mean(boot_p['mu_x']),
            'mu_y': np.mean(boot_p['mu_y']),
            'sigma_x': np.mean(boot_p['sigma_x']),
            'sigma_y': np.mean(boot_p['sigma_y']),
            'rho': np.mean(boot_p['rho'])
        }
        times['bootstrap_parametric'].append(time.time() - start)
        bootstrap_p_estimates_list.append(boot_p_est)


    # Усредняем 
    def average_estimates(estimates_list):
        n = len(estimates_list)
        if n == 0:
            return {k: np.nan for k in ['mu_x', 'mu_y', 'sigma_x', 'sigma_y', 'rho']}

        return {
        'mu_x': np.mean([e['mu_x'] for e in estimates_list]),
        'mu_y': np.mean([e['mu_y'] for e in estimates_list]),
        'sigma_x': np.mean([e['sigma_x'] for e in estimates_list]),
        'sigma_y': np.mean([e['sigma_y'] for e in estimates_list]),
        'rho': np.mean([e['rho'] for e in estimates_list])
    }


    avg_mle = average_estimates(mle_estimates_list)
    avg_mm = average_estimates(mm_estimates_list)
    avg_boot_np = average_estimates(bootstrap_np_estimates_list)
    avg_boot_p = average_estimates(bootstrap_p_estimates_list)
    
    # Вычисляем дисперсии
    var_mle = compute_variance(mle_estimates_list)
    var_mm = compute_variance(mm_estimates_list)
    var_boot_np = compute_variance(bootstrap_np_estimates_list)
    var_boot_p = compute_variance(bootstrap_p_estimates_list)
    
    # Вычисляем MSE
    mse_mle = compute_mse(avg_mle, true_params, var_mle)
    mse_mm = compute_mse(avg_mm, true_params, var_mm)
    mse_boot_np = compute_mse(avg_boot_np, true_params, var_boot_np)
    mse_boot_p = compute_mse(avg_boot_p, true_params, var_boot_p)


    # Усреднённое время выполнения
    avg_times = {
        'mle': np.mean(times['mle']),
        'mm': np.mean(times['mm']),
        'bootstrap_nonparametric': np.mean(times['bootstrap_nonparametric']),
        'bootstrap_parametric': np.mean(times['bootstrap_parametric'])
    }


    return {
        'true_params': true_params,
        'n_samples': n_samples,
        'n_experiments': n_experiments,
        'confidence_level': confidence_level,
        'methods': {
            'mle': {
                'avg_estimates': avg_mle,
                'variance': var_mle,
                'mse': mse_mle,
                'bias': compute_bias(avg_mle, true_params)
            },
            'mm': {
                'avg_estimates': avg_mm,
                'variance': var_mm,
                'mse': mse_mm,
                'bias': compute_bias(avg_mm, true_params)
            },
            'bootstrap_nonparametric': {
                'avg_estimates': avg_boot_np,
                'variance': var_boot_np,
                'mse': mse_boot_np,
                'bias': compute_bias(avg_boot_np, true_params)
            },
            'bootstrap_parametric': {
                'avg_estimates': avg_boot_p,
                'variance': var_boot_p,
                'mse': mse_boot_p,
                'bias': compute_bias(avg_boot_p, true_params)
            }
        },
        'avg_time': avg_times
    }


def comparison_to_dataframe(results: Dict) -> Dict[str, pd.DataFrame]:

    true_params = results['true_params']
    methods = results['methods']

    estimates_df = pd.DataFrame({
        'Параметр': ['μ_x', 'μ_y', 'σ_x', 'σ_y', 'ρ'],
        'Истинное значение': [true_params['mu_x'], true_params['mu_y'],
                              true_params['sigma_x'], true_params['sigma_y'],
                              true_params['rho']],
        'ММП': [methods['mle']['avg_estimates']['mu_x'],
                methods['mle']['avg_estimates']['mu_y'],
                methods['mle']['avg_estimates']['sigma_x'],
                methods['mle']['avg_estimates']['sigma_y'],
                methods['mle']['avg_estimates']['rho']],
        'Метод моментов': [methods['mm']['avg_estimates']['mu_x'],
                           methods['mm']['avg_estimates']['mu_y'],
                           methods['mm']['avg_estimates']['sigma_x'],
                           methods['mm']['avg_estimates']['sigma_y'],
                           methods['mm']['avg_estimates']['rho']],
        'Bootstrap (NP)': [methods['bootstrap_nonparametric']['avg_estimates']['mu_x'],
                           methods['bootstrap_nonparametric']['avg_estimates']['mu_y'],
                           methods['bootstrap_nonparametric']['avg_estimates']['sigma_x'],
                           methods['bootstrap_nonparametric']['avg_estimates']['sigma_y'],
                           methods['bootstrap_nonparametric']['avg_estimates']['rho']],
        'Bootstrap (P)': [methods['bootstrap_parametric']['avg_estimates']['mu_x'],
                          methods['bootstrap_parametric']['avg_estimates']['mu_y'],
                          methods['bootstrap_parametric']['avg_estimates']['sigma_x'],
                          methods['bootstrap_parametric']['avg_estimates']['sigma_y'],
                          methods['bootstrap_parametric']['avg_estimates']['rho']]
    })

    mse_df = pd.DataFrame({
        'Параметр': ['μ_x', 'μ_y', 'σ_x', 'σ_y', 'ρ'],
        'MSE (ММП)': [methods['mle']['mse']['mu_x'],
                      methods['mle']['mse']['mu_y'],
                      methods['mle']['mse']['sigma_x'],
                      methods['mle']['mse']['sigma_y'],
                      methods['mle']['mse']['rho']],
        'MSE (ММ)': [methods['mm']['mse']['mu_x'],
                     methods['mm']['mse']['mu_y'],
                     methods['mm']['mse']['sigma_x'],
                     methods['mm']['mse']['sigma_y'],
                     methods['mm']['mse']['rho']],
        'MSE (Bootstrap NP)': [methods['bootstrap_nonparametric']['mse']['mu_x'],
                               methods['bootstrap_nonparametric']['mse']['mu_y'],
                               methods['bootstrap_nonparametric']['mse']['sigma_x'],
                               methods['bootstrap_nonparametric']['mse']['sigma_y'],
                               methods['bootstrap_nonparametric']['mse']['rho']],
        'MSE (Bootstrap P)': [methods['bootstrap_parametric']['mse']['mu_x'],
                              methods['bootstrap_parametric']['mse']['mu_y'],
                              methods['bootstrap_parametric']['mse']['sigma_x'],
                              methods['bootstrap_parametric']['mse']['sigma_y'],
                              methods['bootstrap_parametric']['mse']['rho']]
    })


     # Сравнение смещения
    bias_df = pd.DataFrame({
        'Параметр': ['μ_x', 'μ_y', 'σ_x', 'σ_y', 'ρ'],
        'Смещение (ММП)': [methods['mle']['bias']['mu_x'],
                           methods['mle']['bias']['mu_y'],
                           methods['mle']['bias']['sigma_x'],
                           methods['mle']['bias']['sigma_y'],
                           methods['mle']['bias']['rho']],
        'Смещение (ММ)': [methods['mm']['bias']['mu_x'],
                          methods['mm']['bias']['mu_y'],
                          methods['mm']['bias']['sigma_x'],
                          methods['mm']['bias']['sigma_y'],
                          methods['mm']['bias']['rho']],
        'Смещение (Bootstrap NP)': [methods['bootstrap_nonparametric']['bias']['mu_x'],
                                    methods['bootstrap_nonparametric']['bias']['mu_y'],
                                    methods['bootstrap_nonparametric']['bias']['sigma_x'],
                                    methods['bootstrap_nonparametric']['bias']['sigma_y'],
                                    methods['bootstrap_nonparametric']['bias']['rho']],
        'Смещение (Bootstrap P)': [methods['bootstrap_parametric']['bias']['mu_x'],
                                   methods['bootstrap_parametric']['bias']['mu_y'],
                                   methods['bootstrap_parametric']['bias']['sigma_x'],
                                   methods['bootstrap_parametric']['bias']['sigma_y'],
                                   methods['bootstrap_parametric']['bias']['rho']]
    })

    time_df = pd.DataFrame({
        'Метод': ['ММП', 'Метод моментов', 'Bootstrap (NP)', 'Bootstrap (P)'],
        'Среднее время (сек)': [results['avg_time']['mle'],
                                results['avg_time']['mm'],
                                results['avg_time']['bootstrap_nonparametric'],
                                results['avg_time']['bootstrap_parametric']]
    })
    
    return {
        'estimates': estimates_df,
        'mse': mse_df,
        'bias': bias_df,
        'time': time_df
    }



