import pytest
import numpy as np
import pandas as pd
import sys, os 
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from src.statistical_tests import (
    shapiro_wilk_test,
    shapiro_wilk_test_summary,
    qq_plot
)
from src.data_generator import MultivariateNormalGenerator

class TestStatisticalTests:
    def test_shapiro_normal_data(self):
        """Тесты для статистических тестов"""
        gen = MultivariateNormalGenerator(0, 0, 1, 1, 0) # иницилизируем данные для генерации 
        df = gen.generate(500, random_seed=42) #  генерация 

        # проверка функции Шапиро Уилка в модуле statistical_tests

        result = shapiro_wilk_test(df)

        assert result[result['Переменная'] == 'X']['p-value'].iloc[0] > 0.05
        assert result[result['Переменная'] == 'Y']['p-value'].iloc[0] > 0.05

    def test_shapiro_non_normal_data(self):
        """Тест: тест Шапиро-Уилка на не нормальных данных"""
        np.random.seed(42)
        df = pd.DataFrame({
            'X': np.random.exponential(scale=1, size=500),
            'Y': np.random.exponential(scale=1.5, size=500)
        })
        
        result = shapiro_wilk_test(df)
        
        # Для не нормальных данных p-value должно быть < 0.05
        assert result[result['Переменная'] == 'X']['p-value'].iloc[0] < 0.05
        assert result[result['Переменная'] == 'Y']['p-value'].iloc[0] < 0.05

    def test_shapiro_summary(self):
        """Тест: сводка теста Шапиро-Уилка"""
        gen = MultivariateNormalGenerator(0, 0, 1, 1, 0)
        df = gen.generate(500, random_seed=42)
        
        summary = shapiro_wilk_test_summary(df)
        
        assert 'X' in summary
        assert 'Y' in summary
        assert 'statistic' in summary['X']
        assert 'p_value' in summary['X']
        assert 'is_normal' in summary['X']


    def test_qq_plot_returns_figure(self):
        """Тест: Q-Q plot возвращает объект Figure"""
        gen = MultivariateNormalGenerator(0, 0, 1, 1, 0)
        df = gen.generate(500, random_seed=42)
        
        fig = qq_plot(df)
        
        assert fig is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])