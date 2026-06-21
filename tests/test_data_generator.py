import pytest
import numpy as np 
import pandas as pd 
import sys, os 
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from src.data_generator import MultivariateNormalGenerator

class TestMultivariateNormalGenerator:
    """Тесты для генератора двумерного нормального распределения"""
    def test_initialization_valid(self):
        # проверка инициализации данных 
        gen = MultivariateNormalGenerator(0, 0, 1, 1, 0.5)
        assert gen.mu_x == 0 # assert это утверждение что данные валидны 
        assert gen.mu_y == 0
        assert gen.sigma_x == 1
        assert gen.sigma_y == 1
        assert gen.rho == 0.5

    def test_initialization_invalid_sigma_x(self):
        # Отрицательная сигма вызовет ошибку 
        with pytest.raises(ValueError, match="σ_x < 0"):
            MultivariateNormalGenerator(0, 0, -1, 1, 0)

    def test_initialization_invalid_sigma_y(self):
        # Отрицательная сигма вызовет ошибку 
        with pytest.raises(ValueError, match="σ_y < 0"):
            MultivariateNormalGenerator(0, 0, 1, -1, 0)

    def test_initialization_invalid_rho(self):
        # корреляция Пирсона больше единицы выдаст ошибку 
        with pytest.raises(ValueError):
            MultivariateNormalGenerator(0, 0, 1, 1, 2)

    def test_initialization_invalid_rho_negative(self):
        # корреляция Пирсона меньше минус единицы выдаст ошибку 
        with pytest.raises(ValueError):
            MultivariateNormalGenerator(0, 0, 1, 1, -2)

    def test_build_covariance_matrix(self):
        # проверяем эталонную ковариационную матрицу в двумерном нормально распределении 
        gen = MultivariateNormalGenerator(0, 0, 2, 3, 0.7) # инициализация 
        cov = gen.covariance_matrix
        
        expected = np.array([ # матрица которую ожидаем 
            [4.0, 4.2],
            [4.2, 9.0]
        ])
        
        np.testing.assert_array_almost_equal(cov, expected, decimal=10)


    def test_cholesky_decomposition(self):
        """Тест: разложение Холецкого"""
        gen = MultivariateNormalGenerator(0, 0, 2, 3, 0.7)
        L = gen.cholesky_l
        
        # Проверка, что L * L^T = cov_matrix
        reconstructed = L @ L.T
        np.testing.assert_array_almost_equal(reconstructed, gen.covariance_matrix, decimal=10)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])