import logging # библиотека для логирования результатов 
import numpy as np 
from typing import Optional
import pandas as pd 

# Выведем подсчет определителя ковариационной матрицы 

logging.basicConfig(
    filename="matrix_validation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

class MultivariateNormalGenerator:
    """
    Генератор двумерного нормального распределения.
    
    Использует разложение Холецкого для генерации коррелированных
    нормальных величин.
    
    Attributes:
        mu_x, mu_y: Математические ожидания
        sigma_x, sigma_y: Стандартные отклонения
        rho: Коэффициент корреляции
    """

    # константы класса 
    EPSILON = 1e-10 

    def __init__(self, mu_x: float, mu_y: float, sigma_x: float, sigma_y: float, rho: float): 
        """
        Инициализация генератора.
        
        Args:
            mu_x, mu_y: Средние значения
            sigma_x, sigma_y: Стандартные отклонения (>0)
            rho: Коэффициент корреляции ([-1, 1])
        """

        self.mu_x = mu_x
        self.mu_y = mu_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rho = rho


        self._validate_parameters(sigma_x, sigma_y, rho, self.EPSILON) # проверка корректности 

        self.covariance_matrix = self._build_covariance_matrix() # Ковариационная матрица 
        self.cholesky_l = self._cholesky_decomposition(self.EPSILON) # разложение Холецкого 

        # Логирование успешного создания
        logging.info(f"Создан генератор: μ=({mu_x}, {mu_y}), "
                    f"σ=({sigma_x}, {sigma_y}), ρ={rho}")

    def _validate_parameters(self, sigma_x: float, sigma_y: float, rho: float, epsilon: float) -> None:
        """Проверка корректности"""

        # Стандартные отклонения должны быть положительными
        if sigma_x <= 0: 
            msg = f"σ_x < 0: {sigma_x}"
            logging.error(msg)
            raise ValueError(msg)
            
        
        if sigma_y <= 0: 
            msg = f"σ_y < 0: {sigma_y}"
            logging.error(msg)
            raise ValueError(msg)
        
        # коэффициент корреляции должен быть от -1 до 1
        if not (-1.0 <= rho <= 1.0):
            msg = f"ρ не в пределах -1 и 1: {rho}"
            logging.warning(msg)
            raise ValueError(msg)
        
        # посчитаем определитель ковариационно матрицы 
        # Также проверка положительной определенности
        determinant = (sigma_x ** 2) * (sigma_y ** 2) * (1 - rho ** 2)


        if determinant <= epsilon: 
            if abs(abs(rho) - 1.0) < epsilon: 
                msg = f"Предупреждение: ρ = {abs(rho):.3f}, матрица вырожденная (|Σ| = {determinant:.2e}). Точки лежат на прямой"
                print(msg) # Выводим в консоль 
                logging.warning(msg) # записываем в логи 
            else:
                msg = f"Ошибка: Матрица не является положительно определённой. |Σ| = {determinant:.2e}. σ_x={sigma_x}, σ_y={sigma_y}, ρ={rho}"
                logging.error(msg)   # запишет ошибку в файл лога
                raise ValueError(msg)
        else: 
            msg = f"Проверка успешна: Матрица ковариации корректна."
            print(msg) # Выводим в консоль 
            logging.info(msg) # записываем в логи 

    def _build_covariance_matrix(self) -> np.ndarray:
        """Строим ковариационную матрицу """
        
        cov_xy = self.rho * self.sigma_x * self.sigma_y

        return np.array([
            [self.sigma_x ** 2, cov_xy],
            [cov_xy, self.sigma_y ** 2]
        ])
    

    def _cholesky_decomposition(self, epsilon: float) -> np.ndarray:
        """
        
        L - нижняя треугольная матрица:
        L = [[a, 0],
             [b, c]]
        
        где:
        a = σ_x
        b = ρ·σ_y
        c = σ_y·√(1 - ρ²)
        
        Для вырожденного случая (ρ = ±1): c = 0
        """

        a = self.sigma_x

        b = self.rho * self.sigma_y


        # Вычисляем c с защитой от отрицательных значений
        determinant_factor = max(0, 1 - self.rho ** 2)
        c = self.sigma_y * np.sqrt(determinant_factor)

        if c < epsilon: 
            msg = f"  Примечание: c = {c:.2e} (практически вырожденная матрица)"
            logging.warning(msg)

        # возвращае матрицу Холевского 
        return np.array([
            [a, 0.0],
            [b, c]
        ])

    def generate(self, n_samples: int, random_seed: Optional[int] = None) -> pd.DataFrame:

        if n_samples <= 0: 
            msg = f"Размер выборки должен быть > 0, получено {n_samples}"
            logging.warning(msg)
            raise ValueError(msg)
        
        # Устанавливаем seed для воспроизводимости
        if random_seed is not None:
            np.random.seed(random_seed)
            logging.info(f"Установлен random_seed = {random_seed}")

        # Генерация независимых Z ~ N(0, 1)
        # генерируем двухмерный массив распределенный по Гауссовскому закону нормального распределения 
        z = np.random.normal(0, 1, size=(n_samples, 2))

        mu_vector = np.array([self.mu_x, self.mu_y]) # Для переноса облака точек 

        generated = (self.cholesky_l @ z.T).T + mu_vector

        df = pd.DataFrame(generated, columns=['X', 'Y'])


        return df 
    
    def get_parametrs(self) -> dict: 
        """Возвращает параметры распределения в виде словаря."""
        return {
            'mu': (self.mu_x, self.mu_y),
            'sigma': (self.sigma_x, self.sigma_y),
            'rho': self.rho,
            'covariance_matrix': self.covariance_matrix,
            'cholesky_L': self.cholesky_l,
            'determinant': np.linalg.det(self.covariance_matrix)
        }