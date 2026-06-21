import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2, norm
from typing import Tuple

class PlotsForGeneratedData: 
    def __init__(self, df: pd.DataFrame, mu_x: float, mu_y: float, sigma_x: float, sigma_y: float, rho: float):
        self.df = df
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rho = rho

        self.cov_matrix = np.array([
            [sigma_x**2, rho * sigma_x * sigma_y],
            [rho * sigma_x * sigma_y, sigma_y**2]
        ])

        # Обратная ковариационная матрица
        self.cov_inv = np.linalg.inv(self.cov_matrix)
        
    def _ellipse_points(self, confidence: float = 0.95, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        # внутрення функция для использования внутри класса 
        """
            Вычисляет точки эллипса постоянной плотности для заданного уровня доверия
            
            Уравнение: (x - μ)^T Σ^{-1} (x - μ) = χ²_{2}(p)
            
            Возвращает:
            -----------
            (x_points, y_points) : координаты эллипса
        """
        # Квантиль хи-квадрат
        chi_2_value = chi2.ppf(confidence, df=2)
        
        # массив равномерно распределенных углов в радианах 
        theta = np.linspace(0, 2 * np.pi, n_points)

        # Собственные значения и векторы ковариационной матрицы
        eigvals, eigvecs = np.linalg.eig(self.cov_matrix)

        # Длины полуосей эллипса
        a = np.sqrt(chi_2_value * eigvals[0])  # большая полуось
        b = np.sqrt(chi_2_value * eigvals[1])  # малая полуось

        # Точки эллипса в системе координат собственных векторов
        ellipse_points = np.array([a * np.cos(theta), b * np.sin(theta)])

        # Поворот эллипса 
        rotated_points = eigvecs @ ellipse_points

        x_points = rotated_points[0, :] + self.mu_x
        y_points = rotated_points[1, :] + self.mu_y
        
        return x_points, y_points
    

    def plot_marginal_histogram(self, figsize: Tuple[int, int] = (6, 3), 
                                  bins: int = 30): 
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.hist(self.df['X'], bins=bins, density=True, alpha=0.6, 
                color='steelblue', edgecolor='black')
        
        # Наложение нормальной кривой для X
        x_range = np.linspace(self.df['X'].min(), self.df['X'].max(), 100)
        x_pdf = norm.pdf(x_range, self.mu_x, self.sigma_x)
        ax1.plot(x_range, x_pdf, 'r-', linewidth=2, label='Теоретическая N(μ, σ)')
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Плотность', fontsize=12)
        ax1.set_title(f'Маргинальное распределение X\n(μ_x={self.mu_x}, σ_x={self.sigma_x})', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)


        # Гистограмма для Y
        ax2.hist(self.df['Y'], bins=bins, density=True, alpha=0.6,
                color='seagreen', edgecolor='black')
        
        # Наложение нормальной кривой для Y
        y_range = np.linspace(self.df['Y'].min(), self.df['Y'].max(), 100)
        y_pdf = norm.pdf(y_range, self.mu_y, self.sigma_y)
        ax2.plot(y_range, y_pdf, 'r-', linewidth=2, label='Теоретическая N(μ, σ)')
        ax2.set_xlabel('Y', fontsize=12)
        ax2.set_ylabel('Плотность', fontsize=12)
        ax2.set_title(f'Маргинальное распределение Y\n(μ_y={self.mu_y}, σ_y={self.sigma_y})', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig
    
    
    
    def plot_joint_with_marginals(self, figsize: Tuple[int, int] = (6, 3)) -> plt.Figure:

        
        # Используем seaborn для jointplot, но возвращаем figure
        g = sns.jointplot(data=self.df, x='X', y='Y', kind='scatter',
                         marginal_kws=dict(bins=30, fill=True),
                         height=figsize[1], ratio=5)
        
        # Настройка внешнего вида
        g.ax_joint.set_xlabel('X', fontsize=12)
        g.ax_joint.set_ylabel('Y', fontsize=12)
        g.figure.suptitle('Совместное распределение с маргинальными гистограммами', fontsize=14, y=1.05)
        
        # Добавляем эллипс на основной график
        x_ell, y_ell = self._ellipse_points(confidence=0.95)
        g.ax_joint.plot(x_ell, y_ell, 'r-', linewidth=2, label='95% эллипс')
        g.ax_joint.plot(self.mu_x, self.mu_y, 'r*', markersize=12, label='Истинный центр')
        g.ax_joint.legend()
        
        return g.figure


    
def create_visualizations(df: pd.DataFrame, mu_x: float, mu_y: float,
                sigma_x: float, sigma_y: float, rho: float) -> dict:
    
    
    visual = PlotsForGeneratedData(df, mu_x, mu_y, sigma_x, sigma_y, rho)
    
    return {
        'marginal_histogram': visual.plot_marginal_histogram(),
        'scatter_with_marginals': visual.plot_joint_with_marginals()
    }