
from dataclasses import dataclass  # декоратор для дата класса в дашборд
import seaborn as sns
import streamlit as st  # библиотека для создания дашборда
import pandas as pd
from math_content import MATH_MARKDOWN_INTIAL, MATH_MARKDOWN_EVALUATION, MATH_MARKDOWN_GENERATE_DATA
from src.data_generator import MultivariateNormalGenerator
from src.visual import create_visualizations
import os 
import numpy as np 
from src.evaluation.mle_estimator import quick_mle, plot_standard_errors
from src.evaluation.mm_estimator import quick_mm, plot_mm_standrad_errors
from src.evaluation.bootstrap import quick_bootstrap, plot_bootstrap
from src.comparison import run_comparison_simulation, comparison_to_dataframe
from src.statistical_tests import quick_statistical_tests


sns.set_theme(style="whitegrid", palette="muted") # настройка внешнего вида 

@dataclass # декоратор 
class DashboardConfig:
    mu_x: float = 0.0
    mu_y: float = 0.0
    sigma_x: float = 1.0
    sigma_y: float = 1.0
    rho: float = 0.0
    sample_size: int = 500
    bootstrap_iterations: int = 100
    confidence_level: float = 0.95
    estimation_choice: str = "Метод максимального правдоподобия",
    experiments_iterations: int = 100


    def to_dataframe(self) -> pd.DataFrame:
        """Преобразует параметры для dataframe"""
        return pd.DataFrame({
            "Параметр": ["μ_x", "μ_y", "σ_x", "σ_y", "ρ"],
            "Значение": [self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho]
        })
    
    @property
    def covariance_matrix(self) -> pd.DataFrame:
        """Возвращает ковариационную матрицу 2x2"""
        return pd.DataFrame(
            [[self.sigma_x**2, self.rho * self.sigma_x * self.sigma_y],
             [self.rho * self.sigma_x * self.sigma_y, self.sigma_y**2]],
            index=["X", "Y"],
            columns=["X", "Y"]
        )  

    @property # метод класса в свойство 
    def params_dict(self) -> dict:
        """Возвращает параметры в виде словаря"""
        return {
            "μ_x": self.mu_x,
            "μ_y": self.mu_y,
            "σ_x": self.sigma_x,
            "σ_y": self.sigma_y,
            "ρ": self.rho,
            "Размер выборки": self.sample_size,
            "Bootstrap итераций": self.bootstrap_iterations,
            "Уровень доверия": f"{self.confidence_level*100:.0f}%",
            "Метод оценивания": self.estimation_choice,
            "Количество экспериментов": self.experiments_iterations
        }  


def _sidebar_controls() -> DashboardConfig:
    """Вынос логики sidebar в отдельную функцию"""
    with st.sidebar: 
        try: 
            st.header("Параметры распределения для многомерного нормального распределения")

            mu_x = st.slider("μ_x (среднее по X)", min_value=-5.0, max_value=5.0,
            value=0.0, step=0.1,help="Математическое ожидание для координаты X")
                
            mu_y = st.slider("μ_y (среднее по Y)", min_value=-5.0, max_value=5.0,
            value=0.0, step=0.1,help="Математическое ожидание для координаты Y")
 
            sigma_x = st.slider("σ_x (стандартное отклонение по X)", min_value=0.1, max_value=5.0,
            value=1.0, step=0.1,help="Стандартное отклонение для X")

            sigma_y = st.slider("σ_y (стандартное отклонение по Y)", min_value=0.1, max_value=5.0,
            value=1.0, step=0.1,help="Стандартное отклонение для Y")

            st.divider()

            rho = st.slider("ρ (коэффициент корреляции)", min_value=-1.0, max_value=1.0,
            value=0.0, step=0.05,help="Корреляция между X и Y, значения от -1 до 1")

            st.divider()

            sample_size = st.number_input("Размер выборки (n):", 
                    min_value=100, max_value=1000, value=100, step=10, help="Для корректной работы асимптотических методов рекомендуется n ≥ 30")
            

            confidence_level = st.select_slider(
                "Уровень доверия для интервала:",
                options=[0.80, 0.90, 0.95, 0.99],
                value=0.95,
                help="Процент симуляций, попадающих в интервал"
            )

            bootstrap_iterations = st.number_input(
                "Bootstrap итераций (B)",
                min_value=50, max_value=500,
                value=100, step=50,
                help="Количество псевдовыборок для bootstrap"
            )

            experiments_iterations = st.number_input(
                "Количество итераций для сравнения методов оценивания",
                min_value=50, max_value=500,
                value=100, step=50
            )

            st.divider()

            estimation_choice = st.selectbox( # выбор распределения 
                "Выберите метод оценивания:",
                ["Метод максимального правдоподобия", "Метод моментов", "Bootstrap"],
                help="Метод оценки параметров распределения"
            )

            st.divider()
            
            st.info(
                "**Подсказка**: Изменяя параметры можно наблюдать, "
                "как меняется форма распределения и качество оценок."
            )

            st.divider()

            show_logs = st.checkbox("Показывать логи", value=False)
            if show_logs:
                with st.expander("Логи выполнения"):
                    if os.path.exists("matrix_validation.log"):
                        with open("matrix_validation.log", "r") as f:
                            st.text(f.read()[-2000:])

            return DashboardConfig( # возвращаем дата класс 
                mu_x, mu_y, sigma_x, sigma_y, rho, 
                sample_size, bootstrap_iterations, 
                confidence_level, estimation_choice, experiments_iterations
                )
        
        except Exception as e: 
            st.warning(f"Ошибка в выборе параметров: {e}")
            return DashboardConfig() # возвращаем стандартые значения 
        
def dashboard() -> DashboardConfig:
    try: 
        st.set_page_config(
            page_title="Многомерное нормальное распределение",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title("Интерактивный анализ многомерного нормального распределения")
        st.markdown("---")
      
        return _sidebar_controls()

    except Exception as e: 
        st.warning(f"Ошибка в основной функции дашборда: {e}")
        return DashboardConfig() # возвращаем стандартые значения 

if __name__ == "__main__": 
    config = dashboard()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Математический аппарат", 
                                "Данные и описательные статистики",
                                "Сгенерированные данные", "Методы сравнения", "Сравнительный анализ", "Тесты"])

    with tab1: 
        st.markdown(MATH_MARKDOWN_INTIAL) 
        st.markdown(MATH_MARKDOWN_EVALUATION) 

    with tab2: 
        # Карточка с параметрами
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("### Параметры распределения")
                st.dataframe(config.to_dataframe(), hide_index=True, use_container_width=True)

        with col2: 
            st.markdown("### Настройки анализа")
            st.metric("Размер выборки", config.sample_size)
            st.metric("Bootstrap итераций", config.bootstrap_iterations)
            st.metric("Уровень доверия", f"{config.confidence_level*100:.0f}%")
            st.metric("Количество итераций эксперимента", f"{config.experiments_iterations}")
        with st.container(border=True):
            st.markdown("### Ковариационная матрица")
            st.dataframe(config.covariance_matrix.style.format("{:.3f}"), use_container_width=True)

        st.success(f"Выбранный метод: **{config.estimation_choice}**")

    with tab3: 
        st.markdown("### Сгенерированные данные")

        # Кэширование генератора для производительности
        @st.cache_resource
        def get_generator(config): 
            generator = MultivariateNormalGenerator(
                mu_x=config.mu_x,
                mu_y=config.mu_y,
                sigma_x=config.sigma_x,
                sigma_y=config.sigma_y,
                rho=config.rho
            )

            return generator.generate(n_samples=config.sample_size)
    

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            if st.button("Перегенерировать данные", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()


        # Генерация данных
        df = get_generator(config)

        with st.container(border=True):
            st.dataframe(df.describe().round(4))

        col1_gen, col2_gen = st.columns(2)

        with col1_gen: 
            with st.container(border=True):
                st.metric("Среднее X", f"{df['X'].mean():.4f}")
                st.metric("Дисперсия X", f"{df['X'].var():.4f}")
                st.metric("Ст. отклонение X", f"{df['X'].std():.4f}")
        with col2_gen:
            with st.container(border=True):
                st.metric("Среднее Y", f"{df['Y'].mean():.4f}")
                st.metric("Дисперсия Y", f"{df['Y'].var():.4f}")
                st.metric("Ст. отклонение Y", f"{df['Y'].std():.4f}")

        st.subheader("Корреляционный анализ")

        with st.container(border=True):
            corr_pearson = df['X'].corr(df['Y'], method='pearson')
            st.metric("Коэффициент корреляции Пирсона", f"{corr_pearson:.4f}")

            cov_xy = df['X'].cov(df['Y'])
            st.metric("Ковариация", f"{cov_xy:.4f}")

        with st.container(border=True):
                st.markdown(MATH_MARKDOWN_GENERATE_DATA) 
        
        subtab1, subtab2 = st.tabs(["Маргинальные гистограммы для X и Y с наложенной нормальной кривой", "scatter + маргинальные гистограммы"])  
        with subtab1: 
            st.subheader("Маргинальные гистограммы для X и Y с наложенной нормальной кривой")

            figs = create_visualizations(df, config.mu_x, config.mu_y, 
                        config.sigma_x, config.sigma_y, config.rho) # Добавляем параметры 
            
            st.pyplot(figs['marginal_histogram'])


        with subtab2: 
            st.subheader("scatter + маргинальные гистограммы")

            st.pyplot(figs['scatter_with_marginals'])

    with tab4: 

        if config.estimation_choice == "Метод максимального правдоподобия": 
            estimates, ci, fisher, fisher_matrix, fisher_sigma, fisher_inv, fisher_errors, fisher_estimation_precision = quick_mle(df, config.confidence_level) # для ММП

            with st.container(border=True):

                st.subheader("Точечные оценки параметров")

                col1, col2, col3, col4, col5 = st.columns(5)
    
                col1.metric(label="μ_x", value=f"{estimates['mu_x']:.4f}")
                col2.metric(label="μ_y", value=f"{estimates['mu_y']:.4f}")
                col3.metric(label="σ_x", value=f"{estimates['sigma_x']:.4f}")
                col4.metric(label="σ_y", value=f"{estimates['sigma_y']:.4f}")
                col5.metric(label="ρ (rho)", value=f"{estimates['rho']:.4f}")

            with st.container(border=True):

                st.subheader("Доверительные интервалы для параметров")
                st.dataframe(ci, use_container_width=True)

            with st.container(border=True):

                st.subheader("Расчет для коэффициента корреляции Пирсона через преобразование Фишера")
                st.dataframe(fisher, use_container_width=True)

            with st.container(border=True):

                st.subheader("Наблюдаемая информационная матрица Фишера")
                st.dataframe(fisher_matrix, use_container_width=True)

            with st.container(border=True):

                st.subheader("Информационная матрица Фишера для блока ковариации")
                st.dataframe(fisher_sigma, use_container_width=True)

            with st.container(border=True):
                
                st.subheader("Обратная информационная матрица Фишера")
                st.dataframe(fisher_inv, use_container_width=True)

            with st.container(border=True):
                
                st.subheader("Связь информации Фишера и доверительных интервалов")
                st.dataframe(fisher_errors, use_container_width=True)

            with st.container(border=True):
                st.subheader("Точность оценок")

    
                # Цветовое форматирование
                def color_precision(val):
                    if '🟢' in str(val):
                        return 'background-color: #d4edda'  # светло-зеленый
                    elif '🟡' in str(val):
                        return 'background-color: #fff3cd'  # светло-желтый
                    elif '🔴' in str(val):
                        return 'background-color: #f8d7da'  # светло-красный
                    return ''
    
                st.dataframe(
                    fisher_estimation_precision.style.map(color_precision, subset=['Accuracy'])
                    .format({
                        'estimation': '{:.4f}',
                        'SE': '{:.4f}',
                        'Relative_error': '{:.2f}',
                        'CI_width': '{:.4f}'
                    }),
                    use_container_width=True
                )
    
                # Дополнительное пояснение
                st.info("""
                **Интерпретация точности:**
                - 🟢 **Высокая** (< 5%): оценка очень точная
                - 🟡 **Средняя** (5-15%): оценка умеренно точная
                - 🔴 **Низкая** (> 15%): оценка неточная, увеличьте размер выборки
                """)

            st.subheader("Визуализация точности оценок")
            with st.container(border=True):
                fig = plot_standard_errors(df)
                st.pyplot(fig)

        if config.estimation_choice == "Метод моментов": 
            estimates_mm, ci_mm, fisher_mm, fisher_errors_mm, estimate_precision  = quick_mm(df, config.confidence_level) 

            with st.container(border=True):

                st.subheader("Точечные оценки параметров")

                col1, col2, col3, col4, col5 = st.columns(5)
    
                col1.metric(label="μ_x", value=f"{estimates_mm['mu_x']:.4f}")
                col2.metric(label="μ_y", value=f"{estimates_mm['mu_y']:.4f}")
                col3.metric(label="σ_x", value=f"{estimates_mm['sigma_x']:.4f}")
                col4.metric(label="σ_y", value=f"{estimates_mm['sigma_y']:.4f}")
                col5.metric(label="ρ (rho)", value=f"{estimates_mm['rho']:.4f}")

            with st.container(border=True):

                st.subheader("Доверительные интервалы для параметров")
                st.dataframe(ci_mm, use_container_width=True)

            with st.container(border=True):

                st.subheader("Расчет для коэффициента корреляции Пирсона через преобразование Фишера для метода моментов")
                st.dataframe(fisher_mm, use_container_width=True)

            with st.container(border=True):
                
                st.subheader("Стандартные ошибки оценок метод моментов ")
                st.dataframe(fisher_errors_mm, use_container_width=True)

            with st.container(border=True):
                st.subheader("Точность оценок")

    
                # Цветовое форматирование
                def color_precision(val):
                    if '🟢' in str(val):
                        return 'background-color: #d4edda'  # светло-зеленый
                    elif '🟡' in str(val):
                        return 'background-color: #fff3cd'  # светло-желтый
                    elif '🔴' in str(val):
                        return 'background-color: #f8d7da'  # светло-красный
                    return ''
    
                st.dataframe(
                    estimate_precision.style.map(color_precision, subset=['Accurancy_mm'])
                    .format({
                        'estimate_mm': '{:.4f}',
                        'SE_mm': '{:.4f}',
                        'Relative_errors_mm': '{:.2f}',
                        'CI_mm': '{:.4f}'
                    }),
                    use_container_width=True
                )
    
                # Дополнительное пояснение
                st.info("""
                **Интерпретация точности:**
                - 🟢 **Высокая** (< 5%): оценка очень точная
                - 🟡 **Средняя** (5-15%): оценка умеренно точная
                - 🔴 **Низкая** (> 15%): оценка неточная, увеличьте размер выборки
                """)

            st.subheader("Визуализация стандартных ошибок оценок метода моментов")
            with st.container(border=True):
                fig_mm = plot_mm_standrad_errors(df)
                st.pyplot(fig_mm)

        if config.estimation_choice == "Bootstrap": 
            bootstrap_results = quick_bootstrap(
                df, 
                B=config.bootstrap_iterations,
                random_seed=42
            )
            bootstrap_nonparametric = bootstrap_results['nonparametric']
            bootstrap_parametric = bootstrap_results['parametric']

            conf_level = config.confidence_level
            if conf_level > 1:  
                conf_level /= 100

            alpha = 1 - config.confidence_level
            lower_pct = (alpha / 2) * 100
            upper_pct = (1 - alpha / 2) * 100

            with st.container(border=True):
                st.subheader(f"Результаты {config.bootstrap_iterations} итераций Bootstrap из неоцененного распределения")
                summary_data = []

                
                for param, values in bootstrap_nonparametric.items():
                    if param in ['B', 'method']: 
                        continue
                    mean_val = np.mean(values)
                    std_err = np.std(values)
                    median_val = np.median(values)
                    ci_lower = np.percentile(values, lower_pct)
                    ci_upper = np.percentile(values, upper_pct)

                    summary_data.append({
                        "Параметр": param,
                        "Среднее": f"{mean_val:.4f}",
                        "Медиана": f"{median_val:.4f}",
                        "Стандартная ошибка (SE)": f"{std_err:.4f}",
                        f"{int(config.confidence_level * 100)}% Дов. интервал": f"[{ci_lower:.4f}, {ci_upper:.4f}]"
                    })

                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

            with st.container(border=True):
                st.subheader(f"Результаты {config.bootstrap_iterations} итераций Bootstrap из оцененного распределения")
                summary_data_parametric = []

                for param, values in bootstrap_parametric.items():
                    if param in ['B', 'method']: 
                        continue
                    mean_val = np.mean(values)
                    std_err = np.std(values)
                    median_val = np.median(values)
                    ci_lower = np.percentile(values, lower_pct)
                    ci_upper = np.percentile(values, upper_pct)

                    summary_data_parametric.append({
                        "Параметр": param,
                        "Среднее": f"{mean_val:.4f}",
                        "Медиана": f"{median_val:.4f}",
                        "Стандартная ошибка (SE)": f"{std_err:.4f}",
                        f"{int(config.confidence_level * 100)}% Дов. интервал": f"[{ci_lower:.4f}, {ci_upper:.4f}]"
                    })

                st.dataframe(pd.DataFrame(summary_data_parametric), use_container_width=True, hide_index=True)

            with st.container(border=True):
                fig_bootstrap = plot_bootstrap(bootstrap_nonparametric, confidence_level=conf_level)
                st.pyplot(fig_bootstrap)
    with tab5: 
        with st.container(border=True):

            st.subheader("Сравнительный анализ методов оценивания")
            
            with st.spinner("Запуск симуляции сравнения..."):
                comparison_results = run_comparison_simulation(
                true_params={
                    'mu_x': config.mu_x,
                    'mu_y': config.mu_y,
                    'sigma_x': config.sigma_x,
                    'sigma_y': config.sigma_y,
                    'rho': config.rho
                },
                n_samples=config.sample_size,
                n_experiments=config.experiments_iterations,  
                bootstrap_B=config.bootstrap_iterations,
                confidence_level=config.confidence_level,
                random_seed=42
                )

                dfs = comparison_to_dataframe(comparison_results)

                st.markdown("### Сравнение оценок")
                st.dataframe(dfs['estimates'], use_container_width=True)
                
                st.markdown("### Сравнение MSE")
                st.dataframe(dfs['mse'], use_container_width=True)
                
                st.markdown("### Сравнение смещения (bias)")
                st.dataframe(dfs['bias'], use_container_width=True)
                
                st.markdown("### Время выполнения")
                st.dataframe(dfs['time'], use_container_width=True)
    with tab6: 
        st.subheader("Тест Шапиро-Уилка и Q-Q визуализация")

        stats_results = quick_statistical_tests(df)

        st.markdown("### Результаты теста")
        st.dataframe(stats_results['shapiro_results'], use_container_width=True)

        st.markdown("### Результаты теста")
        st.dataframe(stats_results['shapiro_summary'], use_container_width=True)

        st.markdown("---")
        st.markdown("### Q-Q plots")
        st.pyplot(stats_results['qq_plot'])





        