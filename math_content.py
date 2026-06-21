"""Содержимое для вкладки с математическим аппаратом"""

MATH_MARKDOWN_INTIAL = """
### 1. Плотность двумерного нормального распределения

$$ f(\\mathbf{x}) = \\frac{1}{2\\pi\\sqrt{|\\Sigma|}} \\exp\\left( -\\frac{1}{2}(\\mathbf{x} - \\boldsymbol{\\mu})^T \\Sigma^{-1} (\\mathbf{x} - \\boldsymbol{\\mu}) \\right) $$

где:

* $ \\mathbf{x} = (x, y)^T $
* $ \\boldsymbol{\\mu} = (\\mu_x, \\mu_y)^T $ — вектор средних
* $ \\Sigma $ — ковариационная матрица $ 2 \\times 2 $

---

### 2. Ковариационная матрица

$$ \\Sigma = \\begin{bmatrix} \\sigma_x^2 & \\rho\\sigma_x\\sigma_y \\\\ \\rho\\sigma_x\\sigma_y & \\sigma_y^2 \\end{bmatrix} $$

где:

* $ \\sigma_x^2, \\sigma_y^2 $ — дисперсии
* $ \\rho $ — коэффициент корреляции ($ |\\rho| \\leq 1 $)

### Математическая проверка ковариационной матрицы

Определитель (детерминант) рассчитывается по формуле:
$$ |\\Sigma| = \\sigma_x^2 \\sigma_y^2 (1 - \\rho^2) $$


### Критерии проверки матрицы:

1. **Положительно определённая матрица ($ |\\Sigma| > 0 $)**
   $$ \\sigma_x > 0, \\quad \\sigma_y > 0, \\quad |\\rho| < 1 $$
   * Распределение корректно. Существует обратная матрица $ \\Sigma^{-1} $. Точки образуют двумерное эллиптическое облако.

2. **Вырожденная матрица ($ |\\Sigma| = 0 $)**
   $$ |\\rho| = 1 \\quad (\\rho = 1 \\text{ или } \\rho = -1) $$
   * Распределение вырождается в одномерное. Случайные величины связаны строгой линейной зависимостью, все сгенерированные точки будут лежать на одной прямой.

3. **Невалидная матрица ($ |\\Sigma| < 0 $)**
   $$ |\\rho| > 1 $$
   * Математически невозможная ситуация. Коэффициент корреляции не может превышать единицу по модулю. Приводит к делению на ноль или извлечению корня из отрицательного числа при расчёте плотности.

---  

### 3. Параметры распределения (всего 5 параметров):

* $ \\boldsymbol{\\theta} = (\\mu_x, \\mu_y, \\sigma_x, \\sigma_y, \\rho) $

---
                    
### 4. Генерация точек (сэмплирование)    
        
#### Метод: Разложение Холецкого

* $ \\Sigma = L L^T $, где $ L $ — нижняя треугольная матрица разложения Холецкого
* $ \\mathbf{X} = \\boldsymbol{\\mu} + L \\mathbf{Z} $, где $ \\mathbf{Z} \\sim \\mathcal{N}(0, I) $ — вектор стандартных нормальных величин

**Матрица Холецкого:**

$$ L = \\begin{bmatrix} \\sigma_x & 0 \\\\ \\rho \\sigma_y & \\sigma_y \\sqrt{1 - \\rho^2} \\end{bmatrix} $$

**Покомпонентная генерация:**

$$ \\begin{aligned} X &= \\mu_x + \\sigma_x Z_1 \\\\ Y &= \\mu_y + \\rho \\sigma_y Z_1 + \\sigma_y \\sqrt{1 - \\rho^2} Z_2 \\end{aligned} $$
        

---

### 5. Визуализация

* Эллипсы постоянной плотности: $ (\\mathbf{x} - \\boldsymbol{\\mu})^T \\Sigma^{-1} (\\mathbf{x} - \\boldsymbol{\\mu}) = \\chi_2^2(p) $
* Маргинальные гистограммы (**marginal histograms**)


"""

MATH_MARKDOWN_EVALUATION = """
---

### Методы оценивания параметров

#### 1. Метод максимального правдоподобия (ММП)

1.1 Оценки параметров:

**Средние значения:**
$$ \\hat{\\mu}_x = \\frac{1}{n} \\sum_{i=1}^{n} x_i $$
$$ \\hat{\\mu}_y = \\frac{1}{n} \\sum_{i=1}^{n} y_i $$

**Центрированные данные:**
$$ x_{c,i} = x_i - \\hat{\\mu}_x $$
$$ y_{c,i} = y_i - \\hat{\\mu}_y $$

**Дисперсия:**
$$ \\hat{\\sigma}_x^{2,\\text{ММП}} = \\frac{1}{n} \\sum_{i=1}^{n} (x_i - \\hat{\\mu}_x)^2 $$
$$ \\hat{\\sigma}_y^{2,\\text{ММП}} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{\\mu}_y)^2 $$

**Стандартное отклонение:**
$$ \\hat{\\sigma}_x = \\sqrt{\\hat{\\sigma}_x^{2,\\text{ММП}}} $$
$$ \\hat{\\sigma}_y = \\sqrt{\\hat{\\sigma}_y^{2,\\text{ММП}}} $$

**Ковариация:**
$$ \\hat{\\text{cov}}_{xy}^{\\text{ММП}} = \\frac{1}{n} \\sum_{i=1}^{n} (x_i - \\hat{\\mu}_x)(y_i - \\hat{\\mu}_y) $$

**Корреляция:**
$$ \\hat{\\rho}^{\\text{ММП}} = \\frac{\\sum_{i=1}^{n} (x_i - \\hat{\\mu}_x)(y_i - \\hat{\\mu}_y)}{\\sqrt{\\sum_{i=1}^{n} (x_i - \\hat{\\mu}_x)^2 \\sum_{i=1}^{n} (y_i - \\hat{\\mu}_y)^2}} $$

**Стандартные ошибки оценок через информацию Фишера:**
$$ \\text{se}(\\hat{\\mu}_x) = \\frac{\\hat{\\sigma}_x}{\\sqrt{n}} $$
$$ \\text{se}(\\hat{\\mu}_y) = \\frac{\\hat{\\sigma}_y}{\\sqrt{n}} $$
$$ \\text{se}(\\hat{\\sigma}_x) = \\frac{\\hat{\\sigma}_x}{\\sqrt{2n}} $$
$$ \\text{se}(\\hat{\\sigma}_y) = \\frac{\\hat{\\sigma}_y}{\\sqrt{2n}} $$
$$ \\text{se}(\\hat{\\rho}) = \\frac{1 - \\hat{\\rho}^2}{\\sqrt{n}} $$

**Критическое значение (квантиль):**
$$ \\alpha = 1 - \\gamma $$
$$ z_{1 - \\alpha/2} = \\Phi^{-1} \\left(1 - \\frac{\\alpha}{2}\\right) $$

### 1.2 Информационная матрица Фишера

Для параметров двумерного нормального распределения 
$\\boldsymbol{\\theta} = (\\mu_x, \\mu_y, \\sigma_x, \\sigma_y, \\rho)$ 
информационная матрица Фишера имеет блочно-диагональную структуру:

$$ \\mathcal{I}(\\boldsymbol{\\theta}) = \\begin{pmatrix} \\mathcal{I}_{\\boldsymbol{\\mu}\\boldsymbol{\\mu}} & 0 \\\\ 0 & \\mathcal{I}_{\\boldsymbol{\\Sigma}\\boldsymbol{\\Sigma}} \\end{pmatrix} $$

Блок для средних:

$$ \\mathcal{I}_{\\boldsymbol{\\mu}\\boldsymbol{\\mu}} = \\begin{pmatrix} \\frac{n}{\\sigma_x^2(1-\\rho^2)} & \\frac{n\\rho}{\\sigma_x\\sigma_y(1-\\rho^2)} \\\\ \\frac{n\\rho}{\\sigma_x\\sigma_y(1-\\rho^2)} & \\frac{n}{\\sigma_y^2(1-\\rho^2)} \\end{pmatrix} $$

Блок для параметров ковариации:

$$ \\mathcal{I}_{\\boldsymbol{\\Sigma}\\boldsymbol{\\Sigma}} = \\begin{pmatrix} \\frac{2n}{\\sigma_x^2} & \\frac{2n\\rho^2}{\\sigma_x\\sigma_y} & -\\frac{2n\\rho}{\\sigma_x\\sqrt{1-\\rho^2}} \\\\ \\frac{2n\\rho^2}{\\sigma_x\\sigma_y} & \\frac{2n}{\\sigma_y^2} & -\\frac{2n\\rho}{\\sigma_y\\sqrt{1-\\rho^2}} \\\\ -\\frac{2n\\rho}{\\sigma_x\\sqrt{1-\\rho^2}} & -\\frac{2n\\rho}{\\sigma_y\\sqrt{1-\\rho^2}} & \\frac{n(1+\\rho^2)}{(1-\\rho^2)^2} \\end{pmatrix} $$

Обратная матрица Фишера (ковариационная матрица оценок):

$$ \\mathcal{I}^{-1}(\\boldsymbol{\\theta}) = \\text{Cov}(\\hat{\\boldsymbol{\\theta}}) $$

Асимптотические доверительные интервалы:

$$ \\hat{\\theta}_i \\pm z_{1-\\alpha/2} \\cdot \\text{se}(\\hat{\\theta}_i) $$
где $z_{1-\\alpha/2}$ - квантиль стандартного нормального распределения.

#### 1.2 Профиль правдоподобия

Формула профиля правдоподобия: 

$$ \\text{PL}(\\theta_k) = \\max_{\\boldsymbol{\\theta}_{-k}} \\ln L(\\theta_k, \\boldsymbol{\\theta}_{-k}) $$

Формула логарифма функции правдоподобия

$$ \\ln L(\\mu_x, \\mu_y, \\sigma_x, \\sigma_y, \\rho) = -n\\ln(2\\pi) - n\\ln(\\sigma_x) - n\\ln(\\sigma_y) - \\frac{n}{2}\\ln(1-\\rho^2) - \\frac{1}{2(1-\\rho^2)} \\sum_{i=1}^{n} \\left[ \\frac{(x_i - \\mu_x)^2}{\\sigma_x^2} - \\frac{2\\rho(x_i - \\mu_x)(y_i - \\mu_y)}{\\sigma_x\\sigma_y} + \\frac{(y_i - \\mu_y)^2}{\\sigma_y^2} \\right] $$

--- 

#### 2. Метод моментов (ММ)

Приравнивание выборочных и теоретических моментов:

* Первые моменты: $ \\hat{\\mu}_x^{\\text{MM}} = \\bar{x}, \\quad \\hat{\\mu}_y^{\\text{MM}} = \\bar{y} $
* Вторые центральные моменты:

$$ \\hat{\\sigma}_x^{2,\\text{MM}} = \\frac{1}{n} \\sum (x_i - \\bar{x})^2 $$

$$ \\hat{\\sigma}_y^{2,\\text{MM}} = \\frac{1}{n} \\sum (y_i - \\bar{y})^2 $$

$$ \\hat{\\rho}^{\\text{MM}} = \\frac{\\sum (x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum (x_i - \\bar{x})^2 \\sum (y_i - \\bar{y})^2}} $$

**Замечание:** Для нормального распределения ММП и ММ совпадают для первых двух моментов.         

--- 

#### 3. Bootstrap метод

**Non-parametric bootstrap:**

* Генерация $ B $ псевдовыборок путём сэмплирования с возвращением из исходных данных
* Для каждой псевдовыборки вычисляются оценки параметров
* Построение доверительных интервалов (процентильный метод)

**Parametric bootstrap:**

* Генерация $ B $ выборок из оценённого распределения $ \\mathcal{N}(\\hat{\\boldsymbol{\\mu}}, \\hat{\\Sigma}) $
* Оценка параметров на каждой выборке
* Сравнение с аналитическими интервалами из матрицы Фишера

**Bootstrap доверительный интервал (процентильный):**

$$ [\\hat{\\theta}^*_{(\\alpha/2)}, \\hat{\\theta}^*_{(1-\\alpha/2)}] $$

---

### Критерии сравнения:

1. **Смещение (bias):** $ \\text{Bias}(\\hat{\\theta}) = \\mathbb{E}[\\hat{\\theta}] - \\theta $
2. **Дисперсия (variance):** $ \\text{Var}(\\hat{\\theta}) $
3. **Среднеквадратичная ошибка (MSE):** $ \\text{MSE} = \\text{Bias}^2 + \\text{Var} $
4. **Доверительные интервалы:** длина
5. **Время выполнения**ы   
"""












MATH_MARKDOWN_GENERATE_DATA = """
### Построение доверительного эллипса (эллипса рассеяния)

Обозначим $\\chi^2_{2}(p)$ — $p$-квантиль распределения хи-квадрат с 2 степенями свободы.

* **Уравнение эллипса в матричной форме:**

$$ (\\mathbf{x} - \\boldsymbol{\\mu})^T \\Sigma^{-1} (\\mathbf{x} - \\boldsymbol{\\mu}) = \\chi^2_{2}(p) $$

* **Поиск собственных значений ($\\lambda_i$) и векторов ($\\mathbf{v}_i$) матрицы ковариации $\\Sigma$:**

$$ \\Sigma \\mathbf{v}_i = \\lambda_i \\mathbf{v}_i, \\quad i = 1, 2 $$

* **Расчет полуосей эллипса:**

$$ a = \\sqrt{\\chi^2_{2}(p) \\cdot \\lambda_1} \\quad \\text{(большая полуось)} $$

$$ b = \\sqrt{\\chi^2_{2}(p) \\cdot \\lambda_2} \\quad \\text{(малая полуось)} $$

* **Параметрическое уравнение канонического эллипса:**

$$ \\begin{pmatrix} u \\\\ v \\end{pmatrix} = \\begin{pmatrix} a \\cos \\theta \\\\ b \\sin \\theta \\end{pmatrix}, \\quad \\theta \\in [0, 2\\pi) $$

* **Поворот эллипса вдоль собственных векторов матрицы ковариации:**

$$ \\begin{pmatrix} x' \\\\ y' \\end{pmatrix} = V \\begin{pmatrix} u \\\\ v \\end{pmatrix}, \\quad \\text{где } V = [\\mathbf{v}_1 \\; \\mathbf{v}_2] $$

* **Сдвиг центра эллипса в математическое ожидание $\\boldsymbol{\\mu}$:**

$$ \\begin{pmatrix} x \\\\ y \end{pmatrix} = \\begin{pmatrix} x' \\\\ y' \\end{pmatrix} + \\begin{pmatrix} \\mu_x \\\\ \\mu_y \\end{pmatrix} $$

**Замечание:** Эллипс, построенный таким образом, содержит $p\\%$ точек двумерного нормального распределения.
"""
