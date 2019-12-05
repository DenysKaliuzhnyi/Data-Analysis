from sklearn.datasets import load_iris
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn; seaborn.set()
from seaborn import stripplot
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
register_matplotlib_converters()
import pandas as pd
path = os.getcwd()
end = '\n\n\n'


# 1. Зчитати дані у формі часового ряду (ЧР). Зобразити отриманий ЧР (з підписами).
data = pd.read_csv(f"{path}\datasets\\international-airline-passengers.csv", index_col='Month', parse_dates=True)
data.columns = ['Thousands of passengers']
print(data)
fig, ax = plt.subplots()
data_new = np.log(data)
data_new.plot(ax=ax)
print(data.describe())
fig, ax = plt.subplots()
ax.set_ylabel('Thousands of passengers')
ax.plot(data)
fig.tight_layout()

# fig.savefig(f"{path}\images\\time_series_raw.png")


# 2. Провести згладжування ряду з різним лагом.
fig, ax = plt.subplots(nrows=3, ncols=2, sharex='col')

ax[1, 0].set_ylabel('Thousands of passengers')

ax[0, 0].set_title("MEAN ROLLING\nby 3 months")
ax[0, 1].set_title("MEDIAN ROLLING\nby 3 months")
ax[1, 0].set_title("by 6 months")
ax[1, 1].set_title("by 6 months")
ax[2, 0].set_title("by 12 months")
ax[2, 1].set_title("by 12 months")

ax[0, 0].plot(data.rolling('90D').mean())
ax[1, 0].plot(data.rolling('183D').mean())
ax[2, 0].plot(data.rolling('356D').mean())
ax[0, 1].plot(data.rolling('90D').median())
ax[1, 1].plot(data.rolling('183D').median())
ax[2, 1].plot(data.rolling('356D').median())
fig.tight_layout()
# fig.savefig(f"{path}\images\\time_series_lagged.png")


# 3. Розбити вихідний часовий ряд на систематичну, періодичну та хаотичну складові.
"""
x(t) – тренд, устойчивая долговременная тенденция изменения значений временного ряда, 
    закономерно изменяющаяся во времени;
s(t) – сезонная составляющая, периодически повторяющаяся компонента временного ряда, 
    на которую влияют погодные условия, социальные привычки, религиозные традиции и прочее;
z(t) – остаток – величина, показывающая нерегулярную (не описываемую трендом или сезонностью)
    составляющую исходного ряда в определённом временном интервале.
    
A time series is said to be stationary if it holds the following conditions true.

The mean value of time-series is constant over time, which implies, the trend component is nullified.
The variance does not increase over time.
Seasonality effect is minimal.
"""
split = seasonal_decompose(data, model='multiplicative')
split.plot()
fig.tight_layout()


# 4. Побудувати корелограму ЧР.
"""
Последовательность коэффициентов автокорреляции уровней первого, второго и других порядков называется
автокорреляционной функцией временного ряда. График значений коэффициентов автокорреляции разных 
порядков называют коррелограммой.
Если максимальным оказался коэффициент автокорреляции первого порядка, временной ряд содержит только тенденцию (тренд).
Если максимальным оказался коэффициент автокорреляции порядка n, ряд содержит циклические колебания с периодичностью
n моментов времени.
Если ни один из коэффициентов автокорреляции не является значимым (близок к 0), можно сказать, что либо ряд не 
содержит тенденции и циклических колебаний, либо ряд содержит нелинейную тенденцию, для выявления которой
проводят дополнительный анализ."""
fig, ax = plt.subplots()
otg1diff = data.diff(periods=1).dropna()
sm.graphics.tsa.plot_acf(otg1diff.values.squeeze(), lags=36, ax=ax)
fig.tight_layout()


# 5. Для часового ряду побудувати прогноз відповідним до моделі ЧР методом експоненційного
# згладжування (звичайним, подвійним чи ПОТРІЙНИМ – залежно від моделі) та методом ARIMA
# (Autoregressive Integrated Moving average.).
model = ExponentialSmoothing(data, seasonal_periods=12, trend='mul', seasonal='mul')
model_fit = model.fit()
data_predict = model_fit.predict('1949-01-01', '1970-12-01')
fig, ax = plt.subplots()
data.plot(ax=ax, linewidth=3.0)
data_predict.plot(ax=ax, style='--', linewidth=2)
ax.set_ylabel('Thousands of passengers')
fig.tight_layout()


# 6. Побудувати корелограми залишків та інші діаграми, що характеризують розподіл залишків.
#    Оцінити (усно) якість прогнозу.
resid = data.values.squeeze() - model_fit.predict('1949-01-01', '1960-12-01').values
resid = pd.DataFrame(resid, index=data.index)
fig, ax = plt.subplots()
otg1diff = resid.diff(periods=1).dropna()
sm.graphics.tsa.plot_acf(otg1diff.values.squeeze(), lags=36, ax=ax)
fig.tight_layout()

plt.show()
