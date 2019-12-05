import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import pylab
import seaborn; seaborn.set()

path = os.getcwd()


class X:
    pass


# 1. Визначити вектор спостережень.
x = X()
x.arr = np.genfromtxt(f'{path}\datasets\weather.csv', delimiter=',', usecols=(2, ), skip_header=1)
# x.arr = np.random.normal(0,1,1000)
x.sorted = np.sort(x.arr)
x.min = x.sorted[0]
x.max = x.sorted[-1]
x.n = x.arr.size


# 2. Підрахувати показники центру: середні значення, медіану.
x.mean = np.mean(x.arr)
x.median = np.median(x.arr)


# 3. Підрахувати показники варіації: дисперсію, стандартне відхилення та коефіцієнт варіації,
#    розмах варіації та інтерквартильний розмах.
x.var = np.var(x.arr)
x.std = np.std(x.arr)
x.variation = x.std / x.mean
x.range = x.max - x.min
x.iqr = np.quantile(x.arr, 0.75) - np.quantile(x.arr, 0.25)


# 4. Побудувати ящик з вусами (з підписами).
fig, ax = plt.subplots()
ax.boxplot(
    x.arr,
    vert=False
)
fig.tight_layout()
fig.savefig(f"{path}\images\\boxplot.png")


# 5. Вивести п’ятиточкову характеристику (екстремальні точки та квартилі).
x.dots = tuple(np.quantile(x.arr, p) for p in np.linspace(0, 1, 5))


# 6. Знайти 1-й та 9-й децилі.
x.deciles = (np.quantile(x.arr, 0.1), np.quantile(x.arr, 0.9))


# 7. Підрахувати коефіцієнт асиметрії та коефіцієнт ексцесу.
x.skewness = stats.skew(x.arr)
x.kurtosis = stats.kurtosis(x.arr)


# 9. Зобразити Q-Q-діаграму для перевірки узгодженості з відповідним розподілом.
fig, ax = plt.subplots()
stats.probplot(x.arr, plot=ax)
fig.tight_layout()
fig.savefig(f"{path}\images\qq.png")


# 8. Для вектора спостережень побудувати гістограму, використовуючи різні методи групування (базове правило,
#    правило Скотта, правило Фрідмана-Діаконіса). На тому ж графіку зобразити оцінку щільності та
#    графік щільності гіпотетичного розподілу.
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
xaxis = np.linspace(x.min, x.max, x.n)
yaxis = stats.norm.pdf(xaxis, x.mean, x.std)

ax1.hist(
    x.arr,
    bins="sturges",
    density=True,
    cumulative=False,
    histtype='bar',
    align='mid',
    orientation='vertical',
    color='lightblue'
)
ax1.set_title("sturges")

ax2.hist(
    x.arr,
    bins="scott",
    density=True,
    cumulative=False,
    histtype='bar',
    align='mid',
    orientation='vertical',
    color='pink'
)
ax2.set_title("scott")

ax3.hist(
    x.arr,
    bins='fd',
    density=True,
    cumulative=False,
    histtype='bar',
    align='mid',
    orientation='vertical',
    color='coral'
)
ax3.set_title("fd")

fig.tight_layout()
fig.savefig(f"{path}\images\hist.png")

# За допомогою одного з статистичних критеріїв перевірити згоду з відповідним розподілом
x.test = stats.normaltest(x.arr)


for (var, val) in vars(x).items():
    print(f"{var} = {val}")

