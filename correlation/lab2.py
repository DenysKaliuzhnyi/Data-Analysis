from sklearn.datasets import load_iris
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import stats
from sklearn.linear_model import LinearRegression
path = os.getcwd()

import seaborn; seaborn.set()

class X:
    pass


x = X()
table = load_iris()
x.data = table.data
x.lables = table.feature_names


# 1. Для кількох наборів даних зобразити матричну діаграму розсіювання.
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
for i in range(4):
    ax[i, 0].set_ylabel(x.lables[i])
    ax[-1, i].set_xlabel(x.lables[i])
    ax[i, i].hist(x.data[:, i])
    for j in range(i + 1, 4):
        ax[i, j].scatter(x.data[:, i], x.data[:, j], s=6)
        ax[j, i].scatter(x.data[:, j], x.data[:, i], s=6)

fig.savefig(f"{path}\images\scatter.png")

# 2. Для них же вивести кореляційну матрицю (карту кореляцій) (цифрами та кружечками/квадратиками) та граф кореляцій.

# 3. Для найбільш суттєвих кореляцій порахувати коефіцієнти кореляції Пірсона,
#    Спірмена та Кендала, статистично перевірити їх на значущість.
a, b = x.data[:, 2], x.data[:, 3]
x.pearson = stats.pearsonr(a, b)
x.spearman = stats.spearmanr(a, b)
x.kendall = stats.kendalltau(a, b)


for (var, val) in vars(x).items():
    print(f"{var} = {val}")
