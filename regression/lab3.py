from sklearn.datasets import load_iris
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
path = os.getcwd()
import seaborn; seaborn.set()

class X:
    pass


x = X()
table = load_iris()
x.data = table.data
x.pl, x.pw = x.data[:, 2], x.data[:, 3]
x.pwT = x.pw.reshape(-1, 1)
x.lables = table.feature_names[2: 4]

# 1. Для двох (трьох) масивів взаємопов’язаних даних (з попередньої роботи)
#    побудувати діаграму розсіювання, визначити тип залежності.
# 2. Порахувати параметри регресійної моделі.
# 3. Зобразити отриману лінію регресії разом з діаграмою розсіювання.
mod = sm.OLS(x.pl, sm.add_constant(x.pw))
fit = mod.fit()
print(fit.summary())


model = LinearRegression()
fit = model.fit(x.pwT, x.pl)
determination = model.score(x.pwT, x.pl)
intercept = fit.intercept_
coef = fit.coef_[0]

f = lambda y: intercept + coef*y
f = np.vectorize(f)
predictions = f(x.pw)
rng = np.linspace(np.min(x.pw), np.max(x.pw), x.pw.size)
regrline = f(rng)

fig, ax = plt.subplots()
ax.set_xlabel(x.lables[1])
ax.set_ylabel(x.lables[0])
ax.set_title(
    f"Linear Regression\n"
    f"intercept={np.around(intercept, 2)}, "
    f"coef={np.around(coef, 2)}, "
    f"determination={np.around(determination, 2)}")
ax.scatter(x.pw, x.pl)
ax.plot(rng, regrline, c='red')

fig.savefig(f"{path}\images\scatter.png")


# 4. Побудувати діаграму «відгук-залишки»
residuals = x.pl - predictions
fig, ax = plt.subplots()
ax.set_title(
    f"MSE = {np.around(metrics.mean_squared_error(y_pred=predictions, y_true=x.pl), 2)}, "
    f"MAE = {np.around(metrics.mean_absolute_error(y_pred=predictions, y_true=x.pl), 2)}, "
    f"max error = {np.around(metrics.max_error(y_pred=predictions, y_true=x.pl), 2)}")
ax.set_xlabel("feedback")
ax.set_ylabel("residuals")
ax.scatter(x.pl, residuals)
ax.axhline(color='grey')
fig.savefig(f"{path}\images\\feedback_residuals.png")


# 5. Побудувати діаграму «прогноз-залишки»
# A residual plot is a graph that shows the residuals on the vertical axis and the independent variable
# on the horizontal axis. If the points in a residual plot are randomly dispersed around the horizontal
# axis, a linear regression model is appropriate for the data; otherwise, a non-linear model is more appropriate.
# Mean absolute error, mean squared error
fig, ax = plt.subplots()
ax.set_title(
    f"MSE = {np.around(metrics.mean_squared_error(y_pred=predictions, y_true=x.pl), 2)}, "
    f"MAE = {np.around(metrics.mean_absolute_error(y_pred=predictions, y_true=x.pl), 2)}, "
    f"max error = {np.around(metrics.max_error(y_pred=predictions, y_true=x.pl), 2)}")
ax.set_xlabel("prediction (fitted values)")
ax.set_ylabel("residuals")
ax.axhline(color='grey')
ax.scatter(predictions, residuals)
fig.savefig(f"{path}\images\\prediction_residuals.png")


# 6. Побудувати Q-Q-діаграму для залишків
fig, ax = plt.subplots()
stats.probplot(residuals, plot=ax)
fig.savefig(f"{path}\images\qq.png")


fig, ax = plt.subplots()
ax.hist(residuals, bins='fd', color='lightblue', density=True,)
xaxis = np.linspace(residuals.min(), residuals.max(), residuals.size)
yaxis = stats.norm.pdf(xaxis, residuals.mean(), residuals.std())
ax.plot(xaxis, yaxis)
ax.set_title("residuals")
fig.savefig(f"{path}\images\hist.png")

