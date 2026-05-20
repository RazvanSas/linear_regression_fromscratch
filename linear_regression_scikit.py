import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import numpy as np

import statsmodels.api as sm

data = pd.read_csv('Salary_Data.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

regressor = LinearRegression()
regressor.fit(X, y)

plt.scatter(X, y, color='blue')
plt.plot(X, regressor.predict(X), color='red')
plt.show()

print(regressor.predict([[2.0]]))

print(r2_score(y, regressor.predict(X)))






