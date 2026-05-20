import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

data = pd.read_csv('Salary_Data.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


def loss_function(m, b, X, y):
    n = len(X)
    loss = 0
    for i in range(n):
        loss += (y[i] - (m * X[i] + b)) ** 2
    loss = loss / float(n)
    return loss

def gradient_descent(m_now, b_now, X, y, L):
    m_descent = 0
    b_descent = 0
    n = len(X)

    for i in range(n):
        m_descent += (y[i] - m_now * X[i] - b_now) * X[i]
        b_descent += y[i] - m_now * X[i] - b_now

    m = m_now - ((-2)/n * m_descent) * L
    b = b_now - ((-2)/n * b_descent) * L

    return m, b

m = 0
b = 0
L = 0.01
epochs = 3000

for i in range(epochs):
    m, b = gradient_descent(m, b, X, y, L)

plt.scatter(X, y, color='blue')
plt.plot(X, [m*x + b for x in X], color='red')

plt.show()

print(m, b)

print(r2_score(y, [m*x + b for x in X]))






