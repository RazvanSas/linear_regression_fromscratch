import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

class LinearRegressionClosed:

    def __init__(self):
        self.__coef_ = None
        self.__intercept_ = 0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        Xb = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        A = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y

        self.__intercept_ = A[0]
        self.__coef_ = A[1:]

    def predict(self, X):
        X = np.array(X)

        return X @ self.__coef_ + self.__intercept_

if __name__ == '__main__':


    X,y = fetch_california_housing(return_X_y=True)

    regressor = LinearRegression()
    closed_regressor = LinearRegressionClosed()

    regressor.fit(X, y)
    closed_regressor.fit(X, y)

    y_pred = regressor.predict(X)
    print(r2_score(y, y_pred))

    y_pred = closed_regressor.predict(X)
    print(r2_score(y, y_pred))



