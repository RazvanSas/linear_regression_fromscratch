import pandas as pd
from linear_regression_closed_form import LinearRegressionClosed
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


data = pd.read_csv('50_Startups.csv')

col_index = data.columns.get_loc("State")

categories = data['State'].unique()

for i, category in enumerate(categories):
    data.insert(col_index + i, category, (data['State'] == category).astype(int))

data = data.drop('State', axis = 1)
data = data.drop('New York', axis = 1)

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

regressor = LinearRegressionClosed()
regressor.fit(X, y)

linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

print(r2_score(y, regressor.predict(X)))
print(r2_score(y, linear_regressor.predict(X)))


