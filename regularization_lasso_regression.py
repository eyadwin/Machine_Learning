#https://towardsdatascience.com/how-to-perform-lasso-and-ridge-regression-in-python-3b3b75541ad8
import numpy as np
import pandas as pd

#we only have three advertising mediums, and sales is our target variable.
DATAPATH = 'Advertising.csv'
data = pd.read_csv(DATAPATH)
print(data.head())
data.drop(['Unnamed: 0'], axis=1, inplace=True) #remove first column which have the record number

#Least square regression

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

Xs = data.drop(['sales'], axis=1)
y = data['sales'].values.reshape(-1,1)

lin_reg = LinearRegression()

MSEs = cross_val_score(lin_reg, Xs, y, scoring='neg_mean_squared_error', cv=5)

mean_MSE = np.mean(MSEs)

print("Least square MSE  ",mean_MSE)

#Lasso regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

#GridSearchCV. This will allow us to automatically perform 5-fold cross-validation
# with a range of different regularization parameters in order to find the optimal value of alpha.

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 5)

lasso_regressor.fit(Xs, y)

print("Lasso best alpth value ",lasso_regressor.best_params_)

print("Lasso MSE  score ",lasso_regressor.best_score_) #MSE



