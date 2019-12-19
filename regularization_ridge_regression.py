import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#https://towardsdatascience.com/how-to-perform-lasso-and-ridge-regression-in-python-3b3b75541ad8

def scatter_plot(feature, target):
    plt.figure(figsize=(16, 8))
    plt.scatter(
        data[feature],
        data[target],
        c='black'
    )
    plt.xlabel("Money spent on {} ads ($)".format(feature))
    plt.ylabel("Sales ($k)")
    plt.show()
#we only have three advertising mediums, and sales is our target variable.
DATAPATH = 'Advertising.csv'
data = pd.read_csv(DATAPATH)
print(data.head())
data.drop(['Unnamed: 0'], axis=1, inplace=True) #remove first column which have the record number
scatter_plot('TV', 'sales')

#Least square regression

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

Xs = data.drop(['sales'], axis=1)
y = data['sales'].values.reshape(-1,1)

lin_reg = LinearRegression()

MSEs = cross_val_score(lin_reg, Xs, y, scoring='neg_mean_squared_error', cv=5)

mean_MSE = np.mean(MSEs)

print("Least square ",mean_MSE)

#Ridge regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

#GridSearchCV. This will allow us to automatically perform 5-fold cross-validation
# with a range of different regularization parameters in order to find the optimal value of alpha.
ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(Xs, y)

print("Ridge best alpth value ",ridge_regressor.best_params_)

print("Ridge MSE  scoe  ", ridge_regressor.best_score_)#MSE


