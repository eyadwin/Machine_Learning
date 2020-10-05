#https://towardsdatascience.com/performing-linear-regression-using-the-normal-equation-6372ed3c57

#Performing Linear Regression Using the Normal Equation

import numpy as np
X = np.c_[[1,1,1],[1,2,3]] # defining features
y = np.c_[[1,3,2]] # defining labels
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) # normal equation
print(theta)

#define new features we would like to predict values for
X_new = np.c_[[1,1,1,1],[0, 0.5,1.5,4]]  # new features

#obtain the predicted values
y_pred = X_new.dot(theta)  # making predictions
print(y_pred)

