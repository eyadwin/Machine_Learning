#from https://towardsdatascience.com/multivariate-linear-regression-in-python-step-by-step-128c2b127171

import pandas as pd
import numpy as np
df = pd.read_csv('ex1data2.txt', header = None)
df.head()

#Add a column of ones for the bias term.
df = pd.concat([pd.Series(1, index=df.index, name='00'), df], axis=1)
df.head()

#Define the input variables or the independent variables X and
# the output variable or dependent variable  y.
# In this dataset, columns 0 and 1 are the input variables and column 2 is the output variable.
X = df.drop(columns=2)
y = df.iloc[:, 3]


#Normalize the input variables by dividing each column by the maximum values of that column.
# That way, each column’s values will be between 0 to 1
for i in range(1, len(X.columns)):
    X[i-1] = X[i-1]/np.max(X[i-1])
X.head()

#Initiate the theta values. I am initiating them as zeros
theta = np.array([0]*len(X.columns))
#Output: array([0, 0, 0])

#Calculate the number of training data that is denoted as m in the formula
m = len(df)

#Define the hypothesis function
def hypothesis(theta, X):
    return theta*X

# Define the cost function using the formula of the cost function
def computeCost(X, y, theta):
    y1 = hypothesis(theta, X)
    y1=np.sum(y1, axis=1)
    return sum(np.sqrt((y1-y)**2))/(2*47)

#Write the function for the gradient descent.
# This function will take X, y, theta, learning rate(alpha in the formula),
# and epochs(or iterations) as input.

def gradientDescent(X, y, theta, alpha, i):
    J = []  #cost function in each iterations
    k = 0
    while k < i:
        y1 = hypothesis(theta, X)
        y1 = np.sum(y1, axis=1)
        for c in range(0, len(X.columns)):
            theta[c] = theta[c] - alpha*(sum((y1-y)*X.iloc[:,c])/len(X))
        j = computeCost(X, y, theta)
        J.append(j)
        k += 1
    return J, j, theta

#Use the gradient descent function to get the final cost, the list of cost in each iteration,
# and the optimized parameters theta. I chose alpha as 0.05.
#I ran it for 10000 iterations.
J, j, theta = gradientDescent(X, y, theta, 0.05, 10000)

#Predict the output using the optimized theta
y_hat = hypothesis(theta, X)
y_hat = np.sum(y_hat, axis=1)

#Plot the original y and the predicted output ‘y_hat’
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(x=list(range(0, 47)),y= y, color='blue')
plt.scatter(x=list(range(0, 47)), y=y_hat, color='black')
plt.show()

#Plot the cost of each iteration to see the behavior
plt.figure()
plt.scatter(x=list(range(0, 10000)), y=J)
plt.show()
