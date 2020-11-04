#https://towardsdatascience.com/multiclass-classification-algorithm-from-scratch-with-a-project-in-python-step-by-step-guide-485a83c79992
#y column has the digits from 1 to 10. That means we have 10 classes
#5 rows × 400 columns

#import the necessary packages and the dataset
import pandas as pd
import numpy as np
xl = pd.ExcelFile('ex3d1.xlsx')
df = pd.read_excel(xl, 'X', header=None) #5000 rows x 400 columns

#y, which is the output variable
y = pd.read_excel(xl, 'y', hearder = None) #4999 row x 1 column

#iloc: Purely integer-location based indexing for selection by position.
df = df.iloc[1:] ## All except first column of data frame
y = y.iloc[:, 0]  ## first column of data frame

y1 = np.zeros([df.shape[0], len(y.unique())])#df.shape returns the dimensions of matrix 4999x10
y1 = pd.DataFrame(y1)


#هذا الجزء الهدف منه جعل كل صنف من y يكون صفر عند قيمة الصنف الحالية لكل القيم غير الصنف وواحد عند قيمة الصنف
#فلو كان الصنف الذي نحوله الان هو 3 يكون الناتج كالتالي
# 3 => 1
# 6 => 0
# 2 => 0
#We will make one column for each of the classes with the same length as y.
# When the class is 5, make a column that has 1 for the rows with 5 and 0 otherwise.
for i in range(0, len(y.unique())):
    for j in range(0, len(y1)):
        if y[j] == y.unique()[i]:
            y1.iloc[j, i] = 1
        else:
            y1.iloc[j, i] = 0
print(y1.head())


#The data is clean. Not much preprocessing is required.
# We need to add a bias column in the input variables.
# Please check the length of df and y. If the length is different, the model will not work.

print(len(df))
print(len(y))
X = pd.concat([pd.Series(1, index=df.index, name='00'), df], axis=1)

theta = np.zeros([df.shape[1]+1, y1.shape[1]])
theta = np.ones(df.shape[1]+1)

#Define the hypothesis that takes the input variables and theta.
# It returns the calculated output variable. Using segmoid function

def hypothesis(theta, X):
    return 1 / (1 + np.exp(-(np.dot(theta, X.T)))) - 0.0000001

h = hypothesis(theta, X)

#Build the cost function that takes the input variables, output variable, and theta.
# It returns the cost of the hypothesis.

def cost(X, y, theta):
    y1 = hypothesis(X, theta)
    return -(1/len(X)) * np.sum(y*np.log(y1) + (1-y)*np.log(1-y1))


theta = pd.DataFrame(theta)


#Define the function ‘gradient_descent’ now. This function will take input variables,
# output variable, theta, alpha, and the number of epochs as the parameter.
# Here, alpha is the learning rate.
def gradient_descent(X, y, theta, alpha, epochs):
    m = len(X)
    for i in range(0, epochs):
        for j in range(0, 10):
            theta = pd.DataFrame(theta)
            h = hypothesis(theta.iloc[:,j], X)
            for k in range(0, theta.shape[0]):
                theta.iloc[k, j] -= (alpha/m) * np.sum((h-y.iloc[:, j])*X.iloc[:, k])
            theta = pd.DataFrame(theta)
    return theta, cost



#Initialize the theta. Remember, we will implement logistic regression for each class.
# There will be a series of theta for each class as well.
#I am running this for 1500 epochs.

theta = np.zeros([df.shape[1]+1, y1.shape[1]])
theta = gradient_descent(X, y1, theta, 0.02, 100)#1500



def predict(X, y):
    #theta1 = gradient_descent(X, y, theta, alpha, epochs)
    accuracy = 0
    for i in range(10):
        h = hypothesis(theta1.iloc[:,j], X)
        for n in range(0, len(h)):
            if h[n] >= 0.5 and y1.iloc[n, i] == 1:
                accuracy += 1
            elif h[n] < 0.5 and y1.iloc[n, i] == 0:
                accuracy += 1
    return accuracy

theta = pd.DataFrame(theta)
hypothesis(theta.iloc[:,9], X)

#With this updated theta, calculate the output variable
output = []
for i in range(0, 10):
    theta1 = pd.DataFrame(theta)
    h = hypothesis(theta1.iloc[:,i], X)
    output.append(h)
output=pd.DataFrame(output)

#Compare the calculated output and the original output variable to calculate the accuracy of the model
accuracy = 0
for col in range(0, 10):
    for row in range(len(y1)):
        if y1.iloc[row, col] == 1 and output.iloc[col, row] >= 0.5:
            accuracy += 1
accuracy = accuracy/len(X)

print(accuracy)