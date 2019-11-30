# Importing Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Set Description: The data set contains the following variables:
#
# Gender: Male or female represented as binary variables
# Age: Age of an individual
# Head size in cm^3: An individuals head size in cm^3
# Brain weight in grams: The weight of an individual’s brain measured in grams

   # Reading Data
data = pd.read_csv('headbrain.csv')
print(data.shape)
print(data.head())

# Coomputing X and Y
X = data['Head Size(cm^3)'].values.reshape(-1, 1)
Y = data['Brain Weight(grams)'].values.reshape(-1, 1)


# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# Total number of values
n = len(X)

# Using the formula to calculate 'm' and 'c'
numer = 0
denom = 0
for i in range(n):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
denom += (X[i] - mean_x) ** 2
m = numer / denom
c = mean_y - (m * mean_x)

# Printing coefficients
print("Coefficients")
print(m, c)

#The value of m and c from above will be added to this equation
#BrainWeight = c + m ∗ HeadSize

# Plotting Values and Regression Line

max_x = np.max(X) + 100
min_x = np.min(X) - 100

# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = c + m * x

# Ploting Line
plt.plot(x, y, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

# Calculating Root Mean Squares Error
rmse = 0
for i in range(n):
    y_pred = c + m * X[i]
    rmse += (Y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/n)
print("RMSE")
print(rmse)

# Calculating R2 Score
ss_tot = 0
ss_res = 0
for i in range(n):
    y_pred = c + m * X[i]
    ss_tot += (Y[i] - mean_y) ** 2
    ss_res += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_res/ss_tot)
print("R2 Score")
print(r2)