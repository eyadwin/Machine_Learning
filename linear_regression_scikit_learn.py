from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
X = data['Head Size(cm^3)'].values.reshape(-1, 1)  # values converts it into a numpy array
Y = data['Brain Weight(grams)'].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

# Creating Model
reg = LinearRegression()
# Fitting training data
reg = reg.fit(X, Y)
# Y Prediction
Y_pred = reg.predict(X)

# Calculating R2 Score
r2_score = reg.score(X, Y)

print(r2_score)


# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

# Ploting Line
plt.plot(X, Y_pred, color='#58b970', label='Regression Line')

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()