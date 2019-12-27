#from https://medium.com/analytics-vidhya/linear-regression-using-python-ce21aa90ade6
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

USAhousing = pd.read_csv('USA_Housing.csv')
USAhousing.head()
USAhousing.info()

#describe() function yields a neat and concise data frame with important statistics from each
# numerical column of the original dataset
USAhousing.describe()
USAhousing.columns

#pairs plot builds on two basic figures, the histogram and the scatter plot
sns.pairplot(USAhousing)

#exploring a single variable is with the histogram.
sns.distplot(USAhousing['Price'])

#find the correlation between the variables in the dataset

USAhousing.corr()

#plot the correlation using a heatmap
sns.heatmap(USAhousing.corr())

#split up our data into an X array that contains the features to train on, and a y array with the
# target variable, in this case the Price column. We will toss out the Address column because it only
# has text info that the linear regression model can’t use.
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']
#Trained data is the data on which we apply the linear regression algorithm.
# And finally we test that algorithm on the test data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)

#visualise the prediction
plt.scatter(y_test,predictions)