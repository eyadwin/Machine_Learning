#from https://www.edureka.co/blog/naive-bayes-tutorial/
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

#Naive Bayes with SKLEARN

print("First Example - Predict Iris \n")
dataset = datasets.load_iris()

#Creating our Naive Bayes Model using Sklearn

model = GaussianNB()
model.fit(dataset.data, dataset.target)

#Making Predictions
expected = dataset.target
predicted = model.predict(dataset.data)

#Getting Accuracy and Statistics

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

#another example ***************************************************************
#Predict Human Activity Recognition
#code from https://www.machinelearningplus.com/predictive-modeling/how-naive-bayes-algorithm-works-with-example-and-full-code/
#train dataset
#https://raw.githubusercontent.com/selva86/datasets/master/har_train.csv
#validate dataset
#https://raw.githubusercontent.com/selva86/datasets/master/har_validate.csv

#dataset information
#http://groupware.les.inf.puc-rio.br/har#sbia_paper_section

# Import packages
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

print("Second Example - Predict Human Activity Recognition \n")
# Import data
training = pd.read_csv("har_train.csv")
test =  pd.read_csv("har_validate.csv")

# Create the X and Y
xtrain = training.drop('classe', axis=1)
ytrain = training.loc[:, 'classe']

xtest = test.drop('classe', axis=1)
ytest = test.loc[:, 'classe']

# Init the Gaussian Classifier
model = GaussianNB()

# Train the model
model.fit(xtrain, ytrain)

# Predict Output
pred = model.predict(xtest)
print(pred[:5])

# Plot Confusion Matrix
mat = confusion_matrix(pred, ytest)
names = np.unique(pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.show()