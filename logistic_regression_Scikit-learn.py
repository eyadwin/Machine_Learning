#import pandas
import pandas as pd
from sklearn.model_selection import train_test_split
# import the class
from sklearn.linear_model import LogisticRegression
# import the metrics class
from sklearn import metrics
# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

col_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
# load dataset
#from https://www.kaggle.com/uciml/pima-indians-diabetes-database
pima = pd.read_csv("diabetes.csv")
pima.head()

#split dataset in features and target variable
feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X = pima[feature_cols] # Features
y = pima.Outcome# Target variable

# split X and y into training and testing sets
#75% data will be used for model training and 25% for model testing.

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)



# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

#Receiver Operating Characteristic(ROC) curve is a plot of the true positive rate against the
# false positive rate. It shows the tradeoff between sensitivity and specificity.
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()