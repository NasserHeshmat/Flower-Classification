import csv
import random
import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

dataset = pd.read_csv(r'D:\task1CV\venv\features1.csv')
feature_columns = ['R', 'G', 'B']
X = dataset[feature_columns].values
Y = dataset['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 4)
# Instantiate learning model (k = 3)
classifier = KNeighborsClassifier(n_neighbors=3)
# Fitting the model
classifier.fit(X_train, y_train)
joblib.dump(classifier, 'knnModel.pkl')
# Predicting the Test set results
#X_test = [[153.46,129.38,33.03]]
y_pred = classifier.predict(X_test)
print("y_test",y_test)
print("y_pred",y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
# creating list of K for KNN
k_list = list(range(1,50,2))
# creating list of cv scores
cv_scores = []

for className in ['Bellflower', 'Phlox' ,'Goldquelle' ,'Calendula' ,'Leucanthemum']:
    print("***********************")
    print (className, " : ")
    tp=0
    tn=0
    fp=0
    fn=0
    for j in range(y_test.shape[0]):
        if y_test[j]==className:
            if y_pred[j]==className:
                tp=tp+1
            else :
                fp=fp+1
        else:
            if y_pred[j]==y_test[j]:
                tn=tn+1
            else :
                fn=fn+1
    print("TP :  ", tp)
    print("TN :  ", tn)
    print("FP :  ", fp)
    print("FN :  ", fn)
    print("Sensitivity :  ", round(100*tp/(tp+fn),2) , '%')
    print("Specifcity :  ", round(100*tn/(tn+fp),2) , '%')
    print("Accuracy :  ", round(100*(tp+tn) / (tn + fp+tp+fn), 2), '%')


