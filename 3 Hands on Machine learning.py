#MNIST dataset learning practice

import numpy as np
import pandas as pd

#reading the dataset

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784',version = 1)

#datset information
mnist.keys()

#we just have to work with data and the target so we define them as X and y

X = mnist["data"]
y =mnist["target"]

#let we represent some random digit in the image 

some_digit = X[0]

import matplotlib.pyplot as plt
some_digit_new = some_digit.reshape(28,28)
plt.imshow(some_digit_new,cmap="binary")
plt.show()

#answer

y[0]

#clearly the datset contain image with a label of the digit 

#shape of the datasets

X.shape
y.shape
#contain 7000 rows so for training a model let we split it into two parts 60000  as training and 10000 as test set

X_train,X_test,y_train,y_test = X[:60000],X[60000:],y[:60000],y[60000:]

y_train_5 = (y_train =='5')
y_test_5 = (y_test == '5')

#training a binary classifier fo digit 5

#SGDClassifier is a binary classifier

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)
sgd_clf.predict([some_digit])
#answer is True ,that means our classifier is working fine 

#Now its time to calculate its accuray.
#We will use cross_val_score to find the accuracy of the model

from sklearn.model_selection import cross_val_score

scores = cross_val_score(sgd_clf,X_train,y_train_5,cv = 3,scoring = "accuracy")
#Scores are 0.95 0.96 0.96 ..pretty good


#using cross_val_predict will return the predicted value instead of the scores as in the case of cross_val_score
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_5,cv = 3)

#lets use confusion matrix to get insights of the data

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5,y_train_pred)


#lets get the precision ,recall and f1score of this

from sklearn.metrics import precision_score, recall_score,f1_score
precision_score(y_train_5,y_train_pred) #: 0.8370879772350012
recall_score(y_train_5,y_train_pred) #0.6511713705958311
f1_score(y_train_5,y_train_pred) # 0.7325171197343846

#Decision Function

y_train_pred1 = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3,method = "decision_function")


#precision recall curve

from sklearn.metrics import precision_recall_curve

precision , recall , threshold = precision_recall_curve(y_train_5,y_train_pred1)

#Till now we are working with binary classification, Now lets move to Multiclass classification

#The first one in list is Support Vector Machine Classifier

from sklearn.svm import SVC
svc_clf = SVC()
svc_clf.fit(X_train,y_train)
svc_clf.predict([some_digit])


#lets try onevsRestClassifier
from sklearn.multiclass import OneVsRestClassifier
ovr_clf =OneVsRestClassifier(SVC())
ovr_clf.fit(X_train,y_train)
ovr_clf.predict([X_test[2]])
