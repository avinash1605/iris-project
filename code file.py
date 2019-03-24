# -*- coding: utf-8 -*-
"""
Created on Thu May 24 16:51:39 2018

@author: tgaga
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('iris_data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


#replacing categorical data 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labenc_y = LabelEncoder()
y = labenc_y.fit_transform(y)


#splitting datasets into training and testing sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.75,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


#creating the svm model
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)

#predicting the results
y_pred = classifier.predict(X_test)

#creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
percentage_accuracy = accuracy_score(y_test,y_pred)*100




