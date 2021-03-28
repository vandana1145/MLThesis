# -*- coding: utf-8 -*-
"""
Created on Tue May 12 19:06:45 2020

@author: vanda
"""

#Import the libraries
import pandas as pd  
import scipy
import numpy as np  
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
%matplotlib inline

#Read the input data data from the external CSV
dataframe = pd.read_csv('Ecuador.csv')
X = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, 8].values


#Rescale data (between 0 and 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)

# Taking care of missing data
from sklearn.impute import SimpleImputer
# creating object for SimpleImputer class as "imputer"
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose=0)
imputer = imputer.fit(rescaledX[:, 1:8]) #upper bound is not included, but lower bound
rescaledX[:, 1:8] = imputer.transform(rescaledX[:, 1:8])

# summarize the class distribution
target = dataframe.values[:,-1]
counter = Counter(target)
for k,v in counter.items():
	per = v / len(target) * 100
	print('Class=%s, Count=%s, Percentage=%.3f%%' % (k, v, per))
    

import pylab as pl
dataframe.drop('Damage',axis=1).hist(bins=30, figsize=(20,20))
pl.suptitle("Histogram for each numeric input variable")
plt.savefig('Damage_hist')
plt.show()

# summarize class distribution
print(X.shape, y.shape,Counter(y))

# Implementing SMOTE for the Imbalanced data in Multi-class classification
smote=SMOTE("minority")
X,y=smote.fit_sample(rescaledX,y)

# Re-summarize class distribution
print(X.shape, y.shape,Counter(y))

# To balance another minority class
smote=SMOTE("minority")
X,y=smote.fit_sample(X,y)

# Re-summarize class distribution
print(X.shape, y.shape,Counter(y))

# Encoding the Dependent Variable
# label encode the target variable to have the classes 0 and 1
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Apply kernel to transform the data to a higher dimension
kernels = ['Polynomial', 'rbf', 'Sigmoid','Linear']
#A function which returns the corresponding SVC model
def getClassifier(ktype):
    if ktype == 0:
        # Polynomial kernal
        return SVC(kernel='poly', degree=8, gamma="auto")
    elif ktype == 1:
        # Radial Basis Function kernal
        return SVC(kernel='rbf', gamma="auto")
    elif ktype == 2:
        # Sigmoid kernal
        return SVC(kernel='sigmoid', gamma="auto")
    elif ktype == 3:
        # Linear kernal
        return SVC(kernel='Linear', gamma="auto")

#Train a model - 
#Call the SVC() model from sklearn and fit the model to the training data
    for k in range(3):
    # Separate data into test and training sets
        from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    
# Train a SVC model using different kernal
    classifier = getClassifier(k) 
    classifier.fit(X_train, y_train)
    
# Make prediction
    y_pred = classifier.predict(X_test)
    
# Evaluate our model
    print("Evaluation:", kernels[k], "kernel")
    print(classification_report(y_test,y_pred))
    

#Create a dictionary called param_grid and fill out some parameters for kernels, C and gamma
#Tuning the hyper-parameters of an estimator
    #GridSearchCV helps us combine an estimator with a grid search preamble to tune hyper-parameters.
#Import GridsearchCV from Scikit Learn
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

#Create a GridSearchCV object and fit it to the training data
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)    

#Find the optimal parameters
print(grid.best_estimator_)


grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
