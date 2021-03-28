# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 16:38:24 2020

@author: vanda
"""

# Loading Libraries

import pandas as pd
from collections import Counter
import numpy as np
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# =============================================================================

# load the csv file as a data frame
dataframe = pd.read_csv('Pohang.csv')
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


# =============================================================================

# summarize the class distribution
target = dataframe.values[:,-1]
counter = Counter(target)
for k,v in counter.items():
	per = v / len(target) * 100
	print('Class=%s, Count=%s, Percentage=%.3f%%' % (k, v, per))
    
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

# To balance another minority class
smote=SMOTE("minority")
X,y=smote.fit_sample(X,y)

# Re-summarize class distribution
print(X.shape, y.shape,Counter(y))
# =============================================================================

# Separate data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# define the model
model = ExtraTreesClassifier()

# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# fit the model on the whole dataset
model.fit(X_train, y_train)

# Make prediction
y_pred = model.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
cmET = confusion_matrix(y_test, y_pred)
sn.heatmap(cmET,cmap='Blues_r',annot=True, xticklabels='1234', yticklabels='1234')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()

#ROCAUC Plot
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from yellowbrick.classifier import ROCAUC


# Encode the non-numeric columns
X = OrdinalEncoder().fit_transform(X)
y = LabelEncoder().fit_transform(y)


# Instaniate the classification model and visualizer
visualizer = ROCAUC(model, Damage=[1, 2, 3, 4])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show()                       # Finalize and render the figure
