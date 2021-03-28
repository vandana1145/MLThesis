# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasheet
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
# creating object for SimpleImputer class as "imputer"
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose=0)
imputer = imputer.fit(X[:, 1:3]) #upper bound is not included, but lower bound
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder #LabelEncoder replaces texts by numbers not necessariliy in orders
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder # OneHotEncoder creates dummy variables
labelencoder_X = LabelEncoder() 
"""Calling the "LabelEncoder" class by object labelencoder
applying labelencoder object to our required colum "Country" in the datasheet"""
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
columtransformer = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough' )                                       # Leave the rest of the columns untouched
X = columtransformer.fit_transform(X)

# Encoding the dependent Variable
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
""" we do not need to do fit_transform for X_test, as they already accodmodate 
themselves as per the X_train set (which got fit_transformed)"""
X_test = sc_X.transform(X_test)
"""Feature scaling is not required here for dependent variable Y, because it
is in a categorical classification scale. But we would need to do feature
scaling for dependent variable in case of regression problems as the dependent
variables in regression take place in between high range values. """

