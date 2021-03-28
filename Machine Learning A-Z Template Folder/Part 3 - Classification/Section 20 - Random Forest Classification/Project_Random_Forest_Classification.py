# Random Forest template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std

# load the csv file as a data frame
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

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling - 
""""We apply FS here in order to run our code faster : 
    As we are plotting our graphic result with the resolution of 0.01 and 
    our classifier is going to predict the class of all the pixel points of 
    our graph with resolution of 0.01, so we need to apply feature scaling 
    here so that our code run faster.""" 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)


# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

