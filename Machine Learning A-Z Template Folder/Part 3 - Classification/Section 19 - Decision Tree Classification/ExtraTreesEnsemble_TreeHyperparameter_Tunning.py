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
	print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))
    
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

# =============================================================================

# get a list of models to evaluate
def get_classifiers():
	classifiers = dict()
	classifiers['10'] = ExtraTreesClassifier(n_estimators=10)
	classifiers['20'] = ExtraTreesClassifier(n_estimators=20)
	classifiers['30'] = ExtraTreesClassifier(n_estimators=30)
	classifiers['40'] = ExtraTreesClassifier(n_estimators=40)

	return classifiers

# evaluate the model
def evaluate_classifier(classifier):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return n_scores

# get the models to evaluate
classifiers = get_classifiers()

# evaluate the models and store results
results, names = list(), list()
for name, classifier in classifiers.items():
	n_scores = evaluate_classifier(classifier)
	results.append(n_scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(n_scores), std(n_scores)))
# plot model performance for comparison
import matplotlib.pyplot as plt
pyplot.boxplot(results, labels=names, showmeans=True)
plt.xlabel('Number of trees')
plt.ylabel('Accuracy score')
pyplot.show()

# fit the model on the whole dataset
classifier.fit(X_train, y_train)

# Make prediction
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmET = confusion_matrix(y_test, y_pred)