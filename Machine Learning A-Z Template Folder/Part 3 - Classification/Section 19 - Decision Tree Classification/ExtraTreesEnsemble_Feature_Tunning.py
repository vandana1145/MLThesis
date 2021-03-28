# Loading Libraries

import pandas as pd
from collections import Counter
import numpy as np
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot as plt
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

# =============================================================================

# get a list of models to evaluate
def get_models():
	models = dict()
	for i in range(1, 9):
		models[str(i)] = ExtraTreesClassifier(max_features=i)
	return models

# evaluate a given model using cross-validation
def evaluate_model(model):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

# get the models to evaluate
models = get_models()

# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    
# plot model performance for comparison
import matplotlib.pyplot as pyplot
pyplot.boxplot(results, labels=names, showmeans=True)
plt.xlabel('Features of the input data')
plt.ylabel('Accuracy score')
pyplot.show()





