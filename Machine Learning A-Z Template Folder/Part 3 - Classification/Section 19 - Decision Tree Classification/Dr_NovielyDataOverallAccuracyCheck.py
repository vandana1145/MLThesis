# Loading Libraries

import pandas as pd
from collections import Counter
import numpy as np

# baseline model and test harness for the damage identification dataset
from numpy import mean
from numpy import std
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

# spot check machine learning algorithms on the damage identification dataset
from matplotlib import pyplot
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# =============================================================================

# load the csv file as a data frame
dataframe = pd.read_csv('C:/Users/vanda/.spyder-py3/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 19 - Decision Tree Classification/Malawi.csv')
X = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, 20].values

#Rescale data (between 0 and 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)

# =============================================================================
# # Taking care of missing data
# from sklearn.impute import SimpleImputer
# # creating object for SimpleImputer class as "imputer"
# imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose=0)
# imputer = imputer.fit(rescaledX[:, 1:20]) #upper bound is not included, but lower bound
# rescaledX[:, 1:20] = imputer.transform(rescaledX[:, 1:20])
# 
# =============================================================================

# =============================================================================

# summarize the class distribution
target = dataframe.values[:,-1]
counter = Counter(target)
for k,v in counter.items():
	per = v / len(target) * 100
	print('Class=%s, Count=%s, Percentage=%.3f%%' % (k, v, per))
    
# summarize class distribution
print(X.shape, y.shape,Counter(y))


# =============================================================================
# Used only in case of Binary classification
# define oversample strategy
# oversample = SMOTE(sampling_strategy=0.5)
# 
# # fit and apply the transform
# X_over, y_over = oversample.fit_resample(rescaledX, y)
# 
# =============================================================================


# =============================================================================
# The SMOTE algorithm is a popular approach for oversampling the minority class. This
# technique can be used to reduce the imbalance or to make the class distribution even.
# =============================================================================

# =============================================================================
# # Implementing SMOTE for the Imbalanced data in Multi-class classification
# smote=SMOTE("minority")
# X,y=smote.fit_sample(rescaledX,y)
# 
# # Re-summarize class distribution
# print(X.shape, y.shape,Counter(y))
# 
# # To balance another minority class
# smote=SMOTE("minority")
# X,y=smote.fit_sample(X,y)
# 
# # Re-summarize class distribution
# print(X.shape, y.shape,Counter(y))
# =============================================================================


# =============================================================================
# print(dataframe.shape)
# 
# print(dataframe['Damage'].unique())
# 
# print(dataframe.groupby('Damage').size())
# 
# import seaborn as sns
# sns.countplot(dataframe['Damage'],label="count")
# plt.show()
# =============================================================================


# =============================================================================
# # create a histogram plot of each variable
# import pylab as pl
# dataframe.drop('Damage',axis=1).hist(bins=30, figsize=(20,20))
# pl.suptitle("Histogram for each numeric input variable")
# plt.savefig('Damage_hist')
# plt.show()
# 
# =============================================================================

# Encoding the Dependent Variable
# label encode the target variable to have the classes 0 and 1
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# =============================================================================

# define models to test
def get_models():
	models, names = list(), list()
	# SVM
	models.append(SVC(gamma='auto'))
	names.append('SVM')
	# KNN
	models.append(KNeighborsClassifier())
	names.append('KNN')
	# Bagging
	models.append(BaggingClassifier(n_estimators=1000))
	names.append('BAG')
	# RF
	models.append(RandomForestClassifier(n_estimators=1000))
	names.append('RF')
	# ET
	models.append(ExtraTreesClassifier(n_estimators=1000))
	names.append('ET')
	return models, names

# define models
models, names = get_models()
results = list()

# evaluate a model
def evaluate_model(X, y, model):
 	# define evaluation procedure
 	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
 	# evaluate model
 	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
 	return scores

# evaluate each model
for i in range(len(models)):
	# evaluate the model and store results
	scores = evaluate_model(X, y, models[i])
	results.append(scores)
	# summarize performance - the mean and standard deviation classification accuracy
	print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
        
# plot the results
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()

# =============================================================================

# =============================================================================
# # Separate data into test and training sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
# 
# 
# # =============================================================================
# 
# # define Random Forest Model Classifier to evaluate
# modelRF = RandomForestClassifier(n_estimators=1000)
# 
# # fit the model
# modelRF.fit(X_train, y_train)
# 
# # Make prediction
# y_pred = modelRF.predict(X_test)
# 
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cmRF = confusion_matrix(y_test, y_pred)
# 
# 
# # =============================================================================
# 
# # define the Extra Tree Ensemble Model Classfier to evaluate
# modelET = ExtraTreesClassifier()
# 
# # fit the model on the whole dataset
# modelET.fit(X_train, y_train)
# 
# # Make prediction
# y_pred = modelET.predict(X_test)
# 
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cmET = confusion_matrix(y_test, y_pred)
# =============================================================================

# =============================================================================

# =============================================================================
# # get a list of models to evaluate
# def get_classifiers():
# 	classifiers = dict()
# 	classifiers['10'] = ExtraTreesClassifier(n_estimators=10)
# 	classifiers['50'] = ExtraTreesClassifier(n_estimators=50)
# 	classifiers['100'] = ExtraTreesClassifier(n_estimators=100)
# 	classifiers['500'] = ExtraTreesClassifier(n_estimators=500)
# 	classifiers['1000'] = ExtraTreesClassifier(n_estimators=1000)
# 	classifiers['5000'] = ExtraTreesClassifier(n_estimators=5000)
# 	return classifiers
# 
# # evaluate the model
# def evaluate_classifier(classifier):
#     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#     n_scores = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
#     return n_scores
# 
# # get the models to evaluate
# classifiers = get_classifiers()
# 
# # evaluate the models and store results
# results, names = list(), list()
# for name, classifier in classifiers.items():
# 	n_scores = evaluate_classifier(classifier)
# 	results.append(n_scores)
# 	names.append(name)
# 	print('>%s %.3f (%.3f)' % (name, mean(n_scores), std(n_scores)))
#     
# # plot model performance for comparison
# pyplot.boxplot(results, labels=names, showmeans=True)
# pyplot.show()
# 
# =============================================================================
