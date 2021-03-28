import pandas as pd
import seaborn as sns
from collections import Counter
from numpy import mean
from numpy import std
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split

sns.set_style('whitegrid')

df = pd.read_csv('C:/Users/vanda/.spyder-py3/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 19 - Decision Tree Classification/Malawi.csv')

dataframe = df.groupby(['F_Length','F_Height','Gable','Gable_height','tot_area_opening','s_height','F_thickness','n_intwall_perp_f','n_intwall_paral_f','Length_perp_f','n_intwall_perp_back_f','Roof_A3_t_A1_m','P_O','Masonry_E_unf_B_fir','Mortar','length_brick','height_brick','staggering_brick','LEFT_Connec','rIGHT_Connec'])['Damage'].sum().reset_index()

dataframe.head()

df.head()

dataframe['Damage'].plot(kind='hist')

dataframe['Damage'].describe()

pd.qcut(dataframe['Damage'], q=10)

dataframe['quantile_ex'] = pd.qcut(dataframe['Damage'], q=10, precision=0)

dataframe.head()

dataframe['quantile_ex'].value_counts()

bin_labels_10 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
dataframe['quantile_ex_1'] = pd.qcut(dataframe['Damage'],
                              q=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                              labels=bin_labels_10)
dataframe.head()

dataframe['quantile_ex_1'].value_counts()

X = dataframe.iloc[:, :-3].values
y = dataframe.iloc[:, 22].values

#Rescale data (between 0 and 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)

# summarize the class distribution
target = dataframe.values[:,-3]
counter = Counter(target)
for k,v in counter.items():
	per = v / len(target) * 100
	print('Class=%s, Count=%s, Percentage=%.3f%%' % (k, v, per))
    
# summarize class distribution
print(rescaledX.shape, y.shape,Counter(y))

# label encode the target variable to have the classes 0 and 1
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

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
def evaluate_model(rescaledX, y, model):
 	# define evaluation procedure
 	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
 	# evaluate model
 	scores = cross_val_score(model, rescaledX, y, scoring='accuracy', cv=cv, n_jobs=-1)
 	return scores
 
# evaluate each model
for i in range(len(models)):
	# evaluate the model and store results
	scores = evaluate_model(rescaledX, y, models[i])
	results.append(scores)
	# summarize performance - the mean and standard deviation classification accuracy
	print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
        
# plot the results
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()

# Separate data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(rescaledX, y, test_size = 0.20)

# define the Extra Tree Ensemble Model Classfier to evaluate
modelET = ExtraTreesClassifier()

# fit the model on the whole dataset
modelET.fit(X_train, y_train)

# Make prediction
y_pred = modelET.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmET = confusion_matrix(y_test, y_pred)

# =============================================================================
# # get a list of models to evaluate
# def get_classifiers():
#  	classifiers = dict()
#  	classifiers['10'] = ExtraTreesClassifier(n_estimators=10)
#  	classifiers['50'] = ExtraTreesClassifier(n_estimators=50)
#  	classifiers['100'] = ExtraTreesClassifier(n_estimators=100)
#  	classifiers['500'] = ExtraTreesClassifier(n_estimators=500)
#  	classifiers['1000'] = ExtraTreesClassifier(n_estimators=1000)
#  	classifiers['5000'] = ExtraTreesClassifier(n_estimators=5000)
#  	return classifiers
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
#  	n_scores = evaluate_classifier(classifier)
#  	results.append(n_scores)
#  	names.append(name)
#  	print('>%s %.3f (%.3f)' % (name, mean(n_scores), std(n_scores)))
#     
# # plot model performance for comparison
# pyplot.boxplot(results, labels=names, showmeans=True)
# pyplot.show()
# 
# =============================================================================
