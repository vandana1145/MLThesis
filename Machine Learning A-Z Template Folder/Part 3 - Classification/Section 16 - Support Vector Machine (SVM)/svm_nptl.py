# Importing the libraries
from __future__ import division, print_function
import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

# Importing the dataset
dataset = pd.read_csv('Ehsan Duzce.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 7].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

def evaluate_on_test_data(model = None):
    predictions = model.predict(X_test)
    correct_classifications = 0
    for i in range(len(y_test)):
        if predictions[i] == y_test[i]:
            correct_classifications += 1
    accuracy = 100*correct_classifications/len(y_test)
    return accuracy

kernels = ('linear', 'poly', 'rbf')
accuracies = []
for index, kernel in enumerate(kernels):
    model = svm.SVC(kernel=kernel)
    model.fit(X_train, y_train)
    acc = evaluate_on_test_data(model)
    accuracies.append(acc)
    print("{} % accuracy obtained with kernel = {}".format(acc, kernel))
    

# Train SVMs with different kernels
svc = svm.SVC(kernel = 'linear').fit(X_train, y_train)
rbf_svc = svm.SVC(kernel = 'rbf', gamma = 0.7).fit(X_train, y_train)
poly_svc = svm.SVC(kernel = 'poly', degree = 3).fit(X_train, y_train)

# Create a mesh to plot in
h = 0.02 #step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#Define title for the plots
titles = ['SVC with Linear Kernel', 'SVC with RBF Kernel', 'SVC with polynomial (degree = 3) Kernel' ]

for i, clf in enumerate((svc, rbf_svc, poly_svc)):
    plt.figure(i)

z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

#Plot results into a colorful plot
z = z.reshape(xx.shape)
plt.contourf(xx, yy, z, cmap = plt.cm.Paired, alpha=0.8)

#Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.ocean)
plt.xlabel('length')
plt.ylabel('width')
plt.xlim(xx.min(), xx.max())
plt.ylim((yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title(titles[i])

plt.show()



















