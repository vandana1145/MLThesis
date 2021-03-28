from mlxtend.classifier import EnsembleVoteClassifier
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
import pandas as pd

dataset = pd.read_csv('Duzce.csv')
dataset.head()
dataset.info()

X = dataset[['System Type', 'Year of cons.', 'No. of Story', 'Ground floor Area', 'Total Floor Area',
      'Overhang Area', 'Story Heights Ground', 'Story Heights Normal', 'Irr-A1', 'Irr-A2', 'Irr-A3',
      'Irr-A4', 'Irr-B1', 'Irr-B2', 'Irr-B3', 'X-dir - Frames', 'Y-dir - Frames', 'SSI',
      'OR (Aover/Agrou)', 'RNS', 'MNLSI', 'MNLSTFI']].to_numpy() 
y = dataset[['Damage']].to_numpy()


X = dataset.data[:, :22]
y = dataset.target


svm = SVC(C=0.5, kernel='linear')
svm.fit(X, y)

# Plotting decision regions
EnsembleVoteClassifier(X, y, clf=svm, legend=2)

# Adding axes annotations
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.title('SVM on Iris_linear')
plt.show()