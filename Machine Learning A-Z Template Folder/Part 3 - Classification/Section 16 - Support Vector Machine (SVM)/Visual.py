# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 03:49:53 2020

@author: vanda
"""

#Import the libraries
import pandas as pd  
import matplotlib.pyplot as plt


#Read the input data data from the external CSV
dataset = pd.read_csv('Pohang.csv')

#Take a look at the data
dataset.head()
dataset.info()

print(dataset.shape)

print(dataset['Damage Class'].unique())

print(dataset.groupby('Damage Class').size())

import seaborn as sns
sns.countplot(dataset['Damage Class'], palette = "Blues")
plt.title("Buildings' distribution over different damage category")
plt.xlabel('Damage Class')
plt.ylabel('Sample Buildings')
plt.show()

import pylab as pl
dataset.drop('Damage Class',axis=1).hist(bins=30, figsize=(9,9), color = 'steelblue')
pl.suptitle("Histogram for each numeric input parameter")
plt.savefig('damage_hist')
plt.show()





























