# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 02:25:46 2020

@author: vanda
"""

# Pandas and seaborn for data manipulation
import pandas as pd
import seaborn as sns
import numpy as np

dataset = pd.read_csv('Ecuador.csv')
dataset.describe()        

# set the backgroud stle of the plot 
sns.set_style('darkgrid') 

# =============================================================================
# # plot the graph using the default estimator mean 
# sns.barplot(x ='Damage Class', y ='No. Floor', data = dataset, hue='Damage Class', palette ='plasma', saturation=1, capsize=0.2, dodge=False) 
# 
# =============================================================================

# plot the graph using the default estimator mean 
ax = sns.barplot(x ='Damage Class', y ='No. Floor', data = dataset, hue='Damage Class', estimator=np.max, palette ='Blues', dodge=False) 
ax.set(ylim=(0, 7))


# =============================================================================
# # plot the graph using the default estimator mean 
# sns.violinplot(x ='Damage Class', y ='No.Floor', data = dataset, hue='Damage Class', cut = 1, palette ='plasma') 
# 
# =============================================================================

ax = sns.barplot(x ='Damage Class', y ='Total Floor Area', data = dataset, hue='Damage Class', estimator=np.max, palette ='Blues', dodge=False) 
ax.set(ylim=(0, 3500))


ax = sns.barplot(x ='Damage Class', y ='Column Area', data = dataset, hue='Damage Class', estimator=np.max, palette ='Blues', dodge=False) 
ax.set(ylim=(0, 10))


ax = sns.barplot(x ='Damage Class', y ='Concrete Wall Area-NS', data = dataset, hue='Damage Class', estimator=np.max, palette ='Blues', dodge=False) 
ax.set(ylim=(0, 3))


ax = sns.barplot(x ='Damage Class', y ='Concrete Wall Area-EW', data = dataset, hue='Damage Class', estimator=np.max, palette ='Blues', dodge=False) 
ax.set(ylim=(0, 3))


ax = sns.barplot(x ='Damage Class', y ='Masonry Wall Area-NS', data = dataset, hue='Damage Class', estimator=np.max, palette ='Blues', dodge=False) 
ax.set(ylim=(0, 14))


ax = sns.barplot(x ='Damage Class', y ='Masonry Wall Area-EW', data = dataset, hue='Damage Class', estimator=np.max, palette ='Blues', dodge=False) 
ax.set(ylim=(0, 14))


ax = sns.barplot(x ='Damage Class', y ='Captive Columns', data = dataset, hue='Damage Class', estimator=np.max, palette ='Blues', dodge=False) 
ax.set(ylim=(0, 1))


# =============================================================================
# 
# # plot the graph using the default estimator mean 
# sns.violinplot(x ='Damage Class', y ='Total Floor Area', data = dataset, hue='Damage Class', cut = 0, palette ='plasma') 
# 
# # plot the graph using the default estimator mean 
# sns.violinplot(x ='Damage Class', y ='Column Area', data = dataset, hue='Damage Class', cut = 0, palette ='plasma') 
# 
# # plot the graph using the default estimator mean 
# sns.violinplot(x ='Damage Class', y ='Concrete Wall Area-NS', data = dataset, hue='Damage Class', cut = 0, palette ='plasma') 
# 
# # plot the graph using the default estimator mean 
# sns.violinplot(x ='Damage Class', y ='Concrete Wall Area-EW', data = dataset, hue='Damage Class', cut = 0, palette ='plasma') 
# 
# # plot the graph using the default estimator mean 
# sns.violinplot(x ='Damage Class', y ='Masonry Wall Area-NS', data = dataset, hue='Damage Class', cut = 0, palette ='plasma') 
# 
# # plot the graph using the default estimator mean 
# sns.violinplot(x ='Damage Class', y ='Masonry Wall Area-EW', data = dataset, hue='Damage Class', cut = 0, palette ='plasma') 
# 
# # plot the graph using the default estimator mean 
# sns.violinplot(x ='Damage Class', y ='Captive Columns', data = dataset, hue='Damage Class', cut = 0, palette ='plasma') 
# 
# =============================================================================














