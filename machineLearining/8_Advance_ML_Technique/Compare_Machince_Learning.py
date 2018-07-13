
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "pima_indians_diabetes.csv"
names = ['preg', 'plas','pres','skin','test','mass','pedi', 'age','class']

dataframe = pd.read_csv(url, names=names)
array = dataframe.values
X =  array[:,0:8]
y = array[:,8]
seed =7
models = []
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))

results =[]
scorning ="accuracy"
names= []
for name,model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model,X,y,cv=kfold, scoring=scorning)
    results.append(cv_results)
    names.append(name)
    msg = "%s : %f (%f)" %(name,  cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure()
fig.suptitle('Algo comparsion')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

