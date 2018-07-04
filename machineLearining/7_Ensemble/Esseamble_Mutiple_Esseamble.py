
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import Binarizer
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier



url = "pima_indians_diabetes.csv"
names = ['preg', 'plas','pres','skin','test','mass','pedi', 'age','class']
dataframe = pd.read_csv(url, names=names)
dataframe.shape
dataframe.head


# In[8]:


array = dataframe.values
X = array[:, 0:8]
y = array[:,8]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)


# In[9]:


estimators =[]
model7 = LogisticRegression()
estimators.append(('logistic',model7))
model8 = DecisionTreeClassifier()
estimators.append(('cart',model8))
model9 = SVC()
estimators.append(('svm',model9))


# In[11]:


ensmble = VotingClassifier(estimators)
result7 = model_selection.cross_val_score(ensmble, X, y, cv= kfold)
print(result7)
print("Result for mean \n")
print(result7.mean())

