
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

url = "pima_indians_diabetes.csv"
names = ['preg', 'plas','pres','skin','test','mass','pedi', 'age','class']


# In[15]:


dataframe = pd.read_csv(url , names=names)
array = dataframe.values
X = array[:,0:8]
y = array[:,8]

estimators = []
estimators.append(('standardixe',StandardScaler()))
estimators.append(('ida',LinearDiscriminantAnalysis()))
model =  Pipeline(estimators)
seed = 7


# In[17]:


kfold = KFold(n_splits=10, random_state=seed)
result = cross_val_score(model,X,y, cv=kfold)
print(result)
print(result.mean())

