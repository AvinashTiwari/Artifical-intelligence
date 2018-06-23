
# coding: utf-8

# In[5]:


import pandas as pd
url = "pima_indians_diabetes.csv"
names = ['preg', 'plas','pres','skin','test','mass','pedi', 'age','class']
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
dataframe = pd.read_csv(url, names=names)
dataframe.shape
dataframe.head


# In[10]:


array = dataframe.values
X = array[:,0:8]
y = array[:,8]
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X,y)
import numpy as np
np.set_printoptions(precision=3)
print(fit.scores_)


# In[14]:


features = fit.transform(X)
print(features[0:5,:])

