
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn import model_selection 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

 

url = "pima_indians_diabetes.csv"
names = ['preg', 'plas','pres','skin','test','mass','pedi', 'age','class']
dataframe = pd.read_csv(url, names=names)
dataframe.shape
dataframe.head


# In[8]:


array = dataframe.values
X = array[:,0:8]
y = array[:,8]
#defaut is 80% train 20% Test data
X_train , X_test , y_train , y_test = train_test_split(X,y, random_state = 7)
np.set_printoptions(precision=3)


# In[10]:


kfold = KFold(n_splits=10 , random_state=7)
model = LogisticRegression()
model.fit(X_train,y_train)
result =  model.score(X_test, y_test)
print(result)


# In[12]:


model2  = LogisticRegression()
results = cross_val_score(model2, X,y, cv = kfold)
print(results)
print(results.mean())

