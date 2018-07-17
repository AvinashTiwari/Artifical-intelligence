
# coding: utf-8

# In[1]:


import pandas as pd
dataset = pd.read_csv(r'.\pima_indians_diabetes.csv', header =None)
print(dataset.head(15))


# In[2]:


print((dataset[[1,2,3,4,5]] ==0).sum())


# In[5]:


import numpy as np
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, np.NaN)
print(dataset.head(15))


# In[7]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

values = dataset.values
X =  values[:,0:8]
y = values[:,8]
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits =3, random_state = 7)
result = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
print(result.mean())


# In[8]:


import numpy as np
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, np.NaN)
dataset.dropna(inplace=True)
print(dataset.head(15))


# In[9]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

values = dataset.values
X =  values[:,0:8]
y = values[:,8]
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits =3, random_state = 7)
result = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
print(result.mean())


# In[10]:


import numpy as np
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, np.NaN)
dataset.fillna(dataset.mean(),inplace=True)
print(dataset.head(15))


# In[11]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

values = dataset.values
X =  values[:,0:8]
y = values[:,8]
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits =3, random_state = 7)
result = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
print(result.mean())


# In[13]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing   import Imputer
dataset = pd.read_csv(r'.\pima_indians_diabetes.csv', header =None)
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, np.NaN)

values = dataset.values
imputer = Imputer()
values = imputer.fit_transform(values)

X =  values[:,0:8]
y = values[:,8]
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits =3, random_state = 7)
result = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
print(result.mean())

