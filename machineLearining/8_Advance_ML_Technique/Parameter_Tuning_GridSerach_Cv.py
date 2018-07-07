
# coding: utf-8

# In[1]:


import numpy as np
from sklearn import datasets 
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[2]:


datasets = datasets.load_diabetes()
alphas = np.array([1,0.1,0.01,0.001,0.0001,0])


# In[4]:


model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
grid.fit(datasets.data, datasets.target)
print(grid)


# In[9]:


print(grid.best_score_)


# In[10]:


print(grid.best_estimator_.alpha)

