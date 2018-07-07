
# coding: utf-8

# In[1]:


import numpy as np
from sklearn import datasets 
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[2]:


datasets = datasets.load_diabetes()
alphas = np.array([1,0.1,0.01,0.001,0.0001,0])


# In[3]:


model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
grid.fit(datasets.data, datasets.target)
print(grid)


# In[4]:


print(grid.best_score_)


# In[5]:


print(grid.best_estimator_.alpha)


# In[7]:


import numpy as np
from scipy.stats import uniform as sp_rand
from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

datasets = datasets.load_diabetes()
param_grid = {'alpha': sp_rand()}
model = Ridge()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)


# In[8]:


rsearch.fit(datasets.data,datasets.target )
print(rsearch)


# In[10]:


print(rsearch.best_score_)


# In[13]:


print(rsearch.best_estimator_.alpha)

