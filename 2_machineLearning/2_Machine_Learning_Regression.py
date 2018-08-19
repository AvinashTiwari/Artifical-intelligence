
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# # Regression
# 

# In[3]:


from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()


# In[4]:


print(boston.DESCR)


# In[5]:


boston.data.shape


# In[6]:


boston.target.shape


# In[7]:


from  sklearn.cross_validation import train_test_split
X_train , X_test, y_train, y_test = train_test_split(boston.data,boston.target)


# # Learning a Regressor

# In[9]:


from sklearn.linear_model import Ridge


# In[10]:


ridge = Ridge()


# In[11]:


ridge.fit(X_train, y_train)


# In[13]:


pred_test = ridge.predict(X_test)
pred_test


# In[14]:


ridge.score(X_test,y_test)


# # MSE:

# In[15]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,pred_test)


# # Random Forest Regression

# In[16]:


from sklearn.ensemble import RandomForestRegressor


# In[17]:


rf = RandomForestRegressor()


# In[18]:


rf.fit(X_train,y_train)


# In[19]:


rf.score(X_test, y_test)


# In[20]:


mean_squared_error(y_test,rf.predict(X_test))

