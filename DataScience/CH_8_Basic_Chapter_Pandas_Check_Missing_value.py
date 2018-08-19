
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


titanic = pd.read_csv("./data/titanicexcel.csv", sep=';')
titanic.head(10)


# In[4]:


titanic.isnull().sum()


# In[5]:


titanic.shape


# In[6]:


titanic['Cabin'].isnull()


# In[7]:


filter6 = titanic['Cabin'].isnull()


# In[8]:


titanic[filter6].head()


# In[10]:


filter7 = titanic['Cabin'].notnull()


# In[11]:


titanic[filter7].head()

