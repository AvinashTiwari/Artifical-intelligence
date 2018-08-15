
# coding: utf-8

# In[1]:


import pandas as pd


# In[5]:


dataset = pd.read_csv("./data/transit-segments.csv")
dataset.head(5)


# In[6]:


dataset.tail(5)


# In[8]:


dataset.shape


# In[10]:


dataset.dtypes


# In[11]:


dataset.describe()

