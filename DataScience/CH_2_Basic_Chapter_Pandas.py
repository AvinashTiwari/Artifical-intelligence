
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset = pd.read_csv("./data/transit-segments.csv")
dataset.head(5)


# In[3]:


dataset.index


# In[4]:


dataset.values


# In[5]:


dataset.columns


# In[6]:


dataset.axes


# In[7]:


dataset.info()


# In[9]:


dataset.get_dtype_counts()

