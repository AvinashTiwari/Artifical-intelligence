
# coding: utf-8

# In[3]:


import pandas as pd
transit = pd.read_csv("./data/transit-segments.csv")
transit.head(10)


# In[4]:


transit.dtypes


# In[6]:


transit1 = transit
transit1.drop_duplicates(subset=['name'], keep=False)


# In[7]:


transit1.head()


# In[8]:


transit1['name'].unique()


# In[9]:


transit1['segment'].nunique()


# In[10]:


transit1['segment'].max()

