
# coding: utf-8

# In[1]:


import pandas as pd
transit = pd.read_csv("./data/transit-segments.csv")
transit.head(10)


# In[2]:


transit.dtypes


# In[3]:


transit1 = pd.read_csv("./data/transit-segments.csv", parse_dates=['st_time'])
transit1.head(10)


# In[4]:


transit1.dtypes


# In[6]:


transit1.end_time = pd.to_datetime(transit1.end_time)


# In[7]:


transit1.head()


# In[10]:


transit1.dtypes


# In[12]:


transit1[transit1['min_sog'].between(10.0,20.0)]
transit1.head()


# In[13]:


transit1[transit1['st_time'].between("2011-01-01","2011-12-31")]

