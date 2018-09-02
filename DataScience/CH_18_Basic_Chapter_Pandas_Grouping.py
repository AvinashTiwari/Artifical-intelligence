
# coding: utf-8

# In[1]:


import pandas as pd
import datetime as dt


ipl = pd.read_csv("./data/matches.csv")


# In[2]:


ipl.head()


# In[3]:


ipl.shape


# In[4]:


ipl.describe()


# In[5]:


groupByCity = ipl.groupby('city')
groupByCity


# In[6]:


groupByCity.size


# In[7]:


groupByCity.size()


# In[8]:


groupByCity.sum()


# In[9]:


groupByCity['win_by_runs'].sum()


# In[10]:


groupByCity[['win_by_runs','win_by_wickets']].sum()


# In[12]:


banaglore = groupByCity.get_group('Bangalore')['win_by_runs'].sum()
banaglore


# In[13]:


banaglore.mean()


# In[14]:


newgroup =ipl.groupby(['city', 'toss_decision'])
newgroup.size()


# In[15]:


groupByCity.agg({'win_by_runs':'sum','win_by_wickets':'mean'})

