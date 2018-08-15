
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset = pd.read_csv("./data/transit-segments.csv")
dataset.head(5)


# In[3]:


datasetnew = pd.read_csv("./data/transit-segments.csv", index_col='name', usecols=['name','min_sog', 'max_sog'])
datasetnew.head()


# In[4]:


datasetnew.min_sog


# In[5]:


datasetnew.min_sog.head()


# In[6]:


datasetnew.min_sog[:10]


# In[7]:


datasetnew['min_sog']


# In[8]:


datasetnew[['min_sog', 'max_sog']]


# In[9]:


datasetnew.head()


# In[10]:


datasetnew['country'] = "USA"
datasetnew.head()


# In[11]:


datasetnew.insert(3, column='country2', value="India")
datasetnew.head()


# In[14]:


chosen = ['seg_length', 'avg_sog','min_sog','max_sog']
df =dataset[chosen]
df.head()


# In[15]:


df.sum(axis=1)


# In[16]:


df.min_sog.add(10)


# In[17]:


df.min_sog.mul(10)


# In[18]:


df.min_sog.div(10)

