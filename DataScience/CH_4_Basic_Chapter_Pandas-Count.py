
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


baseball = pd.read_csv("./data/baseball.csv")
baseball.head(10)


# In[6]:


baseball.describe()


# In[7]:


baseball.info()


# In[8]:


baseball.isnull()


# In[9]:


baseball.isnull().sum()


# In[12]:


baseball.shape


# In[14]:


baseball['Tm'].value_counts()


# In[15]:


baseball['Tm'].value_counts().head(3)


# In[16]:


baseball['Tm'].value_counts().tail(3)

