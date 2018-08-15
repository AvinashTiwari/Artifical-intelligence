
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


baseball = pd.read_csv("./data/baseball.csv")
baseball.head(10)


# In[3]:


baseball.shape


# In[4]:


baseball.dropna(inplace=True)


# In[5]:


baseball.shape


# In[6]:


baseball = pd.read_csv("./data/baseball.csv")
baseball.dropna(how='all', inplace=True)
baseball.shape


# In[7]:


baseball.dropna(how='any', inplace=True)
baseball.shape


# In[8]:


baseball = pd.read_csv("./data/baseball.csv")
baseball.dropna(subset=['W','L'], inplace=True)
baseball.shape


# In[9]:


baseball.dropna(subset=['W','Attendance'], inplace=True)
baseball.shape


# In[10]:


baseball.dropna(subset=['W','Attendance'], how ='all' ,inplace=True)
baseball.shape


# In[11]:


baseball = pd.read_csv("./data/baseball.csv", usecols=['Tm', 'Playoffs','Attendance'])
baseball.head(10)


# In[12]:


baseball.shape


# In[13]:


baseball.fillna(0, inplace=True)
baseball.head(10)


# In[14]:


baseball.isnull().sum()


# In[15]:


baseball.info()


# In[16]:


baseball = pd.read_csv("./data/baseball.csv", usecols=['Tm', 'Playoffs','Attendance'])
baseball.head(10)


# In[17]:


baseball.fillna(method='bfill', inplace=True)
baseball.head(10)


# In[18]:


baseball = pd.read_csv("./data/baseball.csv", usecols=['Tm', 'Playoffs','Attendance'])
baseball.head(10)


# In[19]:


baseball.fillna(method='ffill', inplace=True)
baseball.head(10)


# In[21]:


baseball = pd.read_csv("./data/baseball.csv")
baseball.head(10)


# In[23]:


baseball.dtypes


# In[24]:


baseball['Year'].astype(float)


# In[25]:


baseball.head()


# In[26]:


baseball['Year'] = baseball['Year'].astype(float)


# In[27]:


baseball.head()


# In[28]:


baseball.dtypes


# In[29]:


baseball['W'] = baseball['W'].astype(str)
baseball.head()


# In[30]:


baseball.dtypes

