
# coding: utf-8

# In[1]:


import pandas as pd
import datetime as dt
from pandas_datareader import data as web


# In[4]:


pokemon = pd.read_csv('./data/Pokemon.csv',encoding='iso-8859-1')
pokemon.head()


# In[5]:


pokemon.memory_usage(deep=True)


# In[6]:


pokemon.dtypes


# In[7]:


pokemon['Type 1'] = pokemon['Type 1'].astype('category')


# In[8]:


pokemon.memory_usage(deep=True)


# In[9]:


pokemon['Name'] = pokemon['Name'].astype('category')


# In[10]:


pokemon.memory_usage(deep=True)


# In[13]:


pokemon[pokemon['Type 2'].isnull()]['Type 2'] = "Flying"


# In[15]:


pokemon.isnull().sum()


# In[18]:


pokemon.loc[pokemon['Type 2'].isnull(), 'Type 2'] = "Flying"


# In[19]:


pokemon.isnull().sum()

