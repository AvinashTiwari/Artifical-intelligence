
# coding: utf-8

# In[5]:


import pandas as pd
pokemon = pd.read_csv("./data/Pokemon.csv", encoding='ISO-8859-1', index_col='Name')
pokemon.head(10)


# In[6]:


pokemon.shape


# In[7]:


pokemon = pd.read_csv("./data/Pokemon.csv", encoding='ISO-8859-1')
pokemon.head(10)


# In[9]:


pokemon.set_index('Name',inplace=True)


# In[10]:


pokemon.head()


# In[12]:


pokemon.reset_index(drop=False,inplace=True)
pokemon.head()

