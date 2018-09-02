
# coding: utf-8

# In[1]:


import pandas as pd
import datetime as dt


pokemon = pd.read_csv('./data/Pokemon.csv', encoding='ISO-8859-1')
pokemon.head()


# In[2]:


pokemon.describe()


# In[3]:


pokemon.shape


# In[4]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


pokemon.hist()


# In[6]:


pokemon.set_index(keys=['Name', 'Type 1'], inplace=True)
pokemon.head()


# In[8]:


pokemon.sort_index(inplace=True)
pokemon.head()


# In[9]:


pokemon.index


# In[10]:


pokemon.index.names


# In[11]:


pokemon.index[0]


# In[14]:


pokemon.index.get_level_values(0)


# In[15]:


pokemon.index.set_names(['Pokemonname','pokemontype'], inplace=True)
pokemon.head()


# In[16]:


pokemon_transposed = pokemon.transpose()
pokemon_transposed.head()


# In[17]:


pokemon.head()


# In[18]:


pokemon.swaplevel()


# In[19]:


pokemon.sort_index(ascending=[True, False], inplace=True)
pokemon.head()

