
# coding: utf-8

# In[1]:


import pandas as pd
pokemon = pd.read_csv("./data/Pokemon.csv", encoding='ISO-8859-1', index_col='Name')
pokemon.head(10)


# In[2]:


pokemon.shape


# In[4]:


pokemon.set_index('Name',inplace=True)


# In[8]:


pokemon.loc['Venusaur']


# In[9]:


pokemon.loc['Bulbasaur':'Charmander']


# In[11]:


pokemon.loc[['Bulbasaur','Charmander']]


# In[13]:


pokemon.loc[['Bulbasaur','Charmander'], 'Type 1' : 'HP']


# In[15]:


pokemon.loc[['Bulbasaur','Charmander'], ['Attack' , 'Defense','Sp. Atk']]


# In[17]:


pokemon.tail()


# In[18]:


'New' in pokemon.index

