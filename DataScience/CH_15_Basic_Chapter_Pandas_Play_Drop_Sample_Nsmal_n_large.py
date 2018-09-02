
# coding: utf-8

# In[42]:


import pandas as pd
pokemon = pd.read_csv("./data/Pokemon.csv", encoding='ISO-8859-1')
pokemon.head(10)


# In[43]:


pokemon.pop('Stage')
pokemon.head()


# In[44]:


del pokemon['Total']
pokemon.head()


# In[45]:


pokemon.drop(labels=['Type 2','Legendary'],axis='columns', inplace=True)
pokemon.head()


# In[46]:


pokemon.sample(10, axis=0)


# In[47]:


pokemon.nlargest(10, columns='Attack')


# In[49]:


pokemon.nsmallest(10, columns='Defense')


# In[51]:


pokemon['Speed'].nlargest(10)

