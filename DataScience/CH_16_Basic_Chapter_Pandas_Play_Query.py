
# coding: utf-8

# In[1]:


import pandas as pd
pokemon = pd.read_csv("./data/Pokemon.csv", encoding='ISO-8859-1')
pokemon.head(10)


# In[2]:


pokemon.query('Name =="Bulbasaur" ')


# In[3]:


pokemon.query('Name !="Bulbasaur" ')


# In[4]:


pokemon.query('HP > 80 ')


# In[6]:


pokemon.query('HP > 80  or Attack > 20')


# In[8]:


pokemon.query('Name in (["Aerodactyl" , "Mew"])')

