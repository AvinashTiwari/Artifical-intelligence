
# coding: utf-8

# In[1]:


import pandas as pd
pokemon = pd.read_csv("./data/Pokemon.csv", encoding='ISO-8859-1', index_col='Name')
pokemon.head(10)


# In[2]:


pokemon.shape


# In[3]:


pokemon.loc['Venusaur']


# In[4]:


pokemon.loc['Bulbasaur':'Charmander']


# In[5]:


pokemon.loc[['Bulbasaur','Charmander']]


# In[6]:


pokemon.loc[['Bulbasaur','Charmander'], 'Type 1' : 'HP']


# In[7]:


pokemon.loc[['Bulbasaur','Charmander'], ['Attack' , 'Defense','Sp. Atk']]


# In[8]:


pokemon.tail()


# In[9]:


'New' in pokemon.index


# In[10]:


pokemon2 = pd.read_csv("./data/Pokemon.csv", encoding='ISO-8859-1')
pokemon2.head(10)


# In[11]:


pokemon2.loc[3,'Attack':'Defense']


# In[12]:


pokemon2.iloc[3]


# In[13]:


pokemon2.iloc[0:3]


# In[14]:


pokemon2.iloc[0:3, 2]


# In[15]:


pokemon2.iloc[0:3, 2:5]


# In[16]:


pokemon2.iloc[[4,10,22]]


# In[17]:


pokemon2.set_index('Name',inplace=True)
pokemon2.head()


# In[18]:


pokemon2.ix['Bulbasaur']


# In[19]:


pokemon2.ix[0]


# In[20]:


pokemon2.reset_index(drop=False, inplace=True)
pokemon2.head()


# In[21]:


pokemon2.ix[0,0]


# In[22]:


pokemon2.ix[0,0] = 'Avinash'
pokemon2.head()


# In[23]:


pokemon2.ix[0,['Attack','Defense','Speed']] = [100,100,100]
pokemon2.head()


# In[24]:


pokemon2[pokemon2['Type 1'] == "Fire"]


# In[25]:


pokemon2.ix[pokemon2['Type 1'] == "Fire"]


# In[26]:


pokemon3 = pd.read_csv("./data/Pokemon.csv", encoding='ISO-8859-1',names=[1,2,3,4,5,6,7,8,9,10,11,12,13])
pokemon3.head(10)


# In[ ]:


pokemon3 = pd.read_csv("./data/Pokemon.csv", encoding='ISO-8859-1',names=[1,2,3,4,5,6,7,8,9,10,11,12,13])
pokemon3.head(10)

