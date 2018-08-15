
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


baseball = pd.read_csv("./data/baseball.csv")
baseball.head(10)


# In[5]:


baseball.sort_values('Tm', ascending=False,na_position='first').head()


# In[6]:


baseball.sort_values(['Tm', 'Lg'], ascending=False,na_position='first').head()


# In[9]:


baseball.sort_values(['Tm', 'Lg'], ascending=[False,True],na_position='first').head()


# In[11]:


baseball.sort_index(ascending=False,inplace=True)
baseball.head()

