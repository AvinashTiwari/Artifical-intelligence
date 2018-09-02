
# coding: utf-8

# In[8]:


import pandas as pd
import datetime as dt
from pandas_datareader import data as web


# In[10]:


sales = pd.read_csv('./data/sales2.csv')
sales.head()


# In[12]:


sales['Title']  = sales['Title'].str.lower()


# In[13]:


sales.head()


# In[14]:


sales['Title']  =sales['Title'].str.upper()


# In[15]:


sales.head()


# In[16]:


sales['Title']  = sales['Title'].str.title()


# In[17]:


sales.head()


# In[19]:


sales['Title'].str.len()


# In[21]:


sales['Title'] = sales['Title'].str.replace('Sales', 'Revenue')
sales.head()


# In[22]:


sales['Title'].str.contains('the')


# In[23]:


sales = pd.read_csv('./data/sales2.csv')


# In[24]:


sales['Currecny'] = sales['Title'].str.extract("\((.*)\)", expand=False).fillna(method="bfill")
sales

