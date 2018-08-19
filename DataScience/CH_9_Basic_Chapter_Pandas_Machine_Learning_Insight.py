
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


# In[15]:


titanic = pd.read_csv("./data/titanic.csv")
titanic.head(10)


# In[16]:


titanic.drop(labels=['PassengerId','Name', 'Ticket','Cabin', 'Embarked'], axis=1, inplace=True)
titanic.head()


# In[17]:


titanic.dtypes


# In[18]:


convertnumeric= {'male':1 , 'female':0}
titanic.Sex = titanic.Sex.map(convertnumeric)
titanic.head()


# In[19]:


titanic.dtypes


# In[20]:


titanic.isnull().sum()


# In[21]:


titanic.shape


# In[23]:


titanic.fillna(value=titanic.Age.mean(), inplace=True)
titanic.isnull().sum()


# In[24]:


titanic.head()


# In[25]:


tree = DecisionTreeClassifier()
survived = titanic['Survived']
survived.head()


# In[26]:


titanic.drop(['Survived'],axis=1, inplace=True)
titanic.head()


# In[29]:


tree.fit(X=titanic[:800], y=survived[:800])
tree.score(X=titanic[800:], y=survived[800:])


# In[36]:


me = np.array([3,3,0,26,0,8])
me = me.reshape(1,-1)
tree.predict(me)

