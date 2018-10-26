
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf


# In[20]:


titanic = pd.read_csv("./data/titanic.csv")
titanic.head(10)


# In[21]:


titanic.drop(labels=['PassengerId','Name', 'Ticket','Cabin', 'Embarked'], axis=1, inplace=True)
titanic.head()


# In[22]:


titanic.dtypes


# In[23]:


convertnumeric= {'male':1 , 'female':0}
titanic.Sex = titanic.Sex.map(convertnumeric)
titanic.head()


# In[24]:


titanic.dtypes


# In[25]:


titanic.isnull().sum()


# In[26]:


titanic.shape


# In[27]:


titanic.fillna(value=titanic.Age.mean(), inplace=True)
titanic.isnull().sum()


# In[28]:


titanic.head()


# In[30]:


tree = DecisionTreeClassifier()
survived = titanic['Survived']
survived.shape


# In[13]:


titanic.drop(['Survived'],axis=1, inplace=True)
titanic.head()


# In[14]:


X=titanic[:800]
X


# In[29]:


tree.fit(X=titanic[:800], y=survived[:800])
tree.score(X=titanic[800:], y=survived[800:])


# In[36]:


me = np.array([3,3,0,26,0,8])
me = me.reshape(1,-1)
tree.predict(me)


# In[ ]:


input_shape = 5
x_input = tf.placeholder(dtype=tf.float32, shape=[None, input_shape], name='x_input')
y_input = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='y_input')

