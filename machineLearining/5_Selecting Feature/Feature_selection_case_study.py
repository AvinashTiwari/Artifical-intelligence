
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.feature_selection  import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
url = "pima_indians_diabetes.csv"
names = ['preg', 'plas','pres','skin','test','mass','pedi', 'age','class']
dataframe = pd.read_csv(url, names=names)
dataframe.shape
dataframe.head


# In[4]:


array = dataframe.values
X = array[:, 0:8]
y = array[:,8]


# In[7]:


seed = np.random.seed(3)
X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=seed)


# In[8]:


model = LogisticRegression()


# In[10]:


model.fit(X_train,y_train)
model.score(X_test, y_test)


# In[11]:


rfe = RFE(model, 6)
training = rfe.fit(X_train, y_train)


# In[12]:


applied = training.transform(X_train)


# In[13]:


applied


# In[14]:


model2 = LogisticRegression()
model2.fit(applied,y_train)


# In[15]:


#model2.score(X_test, y_test)
testdata = training.transform(X_test)


# In[16]:


model2.score(testdata, y_test)


# In[18]:


me = np.array([0,150,70,30,0,40,0.5,25])
me_adjusted = me.reshape(1,-1)
me_adjusted = training.transform(me_adjusted)


# In[19]:


me_adjusted


# In[20]:


model2.predict(me_adjusted)

