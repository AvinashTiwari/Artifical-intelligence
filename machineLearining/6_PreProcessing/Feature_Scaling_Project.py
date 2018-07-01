
# coding: utf-8

# In[1]:


import pandas as pd
from  sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

url ="winequality-red.csv"
dataframe = pd.read_csv(url, sep=";")
dataframe.head()


# In[4]:


x = dataframe.quality
x = dataframe.drop('quality', axis=1)
x


# In[7]:


y = dataframe.quality
X = dataframe.drop('quality', axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=123, stratify=y)


# In[10]:


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)


# In[11]:


clf = Ridge()
clf.fit(X_train,y_train)


# In[12]:


predection = clf.predict(X_test)
print(r2_score(y_test,predection))
print(mean_squared_error(y_test,predection))


# In[13]:


clf2 = Ridge()
clf2.fit(X_train,y_train)
predection2 = clf2.predict(x_test_scaled)
print(r2_score(y_test,predection2))
print(mean_squared_error(y_test,predection2))


