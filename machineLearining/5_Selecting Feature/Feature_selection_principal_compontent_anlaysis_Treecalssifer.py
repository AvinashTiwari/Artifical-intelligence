
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.decomposition import PCA

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


pca = PCA(n_components= 3)


# In[8]:


fit = pca.fit(X)


# In[11]:


print("Explained variant ratio : ", fit.explained_variance_ratio_)


# In[14]:


print(fit.components_)


# In[15]:


import pandas as pd
from sklearn.decomposition import PCA

url = "pima_indians_diabetes.csv"
names = ['preg', 'plas','pres','skin','test','mass','pedi', 'age','class']
dataframe = pd.read_csv(url, names=names)
dataframe.shape
dataframe.head


# In[18]:


from sklearn.ensemble import ExtraTreesClassifier


# In[19]:


array = dataframe.values
X = array[:, 0:8]
y = array[:,8]


# In[20]:


model = ExtraTreesClassifier()


# In[21]:


model.fit(X,y)


# In[22]:


print(model.feature_importances_)

