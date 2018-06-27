
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

url = "pima_indians_diabetes.csv"
names = ['preg', 'plas','pres','skin','test','mass','pedi', 'age','class']
dataframe = pd.read_csv(url, names=names)
dataframe.shape
dataframe.head


# In[4]:


array = dataframe.values
X = array[:, 0:8]
y = array[:,8]
scaler = MinMaxScaler(feature_range=(0,1))
rescaled = scaler.fit_transform(X)


# In[5]:


np.set_printoptions(precision=3)
print(rescaled[0:5,:])

