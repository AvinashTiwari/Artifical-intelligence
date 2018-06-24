
# coding: utf-8

# In[1]:


import pandas as pd
url = "pima_indians_diabetes.csv"
names = ['preg', 'plas','pres','skin','test','mass','pedi', 'age','class']
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
dataframe = pd.read_csv(url, names=names)
dataframe.shape
dataframe.head


# In[5]:


array = dataframe.values
array


# In[15]:


X = array[:,0:8]
y= array[:,0]
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
from sklearn.feature_selection  import RFE
rfe = RFE(model, 3)
fit = rfe.fit(X,y)
print("Number of features" ,fit.n_features_)
print("Selected Features" ,fit.support_)
print("Ranking " ,fit.ranking_)

