
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

url = "pima_indians_diabetes.csv"
names = ['preg', 'plas','pres','skin','test','mass','pedi', 'age','class']


# In[2]:


dataframe = pd.read_csv(url , names=names)
array = dataframe.values
X = array[:,0:8]
y = array[:,8]

estimators = []
estimators.append(('standardixe',StandardScaler()))
estimators.append(('ida',LinearDiscriminantAnalysis()))
model =  Pipeline(estimators)
seed = 7


# In[3]:


kfold = KFold(n_splits=10, random_state=seed)
result = cross_val_score(model,X,y, cv=kfold)
print(result)
print(result.mean())


# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

url = "pima_indians_diabetes.csv"
names = ['preg', 'plas','pres','skin','test','mass','pedi', 'age','class']

dataframe = pd.read_csv(url , names=names)
array = dataframe.values
X = array[:,0:8]
y = array[:,8]



# In[8]:


feature =[]
feature.append(('pca', PCA(n_components=3)))
feature.append(('select_kbest', SelectKBest(k=6)))
feature_union = FeatureUnion(feature)
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', LogisticRegression()))
model = Pipeline(estimators)

seed =7
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(model,X, y,cv=kfold)
print(results)
print(results.mean())

