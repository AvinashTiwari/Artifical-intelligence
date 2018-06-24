
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


# In[2]:


array = dataframe.values
X = array[:,0:8]
y = array[:,8]
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X,y)
import numpy as np
np.set_printoptions(precision=3)
print(fit.scores_)


# In[3]:


features = fit.transform(X)
print(features[0:5,:])


# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
logreg = LogisticRegression()
X_train, X_test,y_train, y_test = train_test_split(X,y)
logreg.fit(X_train,y_train)
logreg.score(X_test, y_test)

