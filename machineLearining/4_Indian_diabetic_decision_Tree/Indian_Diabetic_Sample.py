
# coding: utf-8

# In[1]:


from IPython.core.display import Image, display
display(Image(r'c:\users\avinash.t\iris_setosa.jpg'))
print('iris_setosa\n')
display(Image(r'c:\users\avinash.t\Iris_versicolor.jpg'))
print('iris_versicolor\n')
display(Image(r'c:\users\avinash.t\Iris_virginica.jpg'))
print('iris_virginica\n')


# In[2]:


from sklearn.datasets import load_iris
iris = load_iris()
iris.keys()


# In[3]:


iris.data.shape


# In[4]:


iris.feature_names


# In[5]:


iris.target_names


# In[6]:


iris.target


# In[7]:


##Logisitic Regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model = LogisticRegression()
model.fit(iris.data,iris.target)
#excepted out come
excpetedOutCome = iris.target
predictedValue = model.predict(iris.data)


# In[8]:


print(metrics.classification_report(excpetedOutCome, predictedValue))


# In[9]:


print(metrics.confusion_matrix(excpetedOutCome, predictedValue))


# In[10]:


#navie Base Algoirthim
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(iris.data,iris.target)
excpeted = iris.target
predicted = model.predict(iris.data)
print(metrics.confusion_matrix(excpeted, predicted))
print(metrics.accuracy_score(excpeted,predicted))


# In[11]:


#Svm Single vector Machine
from sklearn.svm import SVC
model = SVC()
model.fit(iris.data,iris.target)
predicted = model.predict(iris.data)
print(metrics.confusion_matrix(excpeted, predicted))
print(metrics.accuracy_score(excpeted,predicted))


# In[12]:


url = "pima_indians_diabetes.csv"
names = ['preg', 'plas','pres','skin','test','mass','pedi', 'age','class']

import pandas as pd
df = pd.read_csv(url, names=names)
df.head()


# In[15]:


#K nearest Neighbour
array = df.values
X =array[:, 0:8]
y = array[:, 8]
test_size = 0.33
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
outcome = y_test
print(metrics.confusion_matrix(outcome, predicted))
print(metrics.accuracy_score(outcome,predicted))


# In[16]:


#Decision Tree
array = df.values
X =array[:, 0:8]
y = array[:, 8]
test_size = 0.33
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train , y_train)
predicted = model.predict(X_test)
outcome = y_test
print(metrics.confusion_matrix(outcome, predicted))
print(metrics.accuracy_score(outcome,predicted))

