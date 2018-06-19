
# coding: utf-8

# In[3]:


from IPython.core.display import Image, display
display(Image(r'c:\users\avinash.t\iris_setosa.jpg'))
print('iris_setosa\n')
display(Image(r'c:\users\avinash.t\Iris_versicolor.jpg'))
print('iris_versicolor\n')
display(Image(r'c:\users\avinash.t\Iris_virginica.jpg'))
print('iris_virginica\n')


# In[6]:


from sklearn.datasets import load_iris
iris = load_iris()
iris.keys()


# In[7]:


iris.data.shape


# In[8]:


iris.feature_names


# In[9]:


iris.target_names


# In[10]:


iris.target


# In[16]:


##Logisitic Regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model = LogisticRegression()
model.fit(iris.data,iris.target)
#excepted out come
excpetedOutCome = iris.target
predictedValue = model.predict(iris.data)


# In[17]:


print(metrics.classification_report(excpetedOutCome, predictedValue))


# In[19]:


print(metrics.confusion_matrix(excpetedOutCome, predictedValue))

