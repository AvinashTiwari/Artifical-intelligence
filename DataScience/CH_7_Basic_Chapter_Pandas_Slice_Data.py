
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


titanic = pd.read_csv("./data/titanicexcel.csv", sep=';')
titanic.head(10)


# In[3]:


titanic.shape


# In[4]:


titanic_female = titanic[titanic['Sex'] == 'female']
titanic_female.head()
                         


# In[5]:


filter1  = titanic['Sex'] == 'female'
filter2 = titanic['Embarked'] == 'C'
titanic_female_C = titanic[filter1 &  filter2]
titanic_female_C.head()


# In[6]:


titanic['Pclass'].value_counts()


# In[8]:


filter3 = titanic['Sex']=='male'
filter4 = titanic['Pclass'] < 3
titanic_male_Pclass=  titanic[filter3 & filter4]
titanic_male_Pclass.head()


# In[9]:


filter3 = titanic['Sex']=='male'
filter4 = titanic['Pclass'] < 3
titanic_male_Pclass=  titanic[filter3 | filter4]
titanic_male_Pclass.head()

