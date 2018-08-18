
# coding: utf-8

# In[24]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# # Classifaction

# In[25]:


from sklearn.datasets import load_digits
digits = load_digits()
digits.keys()


# In[26]:


digits.images.shape


# In[27]:


digits.images[0]


# In[28]:


plt.matshow(digits.images[0],cmap=plt.cm.Greys)


# In[29]:


digits.data.shape


# In[30]:


digits.target.shape


# In[31]:


digits.target


# In[32]:


from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target)


# In[33]:


from  sklearn.svm import LinearSVC


# In[34]:


svm  = LinearSVC(C =0.1) 


# In[35]:


svm.fit(X_train, y_train)


# In[36]:


print(svm.predict(X_train))
print(y_train)


# In[37]:


svm.score(X_train,y_train)


# In[38]:


svm.score(X_test,y_test)


# # Random Forest Classifaction

# In[39]:


from sklearn.ensemble import RandomForestClassifier


# In[40]:


rf = RandomForestClassifier(n_estimators=50, random_state=1)


# In[41]:


rf.fit(X_train, y_train)


# In[42]:


rf.predict(X_test)


# In[43]:


rf.score(X_test, y_test)


# # LABEL CAN BE ANYTHING 

# In[44]:


number = np.array(['0','1','2','3','4','5','6','7','8','9'])


# In[45]:


y_train_string = number[y_train]


# In[ ]:


svm.fit(X_train,y_train_string)


# In[46]:


svm.predict(X_test)

