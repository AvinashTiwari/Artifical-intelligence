
# coding: utf-8

# In[3]:


from  numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[4]:


dataset = ['Pizza', 'Burger', 'Beacon','Beacon','Beacon','Burger','Pizza','Burger']


# In[5]:


values = array(dataset)
print(values)


# In[6]:


labelEncoder  = LabelEncoder()
interger_encoder = labelEncoder.fit_transform(values)
print(interger_encoder)


# In[7]:


oneHot = OneHotEncoder(sparse=False)
interger_encoded = interger_encoder.reshape(len(interger_encoder),1)
onehot_endcoded = oneHot.fit_transform(interger_encoded)
print(onehot_endcoded)


# In[10]:


inverted_result = labelEncoder.inverse_transform([argmax(onehot_endcoded[0,:])])
print(inverted_result)

