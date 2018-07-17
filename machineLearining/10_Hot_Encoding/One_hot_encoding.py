
# coding: utf-8

# In[1]:


from  numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[2]:


dataset = ['Pizza', 'Burger', 'Beacon','Beacon','Beacon','Burger','Pizza','Burger']


# In[3]:


values = array(dataset)
print(values)


# In[4]:


labelEncoder  = LabelEncoder()
interger_encoder = labelEncoder.fit_transform(values)
print(interger_encoder)


# In[5]:


oneHot = OneHotEncoder(sparse=False)
interger_encoded = interger_encoder.reshape(len(interger_encoder),1)
onehot_endcoded = oneHot.fit_transform(interger_encoded)
print(onehot_endcoded)


# In[6]:


inverted_result = labelEncoder.inverse_transform([argmax(onehot_endcoded[0,:])])
print(inverted_result)


# In[7]:


from keras.utils import to_categorical
dataset2 = ['Pizza', 'Burger', 'Beacon','Beacon','Beacon','Burger','Pizza','Burger']
myarray = array(dataset2)
print(myarray)


# In[8]:


interger_encoder = labelEncoder.fit_transform(myarray)
keras_encoded = to_categorical(interger_encoder)
print(keras_encoded)


# In[10]:


inverted_result2= argmax(keras_encoded[0])
print(inverted_result2)

