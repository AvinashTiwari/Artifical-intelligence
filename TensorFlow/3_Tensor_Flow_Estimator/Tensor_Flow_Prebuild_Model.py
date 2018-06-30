
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[11]:


x_train = np.array([1.0,2.0,3.0,4.0])
y_train = np.array([-1.0,-2.0,-3.0,-4.0])
x_eval = np.array([5.0,10.0,15.0,20.0])
y_eval = np.array([-5.0,-10.0,-15.0,-20.0])
x_predict = np.array([50.0,100.0])


# In[12]:


feature_column = tf.feature_column.numeric_column(key='x',shape=[1])
feature_columns = [feature_column]


# In[4]:


estimator = tf.estimator.LinearRegressor(feature_columns = feature_columns)


# In[5]:


input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':x_train},
                                             y=y_train,
                                             batch_size=4,
                                             num_epochs=None,
                                             shuffle=True)


# In[6]:


train_input_function = tf.estimator.inputs.numpy_input_fn(x={'x':x_train},
                                             y=y_train,
                                             batch_size=4,
                                             num_epochs=1000,
                                             shuffle=False)

eval_input_function = tf.estimator.inputs.numpy_input_fn(x={'x':x_eval},
                                             y=y_eval,
                                             batch_size=4,
                                             num_epochs=1000,
                                             shuffle=False)


# In[7]:


estimator.train(input_fn= input_fn,
                steps=1000)


# In[9]:


print(estimator.evaluate(input_fn=train_input_function))
print(estimator.evaluate(input_fn=eval_input_function))


# In[13]:


predict_input_function = tf.estimator.inputs.numpy_input_fn(x={'x':x_predict},
                                             num_epochs=1,
                                             shuffle=False)


# In[16]:


print(list(estimator.predict(input_fn=predict_input_function)))

