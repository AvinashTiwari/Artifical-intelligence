
# coding: utf-8

# In[2]:


import platform
 
print(platform.python_version())


# In[3]:


import tensorflow as tf


# In[4]:


session = tf.Session()


# In[18]:


x_train = [1.0,2.0,3.0,4.0]
y_train = [2.0,3.0,4.0,5.0]
y_actual = [1.5,2.5,3.5,4.5]
loss =tf.Variable(tf.reduce_sum(input_tensor=tf.square(x=tf.subtract(x=y_train, y=y_actual))))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
print(loss)
train_step = optimizer.minimize(loss=loss)
print(train_step.values)

