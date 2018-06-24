
# coding: utf-8

# In[1]:


import platform
 
print(platform.python_version())


# In[2]:


import tensorflow as tf


# In[3]:


session = tf.Session()


# In[9]:


const_1 = tf.constant(value=[1.0])
const_2 = tf.constant(value=[2.0])
result = const_1 + const_2


# In[10]:


print(result)


# In[11]:


print(session.run(fetches=result))


# In[13]:


result_add = tf.add(x=const_1, y=const_2, name="results_add")
print(result_add)


# In[14]:


print(session.run(fetches=result_add))


# In[16]:


placeholder_1 = tf.placeholder(dtype=tf.float32)
result_placeholder = tf.add(x=placeholder_1, y=const_2, name="result_placeholder")
print(session.run(fetches=result_placeholder, feed_dict={placeholder_1:[2.0]}))


# In[18]:


#simple Linear
#y = Wx + b
W = tf.constant(value=[2.0])
b = tf.constant(value=[1.0])
x= tf.placeholder(dtype=tf.float32)
y = W*x + b


# In[19]:


#
print(session.run(fetches=y, feed_dict={x:[2.0]}))


# In[20]:


#using Multiply
mult = tf.multiply(x=W, y=x, name="mult")
y =  tf.add(x=mult, y=b)
print(session.run(fetches=y, feed_dict={x:[2.0]}))

