
# coding: utf-8

# In[12]:


import platform
 
print(platform.python_version())


# In[6]:


import tensorflow as tf

const_1 = tf.constant(value=[[1.0,2.0]], 
                      dtype=tf.float32,
                     shape=(1,2),
                      name='const_1',
                     verify_shape=True)


# In[7]:


print(const_1)


# In[8]:


session = tf.Session()
session.run(fetches=const_1)


# In[9]:


const_2 = tf.constant(value=[[3.0,4.0]], 
                      dtype=tf.float32,
                     shape=(1,2),
                      name='const_2',
                     verify_shape=True)


# In[10]:


#session = tf.Session()
session.run(fetches=[const_1,const_2])

