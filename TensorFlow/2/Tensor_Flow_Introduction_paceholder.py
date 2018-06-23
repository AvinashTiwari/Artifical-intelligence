
# coding: utf-8

# In[1]:


import platform
 
print(platform.python_version())


# In[2]:


import tensorflow as tf


# In[3]:


session = tf.Session()


# In[4]:


placefolder_1 = tf.placeholder(dtype=tf.float32,
                                shape=(1,4),
                                name='placefolder_1')


# In[5]:


print(placefolder_1)


# In[7]:


print(session.run(fetches=placefolder_1,feed_dict={placefolder_1:[[1.0,2.0,3.0,4.0]]}))


# In[8]:


placefolder_2 = tf.placeholder(dtype=tf.float32,
                                shape=(2,2),
                                name='placefolder_2')


# In[11]:


print(session.run(fetches=[placefolder_1, placefolder_2],feed_dict={placefolder_1:[[1.0,2.0,3.0,4.0]],
                                                                    placefolder_2:[[1.0,2.0],[3.0,4.0]]  
                                                                    }))

