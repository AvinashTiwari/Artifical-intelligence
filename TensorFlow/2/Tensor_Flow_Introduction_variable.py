
# coding: utf-8

# In[1]:


import platform
 
print(platform.python_version())


# In[2]:


import tensorflow as tf


# In[3]:


session = tf.Session()


# In[4]:


var_1 = tf.Variable(initial_value=[1.0],
                     trainable=True,
                   collections=None,
                   validate_shape=True,
                   caching_device=None,
                   name='var_1',
                   variable_def=None,
                   dtype=tf.float32,
                   expected_shape=(1, None),
                   import_scope=None
                   )


# In[5]:


print(var_1)
init = tf.global_variables_initializer()
session.run(init)
print(session.run(fetches=var_1))


# In[6]:


var_2 = var_1.assign(value=[2.0])
print(session.run(fetches=var_1))
print(session.run(fetches=var_2))

