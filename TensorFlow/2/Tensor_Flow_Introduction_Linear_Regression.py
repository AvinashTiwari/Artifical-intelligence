
# coding: utf-8

# In[1]:


import platform
 
print(platform.python_version())


# In[2]:


import tensorflow as tf


# In[3]:


session = tf.Session()
x_train = [1.0,2.0,3.0,4.0]
y_train= [-1.0,-2.0,-3.0,-4.0]


# In[4]:


# y Wx +b
W = tf.Variable(initial_value=[1.0], dtype=tf.float32)


# In[5]:


b = tf.Variable(initial_value=[1.0], dtype=tf.float32)


# In[6]:


x = tf.placeholder(dtype=tf.float32)
y_input = tf.placeholder(dtype=tf.float32)


# In[7]:


y_output = W * x +b


# In[8]:


loss = tf.reduce_sum(input_tensor=tf.square(x=y_output - y_input))


# In[9]:


optmizer  = tf.train.GradientDescentOptimizer(learning_rate=0.01)


# In[10]:


train_step = optmizer.minimize(loss=loss)


# In[11]:


session = tf.Session()
session.run(tf.global_variables_initializer())


# In[12]:


#Check before we start
session.run(fetches=loss, feed_dict={x: x_train, y_input: y_train})


# In[13]:


for _ in range(1000):
    session.run(fetches=train_step, feed_dict={x: x_train, y_input: y_train})


# In[16]:



print(session.run(fetches=[loss, W, b], feed_dict={x: x_train, y_input: y_train}))

print(session.run(fetches=y_output, feed_dict={x: [5.0, 10.0, 15.0]}))

