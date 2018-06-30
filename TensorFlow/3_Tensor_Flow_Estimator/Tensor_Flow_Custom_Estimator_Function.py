
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


def model_fn(features, labels, mode):
    if labels == 'infer':
        labels = np.array([0,0])
    
    W = tf.get_variable(name='W', shape=[1], dtype=tf.float64)
    b = tf.get_variable(name='b', shape=[1], dtype=tf.float64)
    y = W * features['x'] + b
    
    loss = tf.reduce_sum(input_tensor=tf.square(x=(y-labels)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    global_step = tf.train.get_global_step()
    train_step = tf.group(optimizer.minimize(loss=loss), tf.assign_add(global_step, 1))
    
    return tf.estimator.EstimatorSpec( mode = mode,
                                      predictions = y,
                                      loss =loss,
                                      train_op = train_step
    )


# In[3]:


x_train = np.array([1.0,2.0,3.0,4.0])
y_train = np.array([-1.0,-2.0,-3.0,-4.0])
x_eval = np.array([5.0,10.0,15.0,20.0])
y_eval = np.array([-5.0,-10.0,-15.0,-20.0])
x_predict = np.array([50.0,100.0])


# In[4]:


feature_column = tf.feature_column.numeric_column(key='x',shape=[1])
feature_columns = [feature_column]


# In[5]:


estimator = tf.estimator.Estimator(model_fn = model_fn)


# In[6]:


input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':x_train},
                                             y=y_train,
                                             batch_size=4,
                                             num_epochs=None,
                                             shuffle=True)


# In[7]:


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


# In[8]:


estimator.train(input_fn= input_fn,
                steps=1000)


# In[9]:


print(estimator.evaluate(input_fn=train_input_function))
print(estimator.evaluate(input_fn=eval_input_function))


# In[10]:


predict_input_function = tf.estimator.inputs.numpy_input_fn(x={'x':x_predict},
                                             num_epochs=1,
                                             shuffle=False)


# In[11]:


print(list(estimator.predict(input_fn=predict_input_function)))

