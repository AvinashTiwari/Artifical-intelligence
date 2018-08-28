
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf


# In[2]:


titanic = pd.read_csv("./data/titanic.csv")
titanic.head(10)


# In[3]:


titanic.drop(labels=['PassengerId','Name', 'Ticket','Cabin', 'Embarked'], axis=1, inplace=True)
titanic.head()


# In[4]:


convertnumeric= {'male':1 , 'female':0}
titanic.Sex = titanic.Sex.map(convertnumeric)
titanic.head()


# In[5]:


titanic.fillna(value=titanic.Age.mean(), inplace=True)
titanic.isnull().sum()
titanic.head()


# In[6]:


survived = titanic['Survived']
survived.head()


# In[7]:


survived.shape


# In[8]:


titanic.drop(['Survived'],axis=1, inplace=True)
titanic.head()


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


array = titanic.values
X =array[:, 0:6]
X


# In[27]:


arraySurvided = survived.values
y = arraySurvided
y


# In[28]:


test_size = 0.33
#X_train=titanic[:800]
#y_train=survived[:800]
#X_test =titanic[800:]
#y_test = survived[800:]
#X=titanic[:800]
#y=survived[:800]


#X =array[:, 0:6]
#y = array[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size)

y_train = y_train.reshape((y_train.shape[0], 1))

y_train


# In[29]:


input_shape = 6


x_input = tf.placeholder(dtype=tf.float32, shape=[None, input_shape], name='x_input')

y_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_input')
W = tf.Variable(initial_value=tf.ones(shape=[input_shape, 1]), name='W')
b = tf.Variable(initial_value=tf.ones(shape=[1]), name='b')
y_output = tf.add(tf.matmul(x_input, W), b, name='y_output')


# In[30]:


loss = tf.reduce_sum(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y_output)))
optimizer = tf.train.AdamOptimizer(0.005).minimize(loss)
saver = tf.train.Saver()


# In[31]:


session = tf.Session()
session.run(tf.global_variables_initializer())

tf.train.write_graph(session.graph_def, '.', './Titanic/Titanic_prediction.pbtxt', False)


# In[32]:


for _ in range(800):
    session.run(optimizer, feed_dict={x_input: X_train, y_input: y_train})

saver.save(session, './Titanic/Titanic_prediction.ckpt')


# In[33]:


from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
freeze_graph.freeze_graph(input_graph='./Titanic/Titanic_prediction.pbtxt',
                          input_saver='',
                          input_binary=True,
                          input_checkpoint='./Titanic/Titanic_prediction.ckpt',
                          output_node_names='y_output',
                          restore_op_name='save/restore_all',
                          filename_tensor_name='save/Const:0',
                          output_graph='./Titanic/frozen_Titanic_prediction.pb',
                          clear_devices=True,
                          initializer_nodes='')

input_graph_def = tf.GraphDef()
with tf.gfile.Open('./Titanic/frozen_Titanic_prediction.pb', 'rb') as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def,
                                                                     ['x_input'],
                                                                     ['y_output'],
                                                                     tf.float32.as_datatype_enum)
f = tf.gfile.FastGFile(name='./Titanic/optimized_Titanic_prediction.pb', mode='w')
f.write(file_content=output_graph_def.SerializeToString())

