
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[2]:


mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
train_image = mnist_data.train.images[0]
train_label = mnist_data.train.labels[0]
#print(train_image)
#print(train_label)
#Y = Wx +b


# In[3]:


x_input = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x_input')
W = tf.Variable(initial_value=tf.zeros(shape=[784,10]), name='W')
b = tf.Variable(initial_value=tf.zeros(shape=[10]), name='b')
y_actual = tf.add(x=tf.matmul(a=x_input,b=W, name='matmul'), y=b, name="y_actual")


# In[4]:


y_expected = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y_expected')
cross_entropy_loss = tf.reduce_mean(
    input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=y_expected, logits=y_actual),
                                    name="cross_entropy")


# In[5]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5, name="optimizer")
train_step = optimizer.minimize(loss=cross_entropy_loss, name="train_step")


# In[11]:


saver = tf.train.Saver()

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

tf.train.write_graph(graph_or_graph_def=session.graph_def,
                    logdir='.\\Mnist_Graph',
                     name='mnist_model.pbtxt',
                    as_text=False)


# In[13]:


for _ in range(1000):
    batch = mnist_data.train.next_batch(100)
    train_step.run(feed_dict={x_input:batch[0], y_expected :batch[1]})

saver.save(sess=session,
           save_path='.\\Mnist_Graph\\mnist_model.ckpt')

Correct_predection = tf.equal(x=tf.argmax(y_actual, 1), y=tf.argmax(y_expected,1))
accuracy = tf.reduce_mean(tf.cast(x=Correct_predection,dtype=tf.float32))
print(accuracy.eval(feed_dict={x_input:mnist_data.test.images, y_expected :mnist_data.test.labels}))


# In[8]:


print(session.run(fetches=y_actual, feed_dict={x_input:[mnist_data.test.images[0]]}))

