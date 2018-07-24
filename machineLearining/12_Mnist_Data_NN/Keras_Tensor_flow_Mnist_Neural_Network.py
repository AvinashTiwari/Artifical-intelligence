
# coding: utf-8

# In[1]:


from keras.datasets import mnist
import matplotlib.pyplot as plt


# In[2]:


(X_train, y_train), (X_test, y_text ) = mnist.load_data()
plt.subplot(221)
plt.imshow(X_train[0],cmap=plt.get_cmap('gray'))

plt.subplot(222)
plt.imshow(X_train[1],cmap=plt.get_cmap('gray'))


plt.subplot(223)
plt.imshow(X_train[2],cmap=plt.get_cmap('gray'))

plt.subplot(224)
plt.imshow(X_train[3],cmap=plt.get_cmap('gray'))



plt.show()


# In[3]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
n_nodes_h1 = 500
n_nodes_h2 = 500
n_nodes_h3 = 500

n_classes = 10
batch_size = 100


# In[17]:


x = tf.placeholder('float',[None,784])
y = tf.placeholder('float',[None,n_classes])

def neuralnetworkmodel(data):
    hidden_1_layer = {'weight': tf.Variable(tf.random_normal([784,n_nodes_h1])), 
                      'biases' : tf.Variable(tf.random_normal([n_nodes_h1]))}
    hidden_2_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_h1,n_nodes_h2])), 
                      'biases' : tf.Variable(tf.random_normal([n_nodes_h2]))}
    hidden_3_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_h2,n_nodes_h3])), 
                      'biases' : tf.Variable(tf.random_normal([n_nodes_h3]))}
    
    output_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_h3,n_classes])), 
                     'biases' : tf.Variable(tf.random_normal([n_classes]))}
    
    l1 =tf.add(tf.matmul(data,hidden_1_layer['weight']),hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 =tf.add(tf.matmul(l1,hidden_2_layer['weight']),hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 =tf.add(tf.matmul(l2,hidden_3_layer['weight']),hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.add(tf.matmul(l3,output_layer['weight'] ) ,output_layer['biases'])
    return output


# In[20]:


def trainneuralnetwork(x):
    prediction = neuralnetworkmodel(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    epocs  =3
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for ep in range(epocs):
            epoch_loss =0
            
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoc_x, epoc_y = mnist.train.next_batch(batch_size)
                _,c =  sess.run([optimizer, cost], feed_dict={x:epoc_x, y:epoc_y })
                epoch_loss += c
                
                print('Ecpoc ', ep , ' Competed of epochs ' , epocs , ' losses ' , epoch_loss  )
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy : ', accuracy.eval({x:mnist.test.images,y:mnist.test.labels }))
trainneuralnetwork(x)


# In[1]:


import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers  import Dense
from keras.layers import Dropout
from keras.utils  import np_utils


# In[2]:


seed = 7
np.random.seed(seed)
(X_train,y_train),(X_test,y_test) = mnist.load_data()
pixels =  X_train.shape[1]*X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0],pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0],pixels).astype('float32')

X_train = X_train/255
X_test = X_test/255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# In[3]:


def nerualNetworkmodel():
    model = Sequential()
    model.add(Dense(pixels,input_dim = pixels, kernel_initializer='normal',activation='relu'))
    model.add(Dense(num_classes,kernel_initializer='normal',activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model


# In[7]:


model = nerualNetworkmodel()
model.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=3, batch_size=200, verbose=2)
scores = model.evaluate(X_test, y_test,  verbose=0)
print(100-scores[1]*100)

