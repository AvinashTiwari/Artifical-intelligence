
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation , Dense
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils  import np_utils
np.random.seed(10)


# In[7]:


TimeSteps = 28
inputrsize = 28
batch_size =  50
batch_index = 0
output  = 10
cell_size = 50
lr = 0.001


# In[8]:


(X_train,y_train),(X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1,28,28)/255.
X_test = X_test.reshape(-1,28,28)/255.
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)


# In[9]:


model = Sequential()
model.add(SimpleRNN(batch_input_shape=(None, TimeSteps, inputrsize), output_dim=output, unroll=True))
model.add(Dense(output))
model.add(Activation('softmax'))


# In[10]:


adam = Adam(lr)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])


# In[13]:


for trainingstep in range(5001):
    X_batch = X_train[batch_index:batch_index+batch_size, :, :]
    y_btach = y_train[batch_index:batch_index+batch_size, :]
    cost = model.train_on_batch(X_batch,y_btach)
    batch_index += batch_size
    batch_index = 0 if batch_index >= X_train.shape[0] else batch_index
    if trainingstep % 500 == 0 :
        cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print('Test Coset', cost)
        print('Test accuracy', accuracy)

