
# coding: utf-8

# In[1]:


import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


# In[2]:


K.set_image_dim_ordering('th')
seed = 7
np.random.seed(seed)
(X_train,y_train), (X_test,y_test) = mnist.load_data()


# In[3]:


X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

X_train = X_train/255
X_test = X_test/255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
numclasses = y_test.shape[1]


# In[12]:


def mycnn():
    model = Sequential()
    model.add(Conv2D(32,(5,5), input_shape = (1,28,28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(numclasses, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    


# In[ ]:


model = mycnn()
model.fit(X_train,y_train, validation_data=(X_test, y_test), epochs=10,batch_size=200, verbose=2)
socers= model.evaluate(X_test, y_test,verbose=0)
print((100 - socers[1] * 100))

