
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation , Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
np.random.seed(10)


# In[3]:


topword = 5000
(X_train,y_train),(X_test, y_test) = imdb.load_data(num_words=topword)
max_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)


# In[4]:


embeddingVector = 32
model = Sequential()
model.add(Embedding(topword, embeddingVector, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[6]:


print(model.summary())
model.fit(X_train,y_train, validation_data=(X_test,y_test),epochs=1, batch_size=64)
score = model.evaluate(X_test, y_test, verbose=0)
print(socre[1]*100)

