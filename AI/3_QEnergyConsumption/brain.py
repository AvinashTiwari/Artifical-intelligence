# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:32:15 2019

@author: avinash.t
"""

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

class Brain(object):
    
    def __init__(self, learning_rate = 0.001, number_actions = 5):
        self.learning_rate = learning_rate
        states = Input(shape = (3,))
        x = Dense(units = 64, activation ='sigmoid')(states)
        y = Dense(units = 32, activation = 'sigmoid')(x)
        q_values = Dense(units = number_actions, activation = 'softmax')(y)
