# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:38:57 2019

@author: avinash.t
"""

import numpy as np

class DQN(object):
    
    def __init__(self, max_memory = 100, discount = 0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount = discount
    
    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]
            
        