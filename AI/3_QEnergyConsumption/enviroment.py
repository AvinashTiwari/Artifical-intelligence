# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 18:27:45 2019

@author: avinash.t
"""

import numpy as np

class Enviroment(object):
    def __init__(self, optimal_temperature = (18.0, 24.0), initial_month=0, initial_number_users= 10 , initial_rate_data = 60):
        