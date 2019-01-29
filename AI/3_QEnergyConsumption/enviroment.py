# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 18:27:45 2019

@author: avinash.t
"""

import numpy as np

class Enviroment(object):
    def __init__(self, optimal_temperature = (18.0, 24.0), initial_month=0, initial_number_users= 10 , initial_rate_data = 60):
        self.monthly_atmospheric_temperatures = [1.0,5.0,7.0,10.0,11.0,20.0,23.0,24.0,22.0,10.0,5.0,1.0]
        self.initial_month = initial_month
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[initial_month]
        self.optimal_temperature = optimal_temperature
        self.min_temperature = 80
        self.min_number_users = 10
        self.max_number_users = 100
        self.max_update_users = 5
        self.min_rate_data = 20
        self.max_rate_data = 300
        self.max_update_data = 10
        self.initial_number_user = initial_number_users
        self.current_number_user = initial_number_users
        self.initial_rate_data = initial_rate_data
        self.current_rate_data = initial_rate_data
        self.instrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_user + 1.25 * self.current_rate_data
        self.temperature_ai = self.instrinsic_temperature
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1])/2.0
        self.total_enegry_ai = 0.0
        self.total_enegry_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1
    
    def update_env(self, direction, energy_ai ,month):
        energy_noai = 0
        
        if(self.temperature_noai < self.optimal_temperature[0]):
            energy_noai = self.optimal_temperature[0] - self.temperature_noai
            self.temperature_noai = self.optimal_temperature[0]
        elif(self.temperature_noai > self.optimal_temperature[1]):
            energy_noai = self.temperature_noai - self.optimal_temperature[1]
            self.temperature_noai = self.optimal_temperature[1]
        
        self.reward = energy_noai - energy_ai
        self.reward = 1e-3 * self.reward 
        
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[month]
        self.current_number_user += np.random.randint(-self.max_update_users,self.max_update_users)
        if(self.current_number_user > self.max_number_users):
            self.current_number_user  =  self.max_number_users
        elif(self.current_number_user < self.min_number_users):
            self.current_number_user  =  self.min_number_users
        
        self.current_rate_data += np.random.randint(-self.max_update_data,self.max_update_data)
        if(self.current_rate_data > self.max_rate_data):
            self.current_rate_data  =  self.max_rater_data
        elif(self.current_rater_data < self.min_rate_data):
            self.current_rate_data  =  self.min_rate_data
        
        past_intrinsic_temperature = self.instrinsic_temperature
        self.instrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_user + 1.25 * self.current_rate_data
        delta_instrinsic_temperature = self.instrinsic_temperature - past_intrinsic_temperature
        
        if(direction == -1):
            delta_temperature = -energy_ai
        elif(direction == 1):
            delta_temperature = energy_ai
            
            
        self.temperature_ai += delta_instrinsic_temperature + delta_temperature
        
        self.temperature_noai += delta_instrinsic_temperature
            
            
        
        