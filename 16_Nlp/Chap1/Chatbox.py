# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:28:54 2019

@author: avinash.tiwari
"""

import numpy as np
import tensorflow as tf
import re
import time

lines = open('movie_lines.txt', encoding= 'utf-8', errors='ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding= 'utf-8', errors='ignore').read().split('\n')

id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
        

conversation_ids =[]
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversation_ids.append(_conversation.split(','))
    
questions = []
answers = []

for conversation in   conversation_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
    