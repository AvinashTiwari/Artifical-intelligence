import os
import requests
import numpy as np
import tensorflow as tf

from scipy.misc import imread, imsave
from tensorflow.keras.datasets import fashion_mnist
from flask import Flask, request, jsonify

(X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()

for i in range(5):
    imsave(name="uploads/{}.png".format(i),arr=X_test[i])