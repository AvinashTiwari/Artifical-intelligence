#Stage 1: Import all project dependencies

import os
import requests
import numpy as np
import tensorflow as tf

from scipy.misc import imread, imsave
from tensorflow.keras.datasets import fashion_mnist
from flask import Flask, request, jsonify

print(tf.__version__)

#Stage 2: Load the pretrained model
with open('fashion_model_flask.json', 'r') as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
# load weights into new model
model.load_weights("fashion_model_flask.h5")

#Stage 3: Creating the Flask API
#Starting the Flask application
app = Flask(__name__)

#Defining the classify_image function
@app.route("/api/v1/<string:img_name>", methods=["POST"])
def classify_image(img_name):
    #Define the uploads folder
    upload_dir = "uploads/"
    #Load an uploaded image
    image = imread(upload_dir + img_name)
    
    #Define the list of class names 
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    #Perform predictions with pre-trained model
    prediction = model.predict([image.reshape(1, 28*28)])

    #Return the prediction to the user
    return jsonify({"object_identified":classes[np.argmax(prediction[0])]})

#Start the Flask application
app.run(port=5000, debug=False)