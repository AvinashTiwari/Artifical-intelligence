# Building the CNN-VAE model
 
# Importing the libraries
 
import numpy as np
import tensorflow as tf
 
# Building the CNN-VAE model within a class
 
class MDNRNN(object):
    
    # Initializing all the parameters and variables of the MDNRNN class
    def __init__(self,hps,reuse=False, gpu_mode=False):
        self.hps = hps
        with tf.variable_scope('mdn_rnn', reuse=reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'):
                    tf.logging.info('Model using cpu.')
                    self.g = tf.Graph()
                    with self.g.as_default():
                        self._build_model(hps)
                        
            else:
                tf.logging.info('Model using gpu.')
                self._build_graph()
        self._init_session()
    
    # Making a method that creates the MDNRNN model architecture itself
    def _build_model(self,hps):