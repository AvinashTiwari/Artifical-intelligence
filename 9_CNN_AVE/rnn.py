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
        self.num_mixture = hps.num_mixture
        KMIX = self.num_mixture
        INWIDTH = hps.input_seq_width
        OUTWIDTH = hps.output_seq_width
        LENGTH = hps.max_seq_length
        if hps.is_training:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
        cell_fn = tf.contrib.rnn.LayerNormalBasicLSTMCell
        user_recurrent_dropout = False if self.hps.use_recurrent_dropout == 0 else True
        user_input_dropout = False if self.hps.use_input_dropout == 0 else True
        user_output_dropout = False if self.hps.use_output_dropout == 0 else True
        use_layer_norm = False if self.hps.use_layer_norm == 0 else True
        if(user_recurrent_dropout):
            cell = cell_fn(hps.rnn_size, layer_norm=use_layer_norm, dropout_keep_prob=self.hps.recurrent_dropout_prob)
        else:
            cell = cell_fn(hps.rnn_size, layer_norm=use_layer_norm)
            
        if(user_input_dropout):
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=self.hps.input_dropout_prob)
        if(user_output_dropout):
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=self.hps.output_dropout_prob)
        self.cell = cell
            
            