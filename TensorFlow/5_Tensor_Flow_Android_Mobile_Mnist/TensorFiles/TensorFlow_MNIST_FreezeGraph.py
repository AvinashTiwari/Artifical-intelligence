
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference, optimize_for_inference_lib


# In[2]:


freeze_graph.freeze_graph(input_graph='.\\Mnist_Graph\\mnist_model.pbtxt',
                          input_saver='',
                          input_binary=True,
                         input_checkpoint='.\\Mnist_Graph\\mnist_model.ckpt',
                          output_node_names='y_actual',
                         restore_op_name='save/restore_all',
                         filename_tensor_name='save/Const:0',
                         output_graph='.\\Mnist_Graph\\mnist_model.pb',
                         clear_devices=True,
                         initializer_nodes= '',
                         variable_names_blacklist= '',
                         )


# In[3]:


input_graph_def = tf.GraphDef()
with tf.gfile.Open('.\\Mnist_Graph\\mnist_model.pb', 'rb') as f:
    data = f.read()
    input_graph_def.ParseFromString(data)
    

output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def=input_graph_def,
                                                                    input_node_names=['x_input'],
                                                                    output_node_names=['y_actual'],
                                                                     placeholder_type_enum=tf.float32.as_datatype_enum                       
                                                                    )

f = tf.gfile.FastGFile(name='.\\Mnist_Graph\\optimized_frozen_mnist_model.pb',
                      mode='w')
f.write(file_content=output_graph_def.SerializeToString())

