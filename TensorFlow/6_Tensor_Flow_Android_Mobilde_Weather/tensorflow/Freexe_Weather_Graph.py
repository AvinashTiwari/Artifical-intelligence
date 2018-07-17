
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib


# In[2]:


freeze_graph.freeze_graph(input_graph='./WeatherGraph/weather_prediction.pbtxt',
                          input_saver='',
                          input_binary=True,
                          input_checkpoint='./WeatherGraph/weather_prediction.ckpt',
                          output_node_names='y_output',
                          restore_op_name='save/restore_all',
                          filename_tensor_name='save/Const:0',
                          output_graph='./WeatherGraph/frozen_weather_prediction.pb',
                          clear_devices=True,
                          initializer_nodes='')

input_graph_def = tf.GraphDef()
with tf.gfile.Open('./WeatherGraph/frozen_weather_prediction.pb', 'rb') as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def,
                                                                     ['x_input'],
                                                                     ['y_output'],
                                                                     tf.float32.as_datatype_enum)
f = tf.gfile.FastGFile(name='./WeatherGraph/optimized_weather_prediction.pb', mode='w')
f.write(file_content=output_graph_def.SerializeToString())

