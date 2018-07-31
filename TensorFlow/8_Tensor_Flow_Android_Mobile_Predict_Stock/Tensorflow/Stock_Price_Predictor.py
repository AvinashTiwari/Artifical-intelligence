import tensorflow as tf
from Data_Handler import build_data_subsets


# Function to measure accuracy by comparing model output to the correct answers
# If the max value for the actual and expected values is in the same position then answer is correct, otherwise not
def measure_accuracy(actual, expected):
    num_correct = 0
    for i in range(len(actual)):
        actual_value = actual[i]
        expected_value = expected[i]
        if actual_value[0] >= actual_value[1] and expected_value[0] >= expected_value[1]:
            num_correct += 1
        elif actual_value[0] <= actual_value[1] and expected_value[0] <= expected_value[1]:
            num_correct += 1
    return (num_correct / len(actual)) * 100


x_train, y_train = build_data_subsets('AAPL', '20180101', '20180205')
x_test, y_test = build_data_subsets('AAPL', '20180205', '20180215')


# y = Wx + b
# Input into the model, any number of 5 element (factor) arrays
x_input = tf.placeholder(dtype=tf.float32, shape=[None, 5], name='x_input')
# Input into the model for training purposes only, use to show model what correct answer is
y_input = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='y_input')

# Weights variable to be changed during training
W = tf.Variable(initial_value=tf.ones(shape=[5, 2]))
# Biases variable to be changed during training
b = tf.Variable(initial_value=tf.ones(shape=[2]))

# Output from the model, multiplies weight and input and adds bias
y_output = tf.add(tf.matmul(x_input, W), b, name='y_output')

# Loss function measures the difference between actual and expected outputs
loss = tf.reduce_sum(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y_output)))
# Optimizer aims at minimizing loss by changing variable node values
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

# Saver variable used to save graph def
saver = tf.train.Saver()

# Session variable used to evaluate nodes
session = tf.Session()
session.run(tf.global_variables_initializer())

# Write the graph definition to a pbtxt file (node names, shapes, etc.)
tf.train.write_graph(session.graph_def, '.', 'stock_prediction.pbtxt', False)

# Train the model by running through training data 20000 times
for _ in range(20000):
    session.run(optimizer, feed_dict={x_input: x_train, y_input: y_train})

# Save the model with all of the trained values at the nodes into a ckpt file
saver.save(session, save_path='stock_prediction.ckpt')

# Measure the accuracy of training and testing sets
print(measure_accuracy(session.run(y_output, feed_dict={x_input: x_train}), y_train))
print(measure_accuracy(session.run(y_output, feed_dict={x_input: x_test}), y_test))
