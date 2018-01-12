# coding: utf-8

# ## Neural Network Overview
#
# Define your own network here
#
# the task for this test is to create a network with 2 hidden layer, achieving at least 90% accuracy on test data
#
# READ ALL TODO TAG AND INSTRUCTION IN EACH BLOCK

# In[1]:


from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

import tensorflow as tf

# Set logging
tf.logging.set_verbosity(tf.logging.INFO)

# In[ ]:


# Parameters
learning_rate = 0.001
num_steps = 2000 # Small number of steps due to CPU usage only.
batch_size = 50
display_step = 10

# Network Parameters
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)




# In[ ]:


# Define the neural network
def neural_net(x_dict):
    img_size = 28
    num_features_1 = 32
    num_features_2 = 64

    x_image = tf.reshape(x_dict["images"], [-1, img_size, img_size, 1])

    # First convolutional layer
    W_conv1 = weight_variable([5, 5, 1, num_features_1])
    b_conv1 = bias_variable([num_features_1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer
    W_conv2 = weight_variable([5, 5, num_features_1, num_features_2])
    b_conv2 = bias_variable([num_features_2])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer
    W_fc1 = weight_variable([7 * 7 * num_features_2, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * num_features_2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, num_classes])
    b_fc2 = bias_variable([num_classes])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    return y_conv

# In[ ]:

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# In[ ]:


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    logits = neural_net(features)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Define the loss function using cross entropy.
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)

    # Define the loss function
    loss_op = tf.reduce_mean(
         tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits))

    # Define optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss=loss_op, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# In[ ]:


# Build the Estimator.
model = tf.estimator.Estimator(model_fn)

# In[ ]:

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=display_step)

# Train the Model
input_trn_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)

model.train(input_fn=input_trn_fn, steps=num_steps, hooks=[logging_hook])

# In[ ]:


# Evaluate the Model


# Define the input function for evaluating. TEMP SET NumEPOCHS!
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)

# Use the Estimator 'evaluate' method
eval_results = model.evaluate(input_fn)
print(eval_results)



