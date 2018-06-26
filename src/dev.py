import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 1E-3
batch_size = 100
num_steps = 10000

def load_data():
    """ Reads the data.csv file to load the images and labels."""

    raw_data = pd.read_csv('data.csv')
    data = raw_data.sample(frac=1)
    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]

    images = X.values.reshape(X.shape[0], 32,32)
    labels = pd.factorize(Y)[0]
    dev = {"train_images" : images[:90000],
           "train_labels" : labels[:90000],
           "test_images"  : images[:-2000],
           "test_labels"  : labels[:-2000]}
    return dev

def cnn(x_dict):
    """ Generates the CNN model."""

    with tf.variable_scope("Input_Layer"):
        x = x_dict['images']
        x = tf.reshape(x, (-1,32,32,1))
        x = tf.cast(x, dtype=tf.float32)
    with tf.variable_scope("First_Layer"):
        conv1 = tf.layers.conv2d(inputs=x, filters=16,
            kernel_size=[5,5], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
    with tf.variable_scope("Second_Layer"):
        conv2 = tf.layers.conv2d(inputs=pool1, filters=32,
            kernel_size=[5,5], padding="valid", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
    with tf.variable_scope("Third_Layer"):
        conv3 = tf.layers.conv2d(inputs=pool2, filters=64,
            kernel_size=[5,5], padding="same", activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=1)
    with tf.variable_scope("Fourth_Layer"):
        conv4 = tf.layers.conv2d(inputs=pool3, filters=128,
            kernel_size=[5,5], padding="same", activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2,2], strides=1)
    with tf.variable_scope("Fully_Connected_Layers"):
        pool4_flat = tf.reshape(pool4, (-1, 4*4*128))
        fc1 = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
        fc2 = tf.layers.dense(inputs=fc1, units=512, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=fc2, rate=0.4)
    output_layer = tf.layers.dense(inputs=dropout, units=56)
    return output_layer

def model_fn(features, labels, mode):
    """ Defines actions for training, evaluation and prediction."""

    logits = cnn(features)

    pred_classes = tf.argmax(logits, axis=1)
    pred_probs = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy':acc_op})
    return estim_specs 

# Display an image.
#plt.imshow(dev['train_images'][4])
#plt.show()

tf.logging.set_verbosity(tf.logging.INFO)
model = tf.estimator.Estimator(model_fn, model_dir='./model/c')
dev = load_data()
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': dev['train_images']}, y = dev['train_labels'],
    batch_size=batch_size, num_epochs=10, shuffle=True)

t = model.train(input_fn, steps=num_steps)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': dev['test_images']}, y = dev['test_labels'],
    batch_size=batch_size, shuffle=False)

e = model.evaluate(input_fn, steps=num_steps)

print("Testing Accuracy: ", e['accuracy'])
