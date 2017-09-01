import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import os


def preprocess(X):
    # X_centered = X = np.mean(X, axis=1)
    X_normalized = np.divide(X, 255)
    return X_normalized


def load_data(path, expect_labels=True):
    assert path.endswith('.csv')
    # If a previous call to this method has already converted
    # the data to numpy format, load the numpy directly
    X_path = path[:-4] + '.X.npy'
    Y_path = path[:-4] + '.Y.npy'
    if os.path.exists(X_path):
        X = np.load(X_path)
        if expect_labels:
            y = np.load(Y_path)
        else:
            y = None
        return preprocess(X), y

    # Convert the .csv file to numpy
    csv_file = open(path, 'r')

    reader = csv.reader(csv_file)

    # Discard header
    row = next(reader)

    y_list = []
    X_list = []

    for row in reader:
        if expect_labels:
            y_str, X_row_str = (row[0], row[1])
            y = int(y_str)
            assert 0 <= y <= 6
            y_list.append(np.eye(7)[y])
        else:
            X_row_str = row[1]
        X_row_strs = X_row_str.split(' ')
        X_row = [float(x) for x in X_row_strs]
        X_list.append(X_row)

    X = np.asarray(X_list).astype('float32')
    if expect_labels:
        y = np.asarray(y_list)
    else:
        y = None

    np.save(X_path, X)
    if y is not None:
        np.save(Y_path, y)

    return preprocess(X), y


def conv2d(x, filter_shape):
    """
        :x:     input layer  
        :filter_shape:  (w, h, input_sz, output_sz)
    """
    assert len(filter_shape) == 4
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.03))
    B = tf.Variable(tf.constant(0.1, shape=[filter_shape[-1]]))
    layer = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + B
    return layer


def relu(x):
    return tf.nn.relu(x)


def fully_connected(x, shape):
    assert len(shape) == 2
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.03))
    B = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
    X_flat = tf.reshape(x, [-1, shape[0]])
    layer = relu(tf.matmul(X_flat, W) + B)
    return layer


def output_layer(x, shape):
    assert len(shape) == 2
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.03))
    B = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
    layer = tf.matmul(x, W) + B
    return layer


def max_pool(x, pool_shape):
    assert len(pool_shape) == 2
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    return tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='SAME')

emotion_dict = { 0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
                4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

X_all, Y_all = load_data('../data/fer2013.csv')
assert len(X_all) == len(Y_all)

# save 20% for testing
test_start = int(.80 * len(X_all))
X_train, Y_train = X_all[:test_start, :], Y_all[:test_start, :]
X_test, Y_test = X_all[test_start:, :], Y_all[test_start:, :]

alpha = 0.0001
epochs = 10
batch_size = 50

x = tf.placeholder(dtype=tf.float32, shape=[None, 2304], name='Input')
x_shaped = tf.reshape(x, [-1, 48, 48, 1])
y = tf.placeholder(dtype=tf.float32, shape=[None, 7], name='Output')

conv_layer1 = relu(conv2d(x_shaped, [5, 5, 1, 32]))
max_pool_layer1 = max_pool(conv_layer1, [2, 2])
conv_layer2 = relu(conv2d(max_pool_layer1, [5, 5, 32, 64]))
max_pool_layer2 = max_pool(conv_layer2, [2, 2])
conv_layer3 = relu(conv2d(max_pool_layer2, [5, 5, 64, 128]))
max_pool_layer3 = max_pool(conv_layer3, [2, 2])
fc_layer4 = fully_connected(max_pool_layer3, [6 * 6 * 128, 1000])
y_predict = output_layer(fc_layer4, [1000, 7])

entropy_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y))
optimiser = tf.train.AdamOptimizer(learning_rate=alpha).minimize(entropy_cost)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_predict, 1))
accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter('./../data/graphs', sess.graph) 

    n_batches = int(len(X_train) / batch_size)

    for epoch in range(epochs):
        avg_cost = 0

        for i in range(n_batches):
            idx = i*batch_size
            batch_x = X_train[idx: idx + batch_size]
            batch_y = Y_train[idx: idx + batch_size]
            _, result = sess.run([optimiser, entropy_cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += result / n_batches
        
        test_accuracy = sess.run(accurary, feed_dict={x: X_test, y: Y_test})
        print('Epoch:', (epoch + 1), 'cost = ', '{:.3f}'.format(avg_cost),
        ' test accuracy: {:.3f}'.format(test_accuracy))

    print('\nTraining Complete')
    print('accurary:', sess.run(accurary, feed_dict={x: X_test, y: Y_test}))

writer.close()
