
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from load_data import load_data
from sklearn.model_selection import train_test_split

emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
                4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

X_all, Y_all = load_data('../data/fer2013.csv')
assert len(X_all) == len(Y_all)# save 20% for testing

# save 20% for testing
test_start = int(.80 * len(X_all))
X_train, X_test, Y_train, Y_test  = train_test_split(X_all, Y_all, test_size=.20, random_state=20)

alpha = 0.0001
epochs = 10
batch_size = 256

tf.reset_default_graph()

#INPUT
input = tf.placeholder(dtype=tf.float32, shape=[None, 2304], name='Input')
input_shaped = tf.reshape(input, [-1, 48, 48, 1])
y = tf.placeholder(dtype=tf.float32, shape=[None, 7], name='Output')

#  ARCHITECTURE
layer1 = tf.layers.conv2d(input_shaped, filters=64, kernel_size=[5, 5],  padding='same', activation=tf.nn.relu, name='l1_conv2d')
layer1 = tf.layers.max_pooling2d(layer1, pool_size=[2, 2], strides=2, name='l1_maxpool')
layer1 = tf.layers.dropout(layer1, rate=0.5, name='l1_dropout')

layer2 = tf.layers.conv2d(layer1, filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='l2_conv2d')
layer2 = tf.layers.max_pooling2d(layer2, pool_size=[2, 2], strides=2, name='l2_maxpool')
layer2 = tf.layers.dropout(layer2, rate=0.5, name='l2_dropout')

layer3 = tf.layers.conv2d(layer2, filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='l3_conv2d')
layer3 = tf.layers.max_pooling2d(layer3, pool_size=[2, 2], strides=2, name='l3_maxpool')
layer3 = tf.layers.dropout(layer1, rate=0.5, name='l3_dropout')

layer4 = tf.layers.conv2d(layer3, filters=128, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='l4_conv2d')
layer4 = tf.layers.max_pooling2d(layer4, pool_size=[2, 2], strides=2, name='l4_maxpool')
layer4 = tf.layers.dropout(layer4, rate=0.5, name='l4_dropout')

layer5 = tf.layers.conv2d(layer4, filters=128, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='l5_conv2d')
layer5 = tf.layers.max_pooling2d(layer5, pool_size=[2, 2], strides=2, name='l5_maxpool')
layer5 = tf.layers.dropout(layer5, rate=0.5, name='l5_dropout')


flattened = tf.reshape(layer5, [-1, 6 * 6 * 128], name='flattened')
dense1024 = tf.layers.dense(flattened, units=1024, activation=tf.nn.relu, name='l6_dense_1024')
dropout = tf.layers.dropout(dense1024, rate=0.5, name='l6_dropout')
dense512 = tf.layers.dense(dropout, units=512, activation=tf.nn.relu, name='l7_dense_512')
dropout2 = tf.layers.dropout(dense512, rate=0.5, name='l7_dropout2')
logits = tf.layers.dense(dropout2, units=7, name='l8_dense_7')

probs = tf.nn.softmax(logits, name='y_softmax')
y_predict = tf.argmax(probs, axis=1, name='y_predict')
y_true = tf.argmax(y, axis=1, name='y_true')

entropy_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimiser = tf.train.AdamOptimizer(learning_rate=alpha).minimize(entropy_cost)
correct_prediction = tf.equal(y_true, y_predict)
accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
logging_steps = 1
file = open('../training_steps.txt', 'w')
file.write('Epoch, cost, accuracy') # Header


with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter('./../graphs/deep', sess.graph)

    n_batches = int(len(X_train) / batch_size)

    for epoch in range(epochs):
        avg_cost = 0

        for i in range(n_batches):
            idx = i * batch_size
            batch_x = X_train[idx: idx + batch_size]
            batch_y = Y_train[idx: idx + batch_size]
            _, result = sess.run([optimiser, entropy_cost], feed_dict={input: batch_x, y: batch_y})
            avg_cost += result / n_batches

        test_accuracy = sess.run(accurary, feed_dict={input: X_test, y: Y_test})
        if epoch % logging_steps == 0:
            print('\nEpoch:' + str(epoch + 1) + ' cost = {:.3f}'.format(avg_cost) + ' test accuracy: {:.3f}'.format(test_accuracy))
        
        file.write(str(epoch + 1) + ',{:.3f}'.format(avg_cost) + ',{:.3f}\n'.format(test_accuracy))

    print('\nTraining Complete')
    print('\naccurary: {0}\n'.format(sess.run(accurary, feed_dict={input: X_test, y: Y_test})))
    file.close()
    saver.save(sess, '../models/deep')

writer.close()

