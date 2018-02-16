import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from load_data import load_data


emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
                4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

X_all, Y_all = load_data('../data/fer2013.csv')
assert len(X_all) == len(Y_all)

# save 20% for testing
X_train, Y_train, X_test, Y_test = train_test_split(X_all, Y_all, test_size=0.20, random_sate=20)

alpha = 0.0001
epochs = 70
batch_size = 256

# INPUT
input = tf.placeholder(dtype=tf.float32, shape=[None, 2304], name='Input')
input_shaped = tf.reshape(input, [-1, 48, 48, 1])
y = tf.placeholder(dtype=tf.float32, shape=[None, 7], name='Output')

#  ARCHITECTURE
layer1 = tf.layers.conv2d(input_shaped, filters=32, kernel_size=[5, 5],  padding='same', activation=tf.nn.relu, name='layer1')
layer1 = tf.layers.max_pooling2d(layer1, pool_size=[2, 2], strides=2, name='layer1')

layer2 = tf.layers.conv2d(layer1, filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='layer2')
layer2 = tf.layers.max_pooling2d(layer2, pool_size=[2, 2], strides=2, name='layer2')

layer3 = tf.layers.conv2d(layer2, filters=128, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='layer3')
layer3 = tf.layers.max_pooling2d(layer3, pool_size=[2, 2], strides=2, name='layer3')

flattened = tf.reshape(layer3, [-1, 6 * 6 * 128], name='flattened')
dense1024 = tf.layers.dense(flattened, units=1024, activation=tf.nn.relu, name='dense_1024')
dropout = tf.layers.dropout(dense1024, rate=0.5, name='dropout')
dense512 = tf.layers.dense(dropout, units=512, activation=tf.nn.relu, name='dense_512')
dropout2 = tf.layers.dropout(dense512, rate=0.5, name='dropout2')
logits = tf.layers.dense(dropout2, units=7, name='dense_7')

probs = tf.nn.softmax(logits, name='y_softmax')
y_predict = tf.argmax(probs, axis=1, name='y_predict')
y_true = tf.argmax(y, axis=1, name='y_true')

entropy_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimiser = tf.train.AdamOptimizer(learning_rate=alpha).minimize(entropy_cost)
correct_prediction = tf.equal(y_true, y_predict)
accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()


saver = tf.train.Saver()
history = {}

# write output to a file
file = open('../training_steps.txt', 'w')

with tf.Session() as sess:
    sess.run(init_op)

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
        history[str(epoch)] = test_accuracy

        file.write('\nEpoch:' + str(epoch + 1) + 'cost = {:.3f}'.format(avg_cost) + ' test accuracy: {:.3f}'.format(test_accuracy))

    file.write('\nTraining Complete')
    file.write('\naccurary: {0}'.format(sess.run(accurary, feed_dict={input: X_test, y: Y_test})))
    file.close()	
    saver.save(sess, 'model-small')

df = pd.DataFrame(history.items(), columns=['epoch', 'accuracy'])
df.to_csv('../models/small_model_training_accuracies.csv')
