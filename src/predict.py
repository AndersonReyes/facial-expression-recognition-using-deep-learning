import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.misc import imread
from skimage.color import rgb2grey
from skimage.io import imshow

emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
                4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
batch = []
for path in sys.argv[1:]:
    img_ndarray = imread(path)
    img_ndarray = rgb2grey(img_ndarray)
    img_ndarray = img_ndarray.astype('float32')
    img_ndarray = np.multiply(img_ndarray, 1.0/255.0)
    flatten_img = img_ndarray.flatten()
    batch.append(flatten_img)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('../models/32_64_128_fc512_100epochs.meta')
    saver.restore(sess, tf.train.latest_checkpoint('../models'))

    graph = tf.get_default_graph()
    # namelist = [node.name for node in tf.get_default_graph().as_graph_def().node]
    # print(*namelist, sep='\n')
    y_predict = graph.get_tensor_by_name('y_predict:0')
    x = graph.get_tensor_by_name('Input:0')
    feed_dict = {x: batch}
    
    result = sess.run(y_predict, feed_dict=feed_dict)
    out = []
    for res in result:
        out.append(np.multiply(result, 1 / np.sum(result)))
        print(emotion_dict[np.argmax(result)])