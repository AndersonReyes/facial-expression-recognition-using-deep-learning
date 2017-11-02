import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import skimage.transform
from scipy.misc import imread, imresize
from skimage.color import rgb2grey
from skimage.io import imshow, imshow_collection
from load_data import load_data


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
                4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
batch = []
originals = []

# PREPROCESS PASSED IMAGES AND ADD THEM TO A LIST
for path in sys.argv[1:]:
    img_ndarray = imread(path)
    originals.append(np.copy(img_ndarray))
    img_ndarray = rgb2grey(img_ndarray)
    img_ndarray = imresize(img_ndarray, (48, 48))
    img_ndarray = img_ndarray.astype('float32')
    img_ndarray = np.multiply(img_ndarray, 1.0/255.0)
    flatten_img = img_ndarray.flatten()
    batch.append(flatten_img)

with tf.Session() as sess:
    # LOAD SAVED WEIGHTS INTO THE CURRENT SEESION AND CREATE THE GRAPH
    saver = tf.train.import_meta_graph('../models/32_64_128_fc512_100epochs.meta')
    saver.restore(sess, tf.train.latest_checkpoint('../models'))
    graph = tf.get_default_graph()
    
    # WE NEED TO GET THE PREDICTION GRAPH NODE 
    y_predict = graph.get_tensor_by_name('y_predict:0')
    x = graph.get_tensor_by_name('Input:0')

    # GET THE PREDICTIONS AND NORMALIZE RESULTS (INTERPRET AS PROBABILITIES)
    feed_dict = {x: batch}
    result = sess.run(y_predict, feed_dict=feed_dict)

    predicted = []
    for res in result:
        predicted.append(emotion_dict[np.argmax(res)])

    emoji_imgs = []
    for img, emotio in zip(originals, predicted):
        emoji = skimage.io.imread( './emojis/' + emotion + '.png')
        emoji = skimage.transform.resize(emoji, img.shape)
        emoji_imgs.extend([img, emoji])

    imshow_collection(emoji_imgs)
    plt.show()