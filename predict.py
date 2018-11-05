import numpy as np
import sys
import tensorflow as tf
import cv2

# class info
classes = ['Qualified', 'Unqualified']

session = tf.Session()

ckpt = tf.train.get_checkpoint_state('model/')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
saver.restore(session, ckpt.model_checkpoint_path)

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 256

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

graph = tf.get_default_graph()

x = graph.get_operation_by_name('x').outputs[0]
y_true = graph.get_operation_by_name('y_true').outputs[0]

y = tf.get_collection('pred_cls')[0]


def sample_prediction(test_im):
    feed_dict_test = {
        x: test_im.reshape(1, img_size_flat),
        y_true: np.array([[1, 0]])
    }

    test_pred = session.run(y, feed_dict=feed_dict_test)
    return classes[test_pred[0]]

if __name__ == '__main__':
    test_file_addr = sys.argv[1]
    test_image = cv2.imread(test_file_addr)
    test_image = cv2.resize(test_image, (img_size, img_size), cv2.INTER_LINEAR) / 255

    print("Predicted class for test image: {}".format(sample_prediction(test_image)))
