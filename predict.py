import numpy as np
import sys
import tensorflow as tf
import dataset

# class info
classes = ['Qualified', 'Unqualified']

session = tf.Session()

ckpt = tf.train.get_checkpoint_state('model/')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
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


def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()


if __name__ == '__main__':
    # the test image directory to be predicted
    test_img_dir = sys.argv[1]

    # the output file where it records the prediction results
    output_addr = sys.argv[2]

    test_imgs, test_img_names = dataset.read_test_set(test_img_dir, img_size)

    output_str = ''
    i = 0
    for img_name in test_img_names:
        j = 1
        predict_class = []
        is_qualified = True
        unqualified_indexes = []
        while j <= 3:
            test_image = test_imgs[i]
            predict_rst = sample_prediction(test_image)
            predict_class.append(predict_rst)
            if predict_rst == 'Unqualified':
                unqualified_indexes.append(j)
                if is_qualified:
                    is_qualified = False
            j += 1
            i += 1
        if is_qualified:
            output_str += "{}: {}\n".format(img_name, 'Qualified')
        else:
            output_str += "{}: {}, {}\n".format(img_name, 'Unqualified', unqualified_indexes)
    save_to_file(output_addr, output_str)
