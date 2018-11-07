import time
import tensorflow as tf
import numpy as np
import dataset

from datetime import timedelta

# class info
classes = ['Qualified', 'Unqualified']
num_classes = len(classes)

# Convolutional Layer 1.
filter_size_1 = 3
num_filters_1 = 32

# Convolutional Layer 2.
filter_size_2 = 3
num_filters_2 = 32

# Convolutional Layer 3.
filter_size_3 = 3
num_filters_3 = 64

# Fully-connected layer.
fc_size = 256  # Number of neurons in fully-connected layer.

# Number of color channels for the images
num_channels = 3

# image dimensions
img_size = 256

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# load images in batch: note that the number of images should be divided evenly
# by `batch_size`
batch_size = 10

# validation ratio
validation_ratio = .2

# how long to wait after validation loss stops improving before terminating training
early_stopping = None

# training images path, we shall split the images into training and validation set
train_path = 'D:/product/dataset/train/'

data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_ratio)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))

# Get some random images and their labels from the train set.
images, cls_true = data.train.images, data.train.cls


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,  # The previous layer.
                   num_input_channels,  # Number of channels in previous layer.
                   filter_size,  # Width and height of each filter.
                   num_filters,  # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights with the shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    layer += biases

    # if use pooling to down-sample the image resolution
    if use_pooling:
        # We use 2x2 max-pooling, which means we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    layer = tf.nn.relu(layer)

    return layer, weights


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The number of features
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


def new_fc_layer(input,  # The previous layer.
                 num_inputs,  # Number of inputs from previous layer.
                 num_outputs,  # Number of outputs.
                 use_relu=True):  # If use Relu

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv_1, weights_conv_1 = new_conv_layer(input=x_image,
                                              num_input_channels=num_channels,
                                              filter_size=filter_size_1,
                                              num_filters=num_filters_1,
                                              use_pooling=True)

layer_conv_2, weights_conv_2 = new_conv_layer(input=layer_conv_1,
                                              num_input_channels=num_filters_1,
                                              filter_size=filter_size_2,
                                              num_filters=num_filters_2,
                                              use_pooling=True)

layer_conv_3, weights_conv_3 = new_conv_layer(input=layer_conv_2,
                                              num_input_channels=num_filters_2,
                                              filter_size=filter_size_3,
                                              num_filters=num_filters_3,
                                              use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv_3)

layer_fc_1 = new_fc_layer(input=layer_flat,
                          num_inputs=num_features,
                          num_outputs=fc_size,
                          use_relu=True)

layer_fc_2 = new_fc_layer(input=layer_fc_1,
                          num_inputs=fc_size,
                          num_outputs=num_classes,
                          use_relu=False)

y_pred = tf.nn.softmax(layer_fc_2)

y_pred_cls = tf.argmax(y_pred, dimension=1)

# save the y_pred_cls to model for future use
tf.add_to_collection('pred_cls', y_pred_cls)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc_2,
                                                        labels=y_true)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.initialize_all_variables())
train_batch_size = batch_size


def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


# Counter for total number of iterations performed so far.
total_iterations = 0


def optimize(num_iterations):
    # Ensure we update the global variable.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]
        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        # Put the batch into a dict with the names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        feed_dict_validate = {x: x_valid_batch,
                              y_true: y_valid_batch}

        # Run the optimizer using this batch of training data.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status at end of each epoch.
        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples / batch_size))

            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)

            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == early_stopping:
                    break

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))


def print_validation_accuracy():
    # Number of images in the validation-set.
    num_validation_set = len(data.valid.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_validation_set, dtype=np.int)

    # Calculate the predicted classes for the batches.
    i = 0

    while i < num_validation_set:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_validation_set)

        # Get the images from the validation-set between index i and j.
        images = data.valid.images[i:j, :].reshape(batch_size, img_size_flat)

        # Get the associated labels.
        labels = data.valid.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    cls_true = np.array(data.valid.cls)
    cls_pred = np.array([classes[x] for x in cls_pred])

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    correct_sum = correct.sum()

    # Classification accuracy.
    acc = float(correct_sum) / num_validation_set

    # Print the accuracy.
    msg = "Accuracy on Validation-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_validation_set))


if __name__ == '__main__':
    # iterate 100 times
    optimize(num_iterations=100)
    print_validation_accuracy()

    saver = tf.train.Saver()
    save_path = saver.save(session, "model/model.ckpt")
    session.close()
