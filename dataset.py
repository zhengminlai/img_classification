import os
import glob
import numpy as np
import cv2
from sklearn.utils import shuffle


def cut_img(img, img_size):
    height = len(img)
    width = len(img[0])
    img = img[int(height / 5): height, int(width / 5): int(4 * width / 5)]

    height_1 = len(img)
    width_1 = len(img[0])

    img_1 = img[0: height_1, 0: int(width_1 / 3)]
    img_2 = img[0: height_1, int(width_1 / 3): int(2 * width_1 / 3)]
    img_3 = img[0: height_1, int(2 * width_1 / 3): int(width_1)]

    img_1 = cv2.resize(img_1, (img_size, img_size), cv2.INTER_LINEAR)
    img_2 = cv2.resize(img_2, (img_size, img_size), cv2.INTER_LINEAR)
    img_3 = cv2.resize(img_3, (img_size, img_size), cv2.INTER_LINEAR)

    images = []

    images.append(img_1)
    images.append(img_2)
    images.append(img_3)

    return images


def load_train(train_path, image_size, classes):
    images = []
    labels = []
    ids = []
    cls = []

    print('Reading training images')
    print("classes:{}".format(classes))

    unqualified_index = classes.index('Unqualified')
    for fld in classes:
        # the data directory has a separate folder for each class,
        # and that each folder is named after the class
        index = classes.index(fld)
        path = os.path.join(train_path, fld, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            cut_imgs = cut_img(image, image_size)
            flbase = os.path.basename(fl)

            fname = flbase.split('.jpg')[0]
            unq_indexes = []
            if fname.__contains__('.'):
                fname = fname.split('.')
                fname.pop(0)
                unq_indexes = [int(i) for i in fname]
                print("unq_indexes:{}".format(unq_indexes))

            for i, cut_img_tmp in enumerate(cut_imgs):
                images.append(cut_img_tmp)
                ids.append(flbase + str(i))
                label = np.zeros(len(classes))
                if i in unq_indexes:
                    label[unqualified_index] = 1.0
                    labels.append(label)
                    cls.append('Unqualified')
                else:
                    label[index] = 1.0
                    labels.append(label)
                    cls.append(fld)

    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, labels, ids, cls


def load_test(test_path, image_size):
    path = os.path.join(test_path, '*g')
    print("Test path:{}".format(path))
    files = sorted(glob.glob(path))

    test_images = []
    test_image_names = []
    print("Reading test images")
    for fl in files:
        flbase = os.path.basename(fl)
        test_image_names.append(flbase)
        img = cv2.imread(fl)
        cut_imgs = cut_img(img, image_size)

        for cut_img_tmp in cut_imgs:
            test_images.append(cut_img_tmp)


    # because we're not creating a DataSet object for the test images,
    # normalization happens here
    test_images = np.array(test_images, dtype=np.uint8)
    test_images = test_images.astype('float32')
    test_images = test_images / 255

    return test_images, test_image_names


class DataSet(object):
    def __init__(self, images, labels, ids, cls):
        """Construct a DataSet. one_hot arg is used only if fake_data is true."""

        self._num_examples = images.shape[0]
        print("num examples: " + str(self._num_examples))
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        # Convert from [0, 255] -> [0.0, 1.0].

        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._ids = ids
        self._cls = cls
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def ids(self):
        return self._ids

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # # Shuffle the data (maybe)
            # perm = np.arange(self._num_examples)
            # np.random.shuffle(perm)
            # self._images = self._images[perm]
            # self._labels = self._labels[perm]
            # Start next epoch

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size=0):
    class DataSets(object):
        pass

    data_sets = DataSets()

    images, labels, ids, cls = load_train(train_path, image_size, classes)
    images, labels, ids, cls = shuffle(images, labels, ids, cls)  # shuffle the data

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_ids = ids[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_ids = ids[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_ids, validation_cls)

    return data_sets


def read_test_set(test_path, image_size):
    images, ids = load_test(test_path, image_size)
    return images, ids
