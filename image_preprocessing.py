import os
import glob
import cv2

def cut_img(img, img_size, flag):
    height = len(img)
    width = len(img[0])
    img = img[int(height / 5): height, int(width / 5): int(4 * width / 5)]

    height_1 = len(img)
    width_1 = len(img[0])

    img_1 = img[0:height_1, 0: int(width_1 / 3)]
    img_2 = img[0:height_1, int(width_1 / 3): int(2 * width_1 / 3)]
    img_3 = img[0:height_1, int(2 * width_1 / 3): int(width_1)]

    img_1 = cv2.resize(img_1, (img_size, img_size), cv2.INTER_LINEAR)
    img_2 = cv2.resize(img_2, (img_size, img_size), cv2.INTER_LINEAR)
    img_3 = cv2.resize(img_3, (img_size, img_size), cv2.INTER_LINEAR)

    if flag:
        cv2.imshow("img1", img_1)
        cv2.imshow("img2", img_2)
        cv2.imshow("img3", img_3)
    images = []

    images.append(img_1)
    images.append(img_2)
    images.append(img_3)
    return images


def load_train(train_path, image_size, classes):

    flag = True
    print('Reading training images')
    print("classes:{}".format(classes))
    for fld in classes:
        # the data directory has a separate folder for each class,
        # and that each folder is named after the class
        index = classes.index(fld)
        print("index: " + str(index))
        path = os.path.join(train_path, fld, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            if flag:
                cv2.imshow("img", image)
            cut_imgs = cut_img(image, image_size, flag)
            flbase = os.path.basename(fl)
            print("flbase:" + flbase)
            if flag:
                cv2.imshow("img_1", cut_imgs[0])
                cv2.imshow("img_2", cut_imgs[1])
                cv2.imshow("img_3", cut_imgs[2])
                cv2.waitKey(4000)
                flag = False


# training images path, we shall split the images into training and validation set
train_path = '/home/zhengminlai/下载/dataset/train/'
image_size = 256
classes = ["Qualified", "Unqualified"]
load_train(train_path, image_size, classes)

# img_size = 256
#
# test_image = cv2.imread("q44.jpg")
# cv2.imshow('original', test_image)
#
# height = len(test_image)
# width = len(test_image[0])
# print("height: " + str(height) + ", width: " + str(width))
# test_image = test_image[int(height / 5): height, int(width / 5): int(4 * width / 5)]
# cv2.imshow("resized", test_image)
#
# height_1 = len(test_image)
# width_1 = len(test_image[0])
#
# img_1 = test_image[0:height_1, 0: int(width_1 / 3)]
# img_2 = test_image[0:height_1, int(width_1 / 3): int(2 * width_1 / 3)]
# img_3 = test_image[0:height_1, int(2 * width_1 / 3): int(width_1)]
#
# img_1 = cv2.resize(img_1, (img_size, img_size), cv2.INTER_LINEAR)
# img_2 = cv2.resize(img_2, (img_size, img_size), cv2.INTER_LINEAR)
# img_3 = cv2.resize(img_3, (img_size, img_size), cv2.INTER_LINEAR)
#
# print("height after resized: " + str(len(img_1)) + ", width after resized: " + str(len(img_1[0])))
#
# cv2.imshow("img_1", img_1)
# cv2.imshow("img_2", img_2)
# cv2.imshow("img_3", img_3)
# cv2.waitKey(10000)
