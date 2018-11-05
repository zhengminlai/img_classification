import cv2

img_size = 256

test_image = cv2.imread("q44.jpg")
cv2.imshow('original', test_image)

height = len(test_image)
width = len(test_image[0])
print("height: " + str(height) + ", width: " + str(width))
test_image = test_image[int(height / 5): height, int(width / 5): int(4 * width / 5)]
test_image = cv2.resize(test_image, (img_size, img_size), cv2.INTER_LINEAR) / 255

height_1 = len(test_image)
width_1 = len(test_image[0])
cv2.imshow("resized", test_image)
print("height after resized: " + str(height_1) + ", width after resized: " + str(width_1))

img_1 = test_image[0:height_1, 0: int(width_1 / 3)]

img_2 = test_image[0:height_1, int(width_1 / 3): int(2 * width_1 / 3)]

img_3 = test_image[0:height_1, int(2 * width_1 / 3): int(width_1)]

cv2.imshow("img_1", img_1)
cv2.imshow("img_2", img_2)
cv2.imshow("img_3", img_3)
cv2.waitKey(10000)
