import cv2

img_size = 256

test_image = cv2.imread("test.jpg")
cv2.imshow('original', test_image)

print("height: " + str(len(test_image)) + ", width: " + str(len(test_image[0])))
test_image = cv2.resize(test_image, (img_size, img_size), cv2.INTER_LINEAR) / 255

cv2.imshow('resized', test_image)
print("height: " + str(len(test_image)) + ", width: " + str(len(test_image[0])))

cv2.imwrite('resized.jpg', test_image)
cv2.waitKey(20171219)
