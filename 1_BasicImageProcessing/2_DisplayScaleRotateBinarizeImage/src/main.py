import cv2

img = cv2.imread("img/img01.jpg")

# display an image
cv2.imshow("HelloWorld",img)
cv2.waitKey(0)
cv2.destroyWindow("HelloWorld")


#
# change scale
#

# 2 times width
h, w = img.shape[:2]
img_width_2_times = cv2.resize(img, (w*2, h))
cv2.imshow("2 times width", img_width_2_times)
cv2.waitKey(0)
cv2.destroyWindow("2 times width")

# one-quarter height

# !!! the following line occurs cv2.error because param 'dsize' only allows integer
# img_height_one_quarter = cv2.resize(img, (w, h/4))

# correct line
img_height_one_quarter = cv2.resize(img, None, fx=1, fy=0.25) # type: ignore
cv2.imshow("one-quarter height", img_height_one_quarter)
cv2.waitKey(0)
cv2.destroyWindow("one-quarter height")


#
# rotate img
#

# 1. using openCV
img_rotate_90_clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("rotate 90 clockwise", img_rotate_90_clockwise)
cv2.waitKey(0)
cv2.destroyWindow("rotate 90 clockwise")



#
# binarize img
#

# load the img in grayscale
img_grayscale = cv2.imread("img/img01.jpg",0)
cv2.imshow("grayscaled img", img_grayscale)
cv2.waitKey(0)
cv2.destroyWindow("grayscaled img")

# set a threshold
threshold = 100
ret, img_threshold = cv2.threshold(img_grayscale, threshold, 255, cv2.THRESH_BINARY)
cv2.imshow("threshold img", img_threshold)
cv2.waitKey(0)
cv2.destroyWindow("threshold img")

# auto threshold
ret2, img_auto_threshold = cv2.threshold(img_grayscale, 0, 255, cv2.THRESH_OTSU)
print("threshold(ret2): ", ret2)

cv2.imshow("auto threshold img", img_auto_threshold)
cv2.waitKey(0)
cv2.destroyWindow("auto threshold img")

cv2.destroyAllWindows()