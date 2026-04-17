import cv2
import os

# set a relative path for using the image folder
dir_path_src = os.path.dirname(__file__)
dir_path_img = os.path.join(dir_path_src, "..", "img")

# read images
img_original = cv2.imread(os.path.join(dir_path_img, "original01.jpg"))
cv2.imshow("original", img_original)
cv2.waitKey(0)
cv2.destroyWindow("original")

img_modified = cv2.imread(os.path.join(dir_path_img, "modified01.jpg"))
cv2.imshow("modified", img_modified)
cv2.waitKey(0)
cv2.destroyWindow("modified")

# check the sizes of two images
if img_original.shape != img_modified.shape:
    print("you need to resize the image.")
    exit()

# get difference between two images
diff = cv2.absdiff(img_original, img_modified)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

cv2.imshow("diff", diff)
cv2.waitKey(0)
cv2.destroyWindow("diff")

cv2.imshow("diff_gray", diff_gray)
cv2.waitKey(0)
cv2.destroyWindow("diff_gray")

cv2.destroyAllWindows()