import cv2
import matplotlib.pyplot as plt
import os

# set a relative path for using the image folder
dir_path_src = os.path.dirname(__file__)
dir_path_data = os.path.join(dir_path_src, "..", "data")
dir_path_output = os.path.join(dir_path_src, "..", "output")

# read images
img_original = cv2.imread(os.path.join(dir_path_data, "original01.jpg"))

img_modified = cv2.imread(os.path.join(dir_path_data, "modified01.jpg"))

# check the sizes of two images
if img_original.shape != img_modified.shape:
    print("you need to resize the image.")
    exit()

# get difference image between two images
diff = cv2.absdiff(img_original, img_modified)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# save as .jpg
result_diff = cv2.imwrite(os.path.join(dir_path_output, "diff.jpg"), diff)
if result_diff == False:
    print("Can't save the image: diff.jpg")
    exit()

result_diff_gray = cv2.imwrite(os.path.join(dir_path_output, "diff_gray.jpg"), diff_gray)
if result_diff_gray == False:
    print("Can't save the image: diff_gray.jpg")
    exit()

# change from BGR to RGB for matplotlib
img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
img_modified_rgb = cv2.cvtColor(img_modified, cv2.COLOR_BGR2RGB)
img_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
img_diff_gray = cv2.cvtColor(diff_gray, cv2.COLOR_BGR2RGB)

# output by matplotlib
fig, axs = plt.subplots(2, 2, figsize=(10, 15))

axs[0, 0].imshow(img_original_rgb)
axs[0, 0].set_title("original")
axs[0, 0].axis("off")

axs[0, 1].imshow(img_modified_rgb)
axs[0, 1].set_title("modified")
axs[0, 1].axis("off")

axs[1, 0].imshow(img_diff)
axs[1, 0].set_title("difference(colored)")
axs[1, 0].axis("off")

axs[1, 1].imshow(img_diff_gray)
axs[1, 1].set_title("difference(grayscale)")
axs[1, 1].axis("off")

plt.tight_layout()
plt.show()