import cv2
import matplotlib.pyplot as plt
import os

def read_img(dir_path, filename):
    return cv2.imread(os.path.join(dir_path, filename))


def get_diff_img(img_original, img_modified):
    # check the sizes of two images
    if img_original.shape != img_modified.shape:
        print("you need to resize the image.")
        exit()

    # get difference image between two images
    return cv2.absdiff(img_original, img_modified)


def save_img(img, dir_path, filename):
    # save as .jpg
    result_diff = cv2.imwrite(os.path.join(dir_path, filename), img)
    if result_diff == False:
        print("Can't save the image: diff.jpg")
        exit()


def plot_img_2_times_2(img_original_bgr, img_modified_bgr, img_diff_bgr, img_diff_gray_bgr):
    # change from BGR to RGB for matplotlib
    img_original_rgb = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB)
    img_modified_rgb = cv2.cvtColor(img_modified_bgr, cv2.COLOR_BGR2RGB)
    img_diff_rgb = cv2.cvtColor(img_diff_bgr, cv2.COLOR_BGR2RGB)
    img_diff_gray_rgb = cv2.cvtColor(img_diff_gray_bgr, cv2.COLOR_BGR2RGB)

    # output by matplotlib
    fig, axs = plt.subplots(2, 2, figsize=(10, 15))

    axs[0, 0].imshow(img_original_rgb)
    axs[0, 0].set_title("original")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(img_modified_rgb)
    axs[0, 1].set_title("modified")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(img_diff_rgb)
    axs[1, 0].set_title("difference(colored)")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(img_diff_gray_rgb)
    axs[1, 1].set_title("difference(grayscale)")
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # set a relative path for using the image folder
    dir_path_src = os.path.dirname(__file__)
    dir_path_data = os.path.join(dir_path_src, "..", "data")
    dir_path_output = os.path.join(dir_path_src, "..", "output")

    # read images
    img_original = read_img(dir_path_data, "original01.jpg")
    img_modified = read_img(dir_path_data, "modified01.jpg")

    # get difference
    diff = get_diff_img(img_original, img_modified)

    # get grayscale difference
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # save the images as jpg
    save_img(diff, dir_path_output, "diff.jpg")
    save_img(diff_gray, dir_path_output, "diff_gray.jpg")

    # display as 2 times 2
    plot_img_2_times_2(img_original, img_modified, diff, diff_gray)

if __name__ == "__main__":
    main()