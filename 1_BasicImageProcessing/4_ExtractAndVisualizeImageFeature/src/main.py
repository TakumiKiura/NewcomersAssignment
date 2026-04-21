import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def read_img(dir_path, filename):
    return cv2.imread(os.path.join(dir_path, filename))

def save_img(img, dir_path, filename):
    # save as jpg
    result_diff = cv2.imwrite(os.path.join(dir_path, filename), img)
    if result_diff == False:
        print("Can't save the image: ", filename)
        exit()


def generate_hist_grayscale(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()

    # normalization
    return hist/hist.sum()


def generate_hist_rgb(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256]).flatten()

    # normalization
    return hist_r/hist_r.sum(), hist_g/hist_g.sum(), hist_b/hist_b.sum()


def display_grayscale_hist(hist1, hist2, img1_bgr, img1_caption, img2_bgr, img2_caption):
    # bgr to rgb for matplotlib
    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

    # bgr to gray
    img1_gray = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

    # plot start
    plt.figure(figsize=(12, 8))

    # img1
    plt.subplot(3, 2, 1)
    plt.imshow(img1_rgb)
    plt.title(img1_caption)
    plt.axis("off")

    # img2
    plt.subplot(3, 2, 2)
    plt.imshow(img2_rgb)
    plt.title(img2_caption)
    plt.axis("off")

    # img1_gray
    plt.subplot(3, 2, 3)
    plt.imshow(img1_gray, cmap="gray")
    plt.title(img1_caption+"(grayscale)")
    plt.axis("off")

    # img2_gray
    plt.subplot(3, 2, 4)
    plt.imshow(img2_gray, cmap="gray")
    plt.title(img2_caption+"(grayscale)")
    plt.axis("off")

    # hist
    plt.subplot(3, 2, 5)
    plt.plot(hist1, label=img1_caption, linewidth=2, alpha=0.7)
    plt.plot(hist2, label=img2_caption, linewidth=2, alpha=0.7)
    plt.title("Grayscale Histogram Comparison")
    plt.xlabel("Pixel Value")
    plt.ylabel("Normalized Frequency")
    plt.xlim(0, 255)
    plt.legend()
    plt.grid()

    # diff
    plt.subplot(3, 2, 6)
    plt.plot(np.abs(hist1-hist2), linewidth=2)
    plt.title("Histgram Differences")
    plt.xlabel("Pixel Value")
    plt.ylabel("Normalized Frequency")
    plt.xlim(0, 255)
    plt.grid()

    # show
    plt.tight_layout()
    plt.show()


def main():
    # set a relative path for using the image folder
    dir_path_src = os.path.dirname(__file__)
    dir_path_data = os.path.join(dir_path_src, "..", "data")
    dir_path_output = os.path.join(dir_path_src, "..", "output")

    # read images
    img_01 = read_img(dir_path_data, "sample_01.png")
    img_02 = read_img(dir_path_data, "sample_02.jpg")

    # calc hist
    hist_01 = generate_hist_grayscale(img_01)
    hist_02 = generate_hist_grayscale(img_02)

    # display the hist
    display_grayscale_hist(hist_01, hist_02, img_01, "sample_01.png", img_02, "sample_02.jpg")


if __name__ == "__main__":
    main()