import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


"""
Given a realistic texture, we cartoonize it using k means clustering to
create a cohesive visual style.
"""


def add_contrast(im, clip_limit: float):
    # see https://stackoverflow.com/questions/19363293/whats-the-fastest-way-to-increase-color-image-contrast-with-opencv-in-python-c

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2, a, b))  # merge channels
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR


def blur_kmeans(path, blur_kernel: int, K: int, clip_limit: float):
    print(blur_kernel, K)
    im = cv2.imread(path)
    min_dim = min(im.shape[0], im.shape[1])
    im = im[:min_dim, :min_dim]  # squarify
    im = add_contrast(im, clip_limit)

    blurred = cv2.medianBlur(im, blur_kernel)
    # blurred = cv2.GaussianBlur(im, (blur_kernel, blur_kernel), 0)
    # blurred.resize((150, 150))

    flattened = blurred.reshape((-1, 3)).astype(np.float32)  # flatten image

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(flattened, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = center.astype(np.uint8)
    res = center[label.flatten()]

    # test tiling
    if False:
        tiled_ex = cv2.cvtColor(res.reshape((im.shape)), cv2.COLOR_BGR2RGB)
        hstack = cv2.hconcat([tiled_ex, tiled_ex])
        plt.imshow(cv2.vconcat([hstack, hstack]))
        plt.show()

    return res.reshape((blurred.shape))


ideal_settings = {
    'cobblestone_medieval.tif': (9, 3, 1.),
    'cobblestone_mixed.tif': (3, 3, 1.),
    'cobblestone_mossy.tif': (5, 3, 1.),
    'faux_cobblestone.tif': (11, 3, 4.),
    'pavement_regular.tif': (7, 3, 0.5),
    'sandstone_wall.tif': (7, 4, 1.),
    'stone_wall.tif': (3, 3, 3.)
}


def main():
    for texture in os.listdir('./source'):
        blur_kernel, K, clip_limit = 7, 3, 1.
        if texture in ideal_settings:
            blur_kernel, K, clip_limit = ideal_settings[texture]

        quantized = blur_kmeans(f'./source/{texture}', blur_kernel, K, clip_limit)
        cv2.imwrite(f'./twod/{texture.replace(".tif", "")}.png', quantized)

        if False:
            quantized = cv2.cvtColor(quantized, cv2.COLOR_BGR2RGB)
            plt.imshow(quantized)
            plt.show()


if __name__ == '__main__':
    main()
