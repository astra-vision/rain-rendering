import os

import cv2
import numpy as np
import tqdm

from common import my_utils


def improve_depth(image, depth, threshold=0.001, threshold_faraway_planes=False):
    window_size = 20

    width = image.shape[0]
    height = image.shape[1]
    if threshold_faraway_planes:
        # NOTE: This could be PERHAPS useful for cases where the depth map is really bad / inexistent
        # for faraway planes; unchanging neighborhood in depth image sometimes means no data which,
        # generally, means too close or too far for measurement; this is dangerous and should probably be done offline
        for i in range(0, width - window_size, window_size // 5):
            for j in range(0, height - window_size, window_size // 5):
                patch = image[i:i + window_size, j:j + window_size]

                if np.std(patch) < threshold:
                    depth[i:i + window_size, j:j + window_size] = 300

    depth = cv2.GaussianBlur(depth, (7, 7), 1)
    return depth


def process_all(images_path, depth_path, output_path):
    img_names = my_utils.os_listdir(images_path)
    depth_names = my_utils.os_listdir(depth_path)
    beta = 0
    pbar = tqdm.tqdm(total=len(img_names))
    for name_file, depth_file in zip(img_names, depth_names):
        pbar.update(1)
        img = cv2.imread(os.path.join(images_path, name_file))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        # divided by 256 to convert it into metres
        original_depth = cv2.imread(os.path.join(depth_path, depth_file), cv2.IMREAD_UNCHANGED) / 256
        smooth_depth = improve_depth(gray_img, original_depth, threshold=beta)
        np.save(os.path.join(output_path, name_file), smooth_depth)
