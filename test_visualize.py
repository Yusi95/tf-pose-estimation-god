import numpy as np
import matplotlib.pyplot as plt
import cv2


def draw_point(img, loc):
    height, width = img.shape[:2]
    x = int(width * loc[0])
    y = int(height * loc[1])
    img[y:y + 5, x:x + 5] = np.array([0, 255, 0])
    return img
