import cv2
import numpy as np
def overlay(frame, warped_rgba):
    alpha = warped_rgba[:, :, 3] / 255.0
    for c in range(0, 3):
        frame[:, :, c] = (1 - alpha) * frame[:, :, c] + alpha * warped_rgba[:, :, c]
    return frame.astype(np.uint8)
