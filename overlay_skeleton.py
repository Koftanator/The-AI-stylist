import cv2
from skeleton import BONES

def draw_skeleton(frame, landmarks, color=(0,255,0), radius=4, thickness=2):
    # Draw bones
    for start, end in BONES:
        pt1 = landmarks.get(start)
        pt2 = landmarks.get(end)
        if pt1 and pt2:
            cv2.line(frame, pt1, pt2, color, thickness)
    # Draw keypoints
    for idx, pt in landmarks.items():
        if pt:
            cv2.circle(frame, pt, radius, color, -1)
    return frame
