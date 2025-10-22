# src/skeleton.py

# MoveNet keypoint indices
KEYPOINT_NAMES = {
    0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
    5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
    9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
}

# Define bones as pairs of keypoint indices
BONES = [
    (0, 1), (0, 2),             # head
    (1, 3), (2, 4),             # ears
    (5, 6),                     # shoulders
    (5, 7), (7, 9),             # left arm
    (6, 8), (8, 10),            # right arm
    (11, 12),                   # hips
    (5, 11), (6, 12),           # torso
    (11, 13), (13, 15),         # left leg
    (12, 14), (14, 16)          # right leg
]
