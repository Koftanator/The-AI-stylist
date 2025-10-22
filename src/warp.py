import cv2
import numpy as np


def _transparent_canvas(frame):
    h, w = frame.shape[:2]
    return np.zeros((h, w, 4), dtype=np.uint8)


def warp_image(frame, garment_rgba, landmarks):
    """Warp the RGBA garment to the torso defined by shoulders and hips.

    Expects `landmarks` to be a dict mapping indices to (x,y) or None.
    Uses keypoints: left_shoulder=5, right_shoulder=6, left_hip=11, right_hip=12.
    If inputs are invalid or keypoints are missing, returns a transparent image
    sized to the frame so overlay() can safely run.
    """
    # Validate inputs
    if garment_rgba is None:
        return _transparent_canvas(frame)
    if landmarks is None:
        return _transparent_canvas(frame)

    # keypoint indices
    LS, RS, LH, RH = 5, 6, 11, 12
    pts = [landmarks.get(LS), landmarks.get(RS), landmarks.get(RH), landmarks.get(LH)]
    if any(p is None for p in pts):
        # Missing required keypoints
        return _transparent_canvas(frame)

    try:
        h, w = garment_rgba.shape[:2]
        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst_pts = np.array(pts, dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            garment_rgba, M, (frame.shape[1], frame.shape[0]),
            borderMode=cv2.BORDER_TRANSPARENT, borderValue=(0, 0, 0, 0)
        )
        return warped
    except Exception:
        return _transparent_canvas(frame)
