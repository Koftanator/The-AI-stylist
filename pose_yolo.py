try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
import cv2
import numpy as np


_model = None

if YOLO is not None:
    try:
        # Use a built-in pose model name; ultralytics will download weights if needed
        _model = YOLO('yolov8n-pose.pt')
    except Exception:
        _model = None


def get_landmarks(frame, conf_threshold=0.25):
    """Run YOLOv8 pose on the frame and return dict of keypoints matching MoveNet indices.

    Returns: dict {index: (x,y) or None} or None if model unavailable.
    """
    if _model is None:
        return None

    # ultralytics expects BGR images; pass frame directly
    results = _model(frame, imgsz=640)[0]

    if not hasattr(results, 'keypoints') or results.keypoints is None:
        return None

    kp_obj = results.keypoints
    # Extract xy coordinates and confidences via the Keypoints API
    try:
        # move to cpu if needed
        kp_cpu = kp_obj.cpu() if hasattr(kp_obj, 'cpu') else kp_obj
        # xy: (num_people, num_kpts, 2)
        xy = kp_cpu.xy.numpy()
        # conf: (num_people, num_kpts) or (num_people, num_kpts, 1)
        if hasattr(kp_cpu, 'conf') and kp_cpu.conf is not None:
            conf = kp_cpu.conf.numpy()
            # if conf has shape (...,1), squeeze
            if conf.ndim == 3 and conf.shape[2] == 1:
                conf = conf[..., 0]
        else:
            # try to look for a third value in raw data
            try:
                raw = kp_cpu.numpy()
                if raw.ndim == 3 and raw.shape[2] == 3:
                    conf = raw[..., 2]
                else:
                    conf = None
            except Exception:
                conf = None
    except Exception:
        return None

    # Select first detected person
    if xy.ndim == 3 and xy.shape[0] >= 1:
        person_xy = xy[0]
    elif xy.ndim == 2:
        person_xy = xy
    else:
        return None

    person_conf = None
    if conf is not None:
        if conf.ndim == 2 and conf.shape[0] >= 1:
            person_conf = conf[0] if conf.shape[0] > 1 else conf
        elif conf.ndim == 1:
            person_conf = conf

    points = {}
    for idx, coord in enumerate(person_xy):
        x, y = coord[0], coord[1]
        c = None
        if person_conf is not None and idx < len(person_conf):
            try:
                val = np.squeeze(person_conf[idx])
                c = float(val)
            except Exception:
                try:
                    c = float(person_conf[idx])
                except Exception:
                    c = None
        # default to detected if no confidence available
        if c is None:
            ok = True
        else:
            ok = (c >= conf_threshold)
        if not ok:
            points[idx] = None
        else:
            points[idx] = (int(x), int(y))
    return points
