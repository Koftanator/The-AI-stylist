import os
import cv2
import numpy as np

try:
    import tensorflow as tf
except Exception:
    tf = None

# Try a list of candidate paths for the TFLite model. The user can override with
# the MOVENET_MODEL_PATH environment variable.
_candidates = [
    os.getenv('MOVENET_MODEL_PATH'),
    os.path.join(os.path.dirname(__file__), 'movenet.tflite'),
    os.path.join(os.getcwd(), 'src', 'movenet.tflite'),
    os.path.join(os.getcwd(), 'movenet.tflite'),
]

_model_path = next((p for p in _candidates if p and os.path.exists(p)), None)

_has_interpreter = False
_interpreter = None
_input_details = None
_output_details = None

if _model_path is None:
    # Don't raise here: allow the rest of the app to run without pose estimation.
    # main.py will continue and get_landmarks() will return None.
    print("WARNING: movenet.tflite not found. Pose estimation disabled.\n"
          "Place a MoveNet TFLite model at one of these paths or set MOVENET_MODEL_PATH:\n  "
          + "\n  ".join([p for p in _candidates if p]))
else:
    if tf is None:
        print(f"WARNING: TensorFlow not available; cannot load model at '{_model_path}'.")
    else:
        try:
            _interpreter = tf.lite.Interpreter(model_path=_model_path)
            _interpreter.allocate_tensors()
            _input_details = _interpreter.get_input_details()
            _output_details = _interpreter.get_output_details()
            _has_interpreter = True
        except Exception as e:
            print(f"WARNING: Failed to initialize TFLite interpreter: {e}")


def get_landmarks(frame):
    """Return a list of (x,y) tuples for keypoints or None if pose model unavailable.

    Each list element is either a (x, y) pixel coordinate or None when the keypoint
    confidence is below threshold. If the model/interpreter is not available this
    returns None so callers can continue without pose information.
    """
    if not _has_interpreter or _interpreter is None:
        return None

    # Prepare input for the model. Some MoveNet variants use 192x192, others 256.
    # Use the model input shape if available, otherwise fall back to 256.
    h, w = frame.shape[:2]
    input_shape = _input_details[0]['shape'] if _input_details is not None else None
    if input_shape is not None and len(input_shape) >= 2:
        target_h, target_w = int(input_shape[1]), int(input_shape[2])
    else:
        target_h = target_w = 256

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(img, (target_w, target_h)).astype(np.float32)
    inp = inp[np.newaxis, ...] / 255.0

    try:
        if _input_details is None or _output_details is None:
            print("WARNING: Interpreter details missing; cannot run model.")
            return None
        _interpreter.set_tensor(_input_details[0]['index'], inp)
        _interpreter.invoke()
        out = _interpreter.get_tensor(_output_details[0]['index'])
    except Exception as e:
        print(f"WARNING: Error running interpreter: {e}")
        return None

    kpts = np.squeeze(out)
    # Some TFLite outputs are (1, N, 3) or (N, 3). Normalize to (N, 3).
    if kpts.ndim == 3 and kpts.shape[0] == 1:
        kpts = kpts[0]

    # Convert to a dict keyed by keypoint index so callers can use .get(index)
    points = {}
    for idx, (y, x, score) in enumerate(kpts):
        if score < 0.3:
            points[idx] = None
        else:
            points[idx] = (int(x * w), int(y * h))
    return points

