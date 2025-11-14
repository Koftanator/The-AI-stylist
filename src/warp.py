import cv2
import numpy as np
from typing import Optional, Literal

# Import TPS warping and garment mapping
try:
    from tps_warp import create_tps_warper, TPSWarper
    from garment_mapping import get_control_point_pairs, check_keypoint_stability
    TPS_AVAILABLE = True
except ImportError as e:
    TPS_AVAILABLE = False
    print(f"[WARP] TPS not available: {e}")


# Global TPS warper instance (created once, reused for performance)
_tps_warper: Optional[TPSWarper] = None
_warp_mode: str = 'perspective'  # 'perspective', 'affine', or 'tps'


def _transparent_canvas(frame):
    h, w = frame.shape[:2]
    return np.zeros((h, w, 4), dtype=np.uint8)


def set_warp_mode(mode: str):
    """
    Set the active warping mode.
    
    Args:
        mode: 'perspective' (default, 4-point homography),
              'affine' (6-point affine transform),
              'tps' (thin-plate spline, organic warping)
    """
    global _warp_mode, _tps_warper
    
    if mode not in ['perspective', 'affine', 'tps']:
        raise ValueError(f"Invalid warp mode: {mode}")
    
    if mode == 'tps' and not TPS_AVAILABLE:
        print("[WARP] TPS mode requested but not available, falling back to perspective")
        _warp_mode = 'perspective'
        return
    
    _warp_mode = mode
    
    # Initialize TPS warper if needed
    if mode == 'tps' and _tps_warper is None:
        _tps_warper = create_tps_warper(fast_mode=True, gpu_enabled=True)
        print(f"[WARP] TPS warper initialized: {_tps_warper.get_stats()}")


def get_warp_mode() -> str:
    """Get current warp mode."""
    return _warp_mode


def warp_image_perspective(frame, garment_rgba, landmarks):
    """
    Original perspective warp (4-point homography).
    Fast but rigid transformation.
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


def warp_image_affine(frame, garment_rgba, landmarks):
    """
    Affine warp (6-point transformation).
    Middle ground between perspective and TPS.
    """
    if garment_rgba is None:
        return _transparent_canvas(frame)
    if landmarks is None:
        return _transparent_canvas(frame)

    # Use 6 points for affine: shoulders, hips, and waist sides
    LS, RS, LH, RH = 5, 6, 11, 12
    required_pts = [landmarks.get(idx) for idx in [LS, RS, LH, RH]]
    
    if any(p is None for p in required_pts):
        return _transparent_canvas(frame)

    try:
        h, w = garment_rgba.shape[:2]
        
        # Define 3 source points on garment
        src_pts = np.array([
            [0, 0],         # top-left (left shoulder)
            [w, 0],         # top-right (right shoulder)
            [w//2, h]       # bottom-center (hip center)
        ], dtype=np.float32)
        
        # Map to body landmarks
        left_shoulder = landmarks[LS]
        right_shoulder = landmarks[RS]
        left_hip = landmarks[LH]
        right_hip = landmarks[RH]
        hip_center = ((left_hip[0] + right_hip[0]) // 2, 
                      (left_hip[1] + right_hip[1]) // 2)
        
        dst_pts = np.array([
            left_shoulder,
            right_shoulder,
            hip_center
        ], dtype=np.float32)
        
        M = cv2.getAffineTransform(src_pts, dst_pts)
        warped = cv2.warpAffine(
            garment_rgba, M, (frame.shape[1], frame.shape[0]),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )
        return warped
    except Exception:
        return _transparent_canvas(frame)


def warp_image_tps(frame, garment_rgba, landmarks, garment_type='shirt_front'):
    """
    TPS warp with organic deformation.
    
    Uses dense control point mapping for natural garment fit.
    Falls back to perspective if keypoints insufficient.
    
    RTX 4060 Optimizations:
    - Cached TPS weights (recompute only on significant movement)
    - GPU-accelerated grid transformation via CuPy
    - Downsampled warping resolution
    """
    global _tps_warper
    
    if garment_rgba is None:
        return _transparent_canvas(frame)
    if landmarks is None:
        return _transparent_canvas(frame)
    if _tps_warper is None:
        # Initialize on first use
        _tps_warper = create_tps_warper(fast_mode=True, gpu_enabled=True)
    
    # Check if we have stable keypoints for TPS
    is_stable, confidence = check_keypoint_stability(landmarks)
    
    if not is_stable:
        # Fallback to perspective warp if pose detection is poor
        return warp_image_perspective(frame, garment_rgba, landmarks)
    
    try:
        # Get control point pairs for TPS
        src_pts, dst_pts = get_control_point_pairs(
            garment_rgba, landmarks, garment_type
        )
        
        if src_pts is None or dst_pts is None or len(src_pts) < 3:
            # Not enough control points, fallback
            return warp_image_perspective(frame, garment_rgba, landmarks)
        
        # Apply TPS warping
        output_shape = (frame.shape[0], frame.shape[1])
        warped = _tps_warper.warp(
            garment_rgba,
            src_pts,
            dst_pts,
            output_shape=output_shape
        )
        
        return warped
        
    except Exception as e:
        # If TPS fails for any reason, fallback gracefully
        print(f"[WARP] TPS failed: {e}, falling back to perspective")
        return warp_image_perspective(frame, garment_rgba, landmarks)


def warp_image(frame, garment_rgba, landmarks, garment_type='shirt_front'):
    """
    Main warping function with mode switching.
    
    Dispatches to appropriate warp method based on current mode.
    Maintains backward compatibility with original perspective warp.
    
    Args:
        frame: Video frame for size reference
        garment_rgba: RGBA garment image
        landmarks: Detected pose keypoints {idx: (x, y)}
        garment_type: 'shirt_front' or 'shirt_back' (for TPS mode)
    
    Returns:
        Warped garment image ready for overlay
    """
    global _warp_mode
    
    if _warp_mode == 'tps':
        return warp_image_tps(frame, garment_rgba, landmarks, garment_type)
    elif _warp_mode == 'affine':
        return warp_image_affine(frame, garment_rgba, landmarks)
    else:  # perspective (default)
        return warp_image_perspective(frame, garment_rgba, landmarks)


def reset_tps_cache():
    """Reset TPS warper cache. Call when switching garments or major scene change."""
    global _tps_warper
    if _tps_warper is not None:
        _tps_warper.reset_cache()
        print("[WARP] TPS cache reset")