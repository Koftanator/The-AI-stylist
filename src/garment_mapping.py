"""
Garment Control Point Mapping System

Defines how garment template control points map to COCO pose keypoints.
This module handles the critical task of establishing correspondence between
pre-segmented garment images and detected human pose landmarks.

COCO 17-Point Keypoint Format (YOLOv8-pose):
    0: nose          1: left_eye       2: right_eye
    3: left_ear      4: right_ear
    5: left_shoulder 6: right_shoulder
    7: left_elbow    8: right_elbow
    9: left_wrist    10: right_wrist
    11: left_hip     12: right_hip
    13: left_knee    14: right_knee
    15: left_ankle   16: right_ankle
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import cv2


# COCO keypoint indices for reference
COCO_KEYPOINTS = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16
}


class GarmentMapping:
    """
    Defines control point mappings for different garment types.
    
    Each garment type has a set of normalized control points (in garment space, 0-1)
    that map to specific COCO pose keypoints (in body space, pixel coordinates).
    """
    
    @staticmethod
    def get_shirt_front_mapping() -> Dict[str, Any]:
        """
        Control point mapping for front-view shirt/top garment.
        
        Strategy for shirt fitting:
        - Shoulders define top width
        - Collar area maps to neck (interpolated from shoulders)
        - Sides follow torso contour (shoulders -> hips)
        - Bottom hem maps to hip line
        - Sleeves extend to elbows (optionally wrists for long sleeves)
        
        Returns:
            Dictionary with 'control_points_normalized' and 'keypoint_mapping'
        """
        # Define control points on the garment template (normalized 0-1 coordinates)
        # These are strategic points on a flat garment image that will be warped
        # Format: (x_norm, y_norm) where 0,0 is top-left, 1,1 is bottom-right
        # NOTE: These coordinates are auto-detected from the actual garment image structure
        
        control_points_normalized = {
            # Collar/neck region (top ~9% of garment)
            'neck_center': (0.475, 0.086),
            'neck_left': (0.195, 0.086),
            'neck_right': (0.756, 0.086),
            
            # Shoulder line (at ~26% height - widest point in upper region)
            'shoulder_left': (0.071, 0.255),
            'shoulder_right': (0.873, 0.255),
            
            # Chest area (~35% height - for better fit)
            'chest_left': (0.034, 0.349),
            'chest_right': (0.916, 0.349),
            'chest_center': (0.475, 0.349),  # Added center for better chest fit
            
            # Mid-torso (~47% height - for smoother waist transition)
            'mid_torso_left': (0.120, 0.470),
            'mid_torso_right': (0.830, 0.470),
            
            # Waist/torso sides (~59% height)
            'waist_left': (0.191, 0.585),
            'waist_right': (0.761, 0.585),
            'waist_center': (0.475, 0.585),  # Added center for better waist fit
            
            # Lower torso (~72% height - for better hip transition)
            'lower_torso_left': (0.186, 0.720),
            'lower_torso_right': (0.761, 0.720),
            
            # Hip/bottom hem (~89% height)
            'hip_left': (0.182, 0.885),
            'hip_center': (0.471, 0.885),
            'hip_right': (0.761, 0.885),
            
            # Sleeve control points (for better arm fitting)
            'sleeve_left_mid': (0.071, 0.303),
            'sleeve_right_mid': (0.873, 0.303),
            'sleeve_left_end': (0.006, 0.416),
            'sleeve_right_end': (0.994, 0.416),
            
            # Underarm points for better sleeve-to-body transition
            'underarm_left': (0.150, 0.380),
            'underarm_right': (0.850, 0.380),
        }
        
        # Map each garment control point to COCO keypoint(s) or interpolation
        # Format: keypoint_index or ['interpolate', kp1, kp2, weight]
        keypoint_mapping = {
            # Neck maps to midpoint between shoulders with offset toward head
            'neck_center': ['interpolate', 'left_shoulder', 'right_shoulder', 0.5, 'nose', -0.3],
            'neck_left': ['interpolate', 'left_shoulder', 'nose', 0.8],
            'neck_right': ['interpolate', 'right_shoulder', 'nose', 0.8],
            
            # Shoulders map directly
            'shoulder_left': 'left_shoulder',
            'shoulder_right': 'right_shoulder',
            
            # Chest area interpolates between shoulders and hips
            'chest_left': ['interpolate', 'left_shoulder', 'left_hip', 0.3],
            'chest_right': ['interpolate', 'right_shoulder', 'right_hip', 0.3],
            'chest_center': ['interpolate', 'left_shoulder', 'right_shoulder', 0.5, 'left_hip', 0.3],
            
            # Mid-torso for smoother waist transition
            'mid_torso_left': ['interpolate', 'left_shoulder', 'left_hip', 0.47],
            'mid_torso_right': ['interpolate', 'right_shoulder', 'right_hip', 0.47],
            
            # Waist maps partway down torso
            'waist_left': ['interpolate', 'left_shoulder', 'left_hip', 0.65],
            'waist_right': ['interpolate', 'right_shoulder', 'right_hip', 0.65],
            'waist_center': ['interpolate', 'left_hip', 'right_hip', 0.5, 'left_shoulder', -0.35],
            
            # Lower torso for better hip transition
            'lower_torso_left': ['interpolate', 'left_shoulder', 'left_hip', 0.82],
            'lower_torso_right': ['interpolate', 'right_shoulder', 'right_hip', 0.82],
            
            # Hips map directly
            'hip_left': 'left_hip',
            'hip_center': ['interpolate', 'left_hip', 'right_hip', 0.5],
            'hip_right': 'right_hip',
            
            # Sleeves extend toward elbows (CRITICAL: using elbows, NOT wrists)
            'sleeve_left_mid': ['interpolate', 'left_shoulder', 'left_elbow', 0.5],
            'sleeve_right_mid': ['interpolate', 'right_shoulder', 'right_elbow', 0.5],
            'sleeve_left_end': 'left_elbow',
            'sleeve_right_end': 'right_elbow',
            
            # Underarm points for better sleeve-to-body transition
            'underarm_left': ['interpolate', 'left_shoulder', 'left_hip', 0.35],
            'underarm_right': ['interpolate', 'right_shoulder', 'right_hip', 0.35],
        }
        
        return {
            'control_points_normalized': control_points_normalized,
            'keypoint_mapping': keypoint_mapping,
            'description': 'Front-view shirt with short sleeves'
        }
    
    @staticmethod
    def get_shirt_back_mapping() -> Dict[str, Any]:
        """
        Control point mapping for back-view shirt/top garment.
        
        Similar to front but without neck detail (occluded).
        Enhanced with more control points for better fit.
        """
        control_points_normalized = {
            # Upper back
            'neck_center': (0.5, 0.08),
            
            # Shoulders
            'shoulder_left': (0.15, 0.15),
            'shoulder_right': (0.85, 0.15),
            
            # Upper mid-back
            'upper_back_left': (0.18, 0.3),
            'upper_back_right': (0.82, 0.3),
            'upper_back_center': (0.5, 0.3),
            
            # Mid-back
            'back_left': (0.2, 0.4),
            'back_right': (0.8, 0.4),
            'back_center': (0.5, 0.4),
            
            # Waist area
            'waist_left': (0.22, 0.55),
            'waist_right': (0.78, 0.55),
            
            # Lower back
            'lower_left': (0.25, 0.7),
            'lower_right': (0.75, 0.7),
            
            # Bottom hem
            'hip_left': (0.3, 0.95),
            'hip_center': (0.5, 0.95),
            'hip_right': (0.7, 0.95),
            
            # Sleeves (map to elbows, NOT wrists)
            'sleeve_left_mid': (0.05, 0.3),
            'sleeve_right_mid': (0.95, 0.3),
            'sleeve_left_end': (0.02, 0.42),
            'sleeve_right_end': (0.98, 0.42),
        }
        
        keypoint_mapping = {
            'neck_center': ['interpolate', 'left_shoulder', 'right_shoulder', 0.5],
            'shoulder_left': 'left_shoulder',
            'shoulder_right': 'right_shoulder',
            'upper_back_left': ['interpolate', 'left_shoulder', 'left_hip', 0.25],
            'upper_back_right': ['interpolate', 'right_shoulder', 'right_hip', 0.25],
            'upper_back_center': ['interpolate', 'left_shoulder', 'right_shoulder', 0.5, 'left_hip', 0.25],
            'back_left': ['interpolate', 'left_shoulder', 'left_hip', 0.35],
            'back_right': ['interpolate', 'right_shoulder', 'right_hip', 0.35],
            'back_center': ['interpolate', 'left_hip', 'right_hip', 0.5, 'left_shoulder', -0.35],
            'waist_left': ['interpolate', 'left_shoulder', 'left_hip', 0.55],
            'waist_right': ['interpolate', 'right_shoulder', 'right_hip', 0.55],
            'lower_left': ['interpolate', 'left_shoulder', 'left_hip', 0.7],
            'lower_right': ['interpolate', 'right_shoulder', 'right_hip', 0.7],
            'hip_left': 'left_hip',
            'hip_center': ['interpolate', 'left_hip', 'right_hip', 0.5],
            'hip_right': 'right_hip',
            # CRITICAL: Sleeves map to ELBOWS, not wrists for stability
            'sleeve_left_mid': ['interpolate', 'left_shoulder', 'left_elbow', 0.5],
            'sleeve_right_mid': ['interpolate', 'right_shoulder', 'right_elbow', 0.5],
            'sleeve_left_end': 'left_elbow',
            'sleeve_right_end': 'right_elbow',
        }
        
        return {
            'control_points_normalized': control_points_normalized,
            'keypoint_mapping': keypoint_mapping,
            'description': 'Back-view shirt'
        }


def denormalize_points(
    normalized_pts: Dict[str, Tuple[float, float]],
    image_shape: Tuple[int, int]
) -> Dict[str, Tuple[int, int]]:
    """
    Convert normalized (0-1) garment coordinates to pixel coordinates.
    
    Args:
        normalized_pts: Dictionary of point_name -> (x_norm, y_norm)
        image_shape: (height, width) of garment image
        
    Returns:
        Dictionary of point_name -> (x_px, y_px)
    """
    h, w = image_shape
    denormalized = {}
    for name, (x_norm, y_norm) in normalized_pts.items():
        denormalized[name] = (int(x_norm * w), int(y_norm * h))
    return denormalized


def interpolate_keypoint(
    landmarks: Dict[int, Tuple[int, int]],
    mapping_spec: Any,
    keypoint_names: Dict[str, int] = COCO_KEYPOINTS
) -> Optional[Tuple[int, int]]:
    """
    Resolve a mapping specification to a pixel coordinate.
    
    Args:
        landmarks: Dictionary of keypoint_idx -> (x, y) from pose detector
        mapping_spec: Either a keypoint name/index or interpolation spec
        keypoint_names: Name to index mapping
        
    Returns:
        (x, y) pixel coordinate or None if keypoints unavailable
    """
    # Direct keypoint reference
    if isinstance(mapping_spec, str) and mapping_spec in keypoint_names:
        idx = keypoint_names[mapping_spec]
        return landmarks.get(idx)
    elif isinstance(mapping_spec, int):
        return landmarks.get(mapping_spec)
    
    # Interpolation between keypoints
    elif isinstance(mapping_spec, list) and mapping_spec[0] == 'interpolate':
        if len(mapping_spec) == 4:
            # Simple linear interpolation: ['interpolate', kp1, kp2, weight]
            _, kp1, kp2, w = mapping_spec
            pt1 = interpolate_keypoint(landmarks, kp1, keypoint_names)
            pt2 = interpolate_keypoint(landmarks, kp2, keypoint_names)
            
            if pt1 is None or pt2 is None:
                return None
            
            x = int(pt1[0] * (1 - w) + pt2[0] * w)
            y = int(pt1[1] * (1 - w) + pt2[1] * w)
            return (x, y)
        
        elif len(mapping_spec) == 6:
            # Weighted interpolation with offset: ['interpolate', kp1, kp2, w1, kp3, w2]
            _, kp1, kp2, w1, kp3, w2 = mapping_spec
            pt1 = interpolate_keypoint(landmarks, kp1, keypoint_names)
            pt2 = interpolate_keypoint(landmarks, kp2, keypoint_names)
            pt3 = interpolate_keypoint(landmarks, kp3, keypoint_names)
            
            if any(pt is None for pt in [pt1, pt2, pt3]):
                return None
            
            # Type assertions to help type checker (we know these aren't None after the check)
            assert pt1 is not None and pt2 is not None and pt3 is not None
            
            # First interpolate between pt1 and pt2
            base_x = int(pt1[0] * (1 - w1) + pt2[0] * w1)
            base_y = int(pt1[1] * (1 - w1) + pt2[1] * w1)
            
            # Then offset toward pt3
            x = int(base_x + (pt3[0] - base_x) * w2)
            y = int(base_y + (pt3[1] - base_y) * w2)
            return (x, y)
    
    return None


def get_control_point_pairs(
    garment_image: np.ndarray,
    landmarks: Dict[int, Tuple[int, int]],
    garment_type: str = 'shirt_front'
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate matched control point pairs for TPS warping.
    
    Args:
        garment_image: RGBA garment image
        landmarks: Detected pose keypoints {idx: (x, y)}
        garment_type: Type of garment ('shirt_front', 'shirt_back', etc.)
        
    Returns:
        (src_points, dst_points) as numpy arrays (N, 2), or (None, None) if insufficient data
    """
    # Get appropriate mapping for garment type
    if garment_type == 'shirt_front':
        mapping_data = GarmentMapping.get_shirt_front_mapping()
    elif garment_type == 'shirt_back':
        mapping_data = GarmentMapping.get_shirt_back_mapping()
    else:
        raise ValueError(f"Unknown garment type: {garment_type}")
    
    control_pts_norm = mapping_data['control_points_normalized']
    keypoint_map = mapping_data['keypoint_mapping']
    
    # Convert normalized garment points to pixels
    garment_h, garment_w = garment_image.shape[:2]
    garment_pts_px = denormalize_points(
        control_pts_norm,
        (garment_h, garment_w)
    )
    
    # Build matched pairs
    src_points = []
    dst_points = []
    
    for pt_name, garment_coord in garment_pts_px.items():
        # Look up corresponding body keypoint
        mapping_spec = keypoint_map.get(pt_name)
        if mapping_spec is None:
            continue
        
        body_coord = interpolate_keypoint(landmarks, mapping_spec)
        if body_coord is None:
            continue  # Skip if keypoint not detected
        
        src_points.append(garment_coord)
        dst_points.append(body_coord)
    
    # Need minimum points for TPS
    if len(src_points) < 3:
        return None, None
    
    return np.array(src_points, dtype=np.float32), np.array(dst_points, dtype=np.float32)


def visualize_control_points(
    image: np.ndarray,
    points: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 5
) -> np.ndarray:
    """
    Draw control points on image for debugging/visualization.
    
    Args:
        image: Input image
        points: Array of (x, y) points
        color: BGR color for points
        radius: Circle radius
        
    Returns:
        Image with points drawn
    """
    vis = image.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(vis, (int(x), int(y)), radius, color, -1)
        cv2.putText(
            vis, str(i), (int(x) + 7, int(y) - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )
    return vis


def check_keypoint_stability(
    landmarks: Dict[int, Tuple[int, int]],
    required_keypoints: List[int] = [5, 6, 11, 12]  # shoulders and hips
) -> Tuple[bool, float]:
    """
    Check if essential keypoints are detected with good quality.
    
    Args:
        landmarks: Detected keypoints
        required_keypoints: List of essential keypoint indices
        
    Returns:
        (is_stable, confidence) tuple
    """
    detected = [landmarks.get(idx) for idx in required_keypoints]
    detected_count = sum(1 for pt in detected if pt is not None)
    confidence = detected_count / len(required_keypoints)
    
    is_stable = confidence >= 0.75  # Need at least 75% of key points
    
    return is_stable, confidence
