import cv2
import numpy as np


def create_body_mask(frame_shape, landmarks, padding=50):
    """
    Create a mask for BODY mode - shoulders, elbows, wrists, hips.
    This covers upper body + arms for full shirt/jacket fitting.
    
    Args:
        frame_shape: (height, width) of the frame
        landmarks: Dict of detected keypoints {idx: (x, y)}
        padding: Extra pixels around the body polygon
        
    Returns:
        Binary mask (0-255) covering the body region
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # BODY MODE: Use shoulders, elbows, wrists, hips
    # This creates a mask covering upper body + arms
    body_keypoints = [
        5,   # left shoulder
        7,   # left elbow
        9,   # left wrist
        11,  # left hip
        12,  # right hip
        10,  # right wrist
        8,   # right elbow
        6,   # right shoulder
    ]
    
    # Collect valid points
    body_points = []
    for idx in body_keypoints:
        pt = landmarks.get(idx)
        if pt is not None:
            body_points.append(pt)
    
    if len(body_points) < 3:
        # Not enough points, return full mask
        return np.ones((h, w), dtype=np.uint8) * 255
    
    # Create convex hull around body points
    body_points = np.array(body_points, dtype=np.int32)
    hull = cv2.convexHull(body_points)
    
    # Expand the hull slightly for padding
    M = cv2.moments(hull)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Expand points away from centroid
        expanded_hull = []
        for point in hull:
            # Extract coordinates from hull point
            pt = point[0]  # type: ignore[index]
            px, py = int(pt[0]), int(pt[1])
            vx, vy = px - cx, py - cy
            length = np.sqrt(vx*vx + vy*vy)
            if length > 0:
                vx = vx / length * padding
                vy = vy / length * padding
            new_x = int(px + vx)
            new_y = int(py + vy)
            expanded_hull.append([new_x, new_y])
        
        hull = np.array(expanded_hull, dtype=np.int32)
    
    # Fill the hull on mask
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Apply Gaussian blur for smooth edges
    mask = cv2.GaussianBlur(mask, (21, 21), 11)
    
    return mask


def create_torso_mask(frame_shape, landmarks, extend_factor=1.3):
    """
    Create TORSO mode mask - shoulders, hips, knees.
    More conservative than body mask - best for shirts that end at waist/hips.
    
    Args:
        frame_shape: (height, width) of the frame
        landmarks: Dict of detected keypoints
        extend_factor: How much to extend beyond torso bounds (1.0 = exact, >1.0 = larger)
        
    Returns:
        Binary mask covering torso region
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # TORSO MODE: Use shoulders, hips, knees
    # Key torso points
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    
    # Collect torso keypoints
    torso_keypoints = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        RIGHT_HIP,
        RIGHT_KNEE,
        LEFT_KNEE,
        LEFT_HIP,
    ]
    
    polygon_points = []
    for idx in torso_keypoints:
        pt = landmarks.get(idx)
        if pt is not None:
            polygon_points.append(list(pt))
    
    # Need at least shoulders and hips
    if len(polygon_points) < 4:
        # Fallback to body mask
        return create_body_mask(frame_shape, landmarks)
    
    polygon = np.array(polygon_points, dtype=np.int32)
    
    # Apply extension factor
    if extend_factor != 1.0:
        M = cv2.moments(polygon)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Scale polygon around centroid
            scaled = []
            for point in polygon:
                px, py = point
                vx = (px - cx) * extend_factor
                vy = (py - cy) * extend_factor
                scaled.append([int(cx + vx), int(cy + vy)])
            
            polygon = np.array(scaled, dtype=np.int32)
    
    # Fill polygon
    cv2.fillPoly(mask, [polygon], 255)
    
    # Smooth edges
    mask = cv2.GaussianBlur(mask, (15, 15), 8)
    
    return mask


def create_head_mask(frame_shape, landmarks, scale_factor=2.5):
    """
    Create HEAD mode mask - nose, eyes.
    Creates a circular/elliptical mask around the face for hats, glasses, masks, etc.
    
    Args:
        frame_shape: (height, width) of the frame
        landmarks: Dict of detected keypoints
        scale_factor: How much to expand the head region (2.0 = 2x head size)
        
    Returns:
        Binary mask covering head/face region
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # HEAD MODE: Use nose and eyes
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    
    nose_pt = landmarks.get(NOSE)
    left_eye_pt = landmarks.get(LEFT_EYE)
    right_eye_pt = landmarks.get(RIGHT_EYE)
    
    # Need at least 2 face points
    face_points = []
    if nose_pt is not None:
        face_points.append(nose_pt)
    if left_eye_pt is not None:
        face_points.append(left_eye_pt)
    if right_eye_pt is not None:
        face_points.append(right_eye_pt)
    
    if len(face_points) < 2:
        # Not enough face points, create small mask at nose if available
        if nose_pt is not None:
            cv2.circle(mask, (int(nose_pt[0]), int(nose_pt[1])), 100, 255, -1)
            mask = cv2.GaussianBlur(mask, (51, 51), 20)
            return mask
        else:
            # No face detected, return empty mask
            return mask
    
    # Calculate face center
    face_points = np.array(face_points)
    center_x = int(np.mean(face_points[:, 0]))
    center_y = int(np.mean(face_points[:, 1]))
    
    # Estimate head size from eye distance or nose-eye distance
    if left_eye_pt is not None and right_eye_pt is not None:
        # Use eye distance as base measurement
        eye_distance = np.sqrt((left_eye_pt[0] - right_eye_pt[0])**2 + 
                              (left_eye_pt[1] - right_eye_pt[1])**2)
        # Head is roughly 2.5x eye distance in width, 3x in height
        head_width = int(eye_distance * scale_factor)
        head_height = int(eye_distance * scale_factor * 1.3)
    else:
        # Fallback: use fixed size based on frame
        head_width = int(w * 0.15)
        head_height = int(h * 0.2)
    
    # Draw ellipse for head region
    cv2.ellipse(mask, (center_x, center_y), (head_width, head_height), 
                0, 0, 360, 255, -1)
    
    # Smooth edges heavily for natural look
    mask = cv2.GaussianBlur(mask, (51, 51), 20)
    
    return mask


def overlay(frame, warped_rgba, landmarks=None, mask_type='torso'):
    """
    Overlay warped garment on frame, constrained to body region.
    
    Args:
        frame: Background frame (BGR)
        warped_rgba: Warped garment with alpha channel
        landmarks: Detected skeleton keypoints (for masking)
        mask_type: 'body' (shoulders/elbows/wrists/hips), 
                   'torso' (shoulders/hips/knees), 
                   'head' (nose/eyes),
                   or 'none' (no mask)
        
    Returns:
        Frame with garment overlaid only on body region
    """
    if warped_rgba.shape[2] != 4:
        # No alpha channel, do simple overlay
        return frame
    
    # Create body mask if landmarks provided
    if landmarks is not None and mask_type != 'none':
        if mask_type == 'head':
            body_mask = create_head_mask(frame.shape, landmarks, scale_factor=2.5)
        elif mask_type == 'torso':
            body_mask = create_torso_mask(frame.shape, landmarks, extend_factor=1.2)
        else:  # 'body'
            body_mask = create_body_mask(frame.shape, landmarks, padding=40)
        
        # Combine garment alpha with body mask
        garment_alpha = warped_rgba[:, :, 3].astype(np.float32) / 255.0
        body_mask_alpha = body_mask.astype(np.float32) / 255.0
        
        # Final alpha is intersection of garment and body mask
        final_alpha = garment_alpha * body_mask_alpha
    else:
        # No masking
        final_alpha = warped_rgba[:, :, 3].astype(np.float32) / 255.0
    
    # Alpha blend
    output = frame.copy()
    for c in range(3):
        output[:, :, c] = (
            (1 - final_alpha) * frame[:, :, c] + 
            final_alpha * warped_rgba[:, :, c]
        )
    
    return output.astype(np.uint8)
