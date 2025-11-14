#!/usr/bin/env python3
"""
Quick test script to verify body masking functionality.
This visualizes the torso and body masks on a static frame.
"""

import cv2
import numpy as np
import os
import sys

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from pose_yolo import get_landmarks
from overlay import create_torso_mask, create_body_mask, create_head_mask


def visualize_mask(frame, landmarks, mask_type='body'):
    """Show the mask overlaid on the frame."""
    if mask_type == 'head':
        mask = create_head_mask(frame.shape, landmarks, scale_factor=2.5)
    elif mask_type == 'torso':
        mask = create_torso_mask(frame.shape, landmarks, extend_factor=1.2)
    else:  # 'body'
        mask = create_body_mask(frame.shape, landmarks, padding=40)
    
    # Create colored overlay
    overlay = frame.copy()
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    
    # Blend mask with frame
    alpha = 0.5
    cv2.addWeighted(mask_colored, alpha, overlay, 1 - alpha, 0, overlay)
    
    # Draw keypoints
    for idx, (x, y) in landmarks.items():
        cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(overlay, str(idx), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return overlay


def main():
    print("\n" + "="*60)
    print("Body Mask Visualization Test")
    print("="*60)
    print("Controls:")
    print("  t: Toggle mask mode (Body -> Torso -> Head)")
    print("  q: Quit")
    print("="*60)
    print("Mask Modes:")
    print("  Body:  Shoulders, Elbows, Wrists, Hips")
    print("  Torso: Shoulders, Hips, Knees")
    print("  Head:  Nose, Eyes")
    print("="*60 + "\n")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    mask_modes = ['body', 'torso', 'head']
    mask_idx = 0
    mask_type = mask_modes[mask_idx]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Detect pose
        landmarks = get_landmarks(frame)
        
        if landmarks:
            # Visualize mask
            output = visualize_mask(frame, landmarks, mask_type)
            
            # Add text
            cv2.putText(
                output, f'Mask Type: {mask_type.upper()}', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )
        else:
            output = frame.copy()
            cv2.putText(
                output, 'No pose detected', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
            )
        
        cv2.imshow('Body Mask Test', output)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            mask_idx = (mask_idx + 1) % len(mask_modes)
            mask_type = mask_modes[mask_idx]
            print(f"Switched to {mask_type} mask")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")


if __name__ == '__main__':
    main()
