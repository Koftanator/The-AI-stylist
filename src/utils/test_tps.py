#!/usr/bin/env python3
"""
Quick TPS Test - Verify TPS warping is producing visible output
"""

import cv2
import numpy as np
import os
from warp import set_warp_mode, warp_image, get_warp_mode
from pose_yolo import get_landmarks
from overlay import overlay
from overlay_skeleton import draw_skeleton

print("="*60)
print("TPS VERIFICATION TEST")
print("="*60)

# Load garment with absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
garment_path = os.path.join(project_root, 'assests', 'garments', 'front_seg.png')

garment = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
if garment is None:
    print(f"ERROR: Could not load garment from {garment_path}")
    exit(1)

print(f"✓ Garment loaded: {garment.shape}")

# Set TPS mode
set_warp_mode('tps')
print(f"✓ Warp mode set to: {get_warp_mode()}")

# Capture frame
cap = cv2.VideoCapture(0)
print("✓ Camera opened")
print("\nPosition yourself in frame, press SPACE to test TPS...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame")
        break
    
    # Show preview
    preview = frame.copy()
    cv2.putText(preview, 'Press SPACE to test TPS', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Preview', preview)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit(0)

cap.release()
cv2.destroyWindow('Preview')

print("\n" + "="*60)
print("TESTING TPS WARPING...")
print("="*60)

# Get landmarks
landmarks = get_landmarks(frame)
if not landmarks:
    print("ERROR: No pose detected!")
    cv2.imshow('Failed - No Pose', frame)
    cv2.waitKey(0)
    exit(1)

detected = sum(1 for v in landmarks.values() if v is not None)
print(f"✓ Pose detected: {detected}/17 keypoints")

# Draw skeleton
display = draw_skeleton(frame.copy(), landmarks)

# Test each mode
modes_to_test = ['perspective', 'affine', 'tps']
results = {}

for mode in modes_to_test:
    print(f"\nTesting {mode.upper()} mode...")
    set_warp_mode(mode)
    
    warped = warp_image(frame, garment, landmarks, 'shirt_front')
    
    if warped is None:
        print(f"  ✗ {mode}: Returned None")
        continue
    
    # Check for actual content
    if warped.shape[2] == 4:
        non_zero = np.count_nonzero(warped[:,:,3] > 0)
    else:
        non_zero = np.count_nonzero(warped)
    
    print(f"  ✓ {mode}: {non_zero:,} non-zero pixels")
    
    # Create visualization
    output = overlay(display.copy(), warped)
    
    # Add labels
    cv2.putText(output, f'{mode.upper()} Mode', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(output, f'Non-zero pixels: {non_zero:,}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    results[mode] = output

# Show all results
print("\n" + "="*60)
print("RESULTS - Press number key to switch modes:")
print("  1 = Perspective")
print("  2 = Affine")
print("  3 = TPS")
print("  q = Quit")
print("="*60)

current_mode = 'tps'
while True:
    if current_mode in results:
        cv2.imshow('TPS Test Results', results[current_mode])
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        current_mode = 'perspective'
        print(f"\nSwitched to: {current_mode.upper()}")
    elif key == ord('2'):
        current_mode = 'affine'
        print(f"\nSwitched to: {current_mode.upper()}")
    elif key == ord('3'):
        current_mode = 'tps'
        print(f"\nSwitched to: {current_mode.upper()}")

cv2.destroyAllWindows()
print("\nTest complete!")
