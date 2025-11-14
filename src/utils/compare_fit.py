#!/usr/bin/env python3
"""
Side-by-Side Fit Comparison Tool

Shows multiple views simultaneously:
1. Original frame with keypoints
2. Warped garment result
3. Control points overlay
4. Mask visualization

This helps identify exactly where adjustments are needed.
"""

import cv2
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from pose_yolo import get_landmarks
from garment_mapping import get_control_point_pairs, GarmentMapping, denormalize_points
from tps_warp import TPSWarper
from overlay import create_body_mask, overlay
from overlay_skeleton import draw_skeleton


def create_grid_layout(frames, labels, grid_size=(2, 2)):
    """Create a grid layout of multiple frames."""
    rows, cols = grid_size
    
    # Ensure all frames are same size
    h, w = frames[0].shape[:2]
    resized = []
    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        resized.append(frame)
    
    # Create rows
    grid_rows = []
    for i in range(rows):
        row_frames = []
        for j in range(cols):
            idx = i * cols + j
            if idx < len(resized):
                frame = resized[idx].copy()
                # Add label
                if idx < len(labels):
                    cv2.putText(frame, labels[idx], (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                row_frames.append(frame)
            else:
                # Empty frame
                row_frames.append(np.zeros((h, w, 3), dtype=np.uint8))
        
        # Concatenate horizontally
        grid_rows.append(np.hstack(row_frames))
    
    # Concatenate vertically
    return np.vstack(grid_rows)


def draw_control_point_analysis(frame, landmarks, src_pts, dst_pts, garment_shape):
    """Draw detailed control point analysis."""
    vis = frame.copy()
    
    # Draw skeleton
    vis = draw_skeleton(vis, landmarks)
    
    # Draw all destination points with connections
    if len(dst_pts) > 0:
        # Draw lines between consecutive points to show structure
        for i in range(len(dst_pts) - 1):
            pt1 = tuple(dst_pts[i].astype(int))
            pt2 = tuple(dst_pts[i + 1].astype(int))
            cv2.line(vis, pt1, pt2, (100, 100, 255), 1)
        
        # Draw points
        for i, (src, dst) in enumerate(zip(src_pts, dst_pts)):
            # Calculate expected distance from garment center
            garment_h, garment_w = garment_shape[:2]
            src_center = np.array([garment_w / 2, garment_h / 2])
            src_distance = np.linalg.norm(src - src_center)
            
            # Color code by distance (helps identify outer vs inner points)
            if src_distance < garment_w * 0.2:
                color = (0, 255, 0)  # Green = center points
            elif src_distance < garment_w * 0.4:
                color = (0, 255, 255)  # Yellow = mid points
            else:
                color = (0, 0, 255)  # Red = edge points
            
            # Draw destination point
            cv2.circle(vis, (int(dst[0]), int(dst[1])), 10, color, -1)
            cv2.circle(vis, (int(dst[0]), int(dst[1])), 12, (255, 255, 255), 2)
            cv2.putText(vis, str(i), (int(dst[0]) + 15, int(dst[1]) + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis


def draw_garment_with_points(garment_rgba, src_pts):
    """Draw garment with source control points marked."""
    if garment_rgba.shape[2] == 4:
        # Convert RGBA to RGB on white background
        rgb = garment_rgba[:, :, :3]
        alpha = garment_rgba[:, :, 3:4] / 255.0
        white_bg = np.ones_like(rgb) * 255
        vis = (alpha * rgb + (1 - alpha) * white_bg).astype(np.uint8)
    else:
        vis = garment_rgba.copy()
    
    # Draw source points
    for i, src in enumerate(src_pts):
        cv2.circle(vis, (int(src[0]), int(src[1])), 8, (0, 255, 0), -1)
        cv2.circle(vis, (int(src[0]), int(src[1])), 10, (255, 255, 255), 2)
        cv2.putText(vis, str(i), (int(src[0]) + 12, int(src[1])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    return vis


def main():
    print("\n" + "="*70)
    print("Side-by-Side Fit Comparison Tool")
    print("="*70)
    print("Shows 4 views simultaneously:")
    print("  1. Top-left: Original frame with skeleton")
    print("  2. Top-right: Final warped result")
    print("  3. Bottom-left: Control points analysis")
    print("  4. Bottom-right: Body mask visualization")
    print("\nPress 'q' to quit")
    print("="*70 + "\n")
    
    # Setup
    project_root = os.path.dirname(script_dir)
    tps_warper = TPSWarper(use_gpu=True, downsample_factor=0.5)
    
    # Load garment
    garment_path = os.path.join(project_root, 'assests', 'garments', 'front_seg.png')
    garment_rgba = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
    
    if garment_rgba is None:
        print(f"❌ Cannot load garment from {garment_path}")
        return
    
    print(f"✓ Garment loaded: {garment_rgba.shape}")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✓ Camera opened")
    print("\nProcessing frames...\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Detect pose
        landmarks = get_landmarks(frame)
        
        if landmarks:
            # Get control points
            src_pts, dst_pts = get_control_point_pairs(
                garment_rgba,
                landmarks,
                'shirt_front'
            )
            
            if src_pts is not None and dst_pts is not None:
                # 1. Original with skeleton
                view1 = draw_skeleton(frame.copy(), landmarks)
                
                # 2. Warped result
                warped = tps_warper.warp(
                    garment_rgba,
                    src_pts,
                    dst_pts,
                    frame.shape[:2]
                )
                view2 = overlay(frame, warped, landmarks, mask_type='body')
                
                # 3. Control points analysis
                view3 = draw_control_point_analysis(frame, landmarks, src_pts, dst_pts, garment_rgba.shape)
                
                # 4. Mask visualization
                mask = create_body_mask(frame.shape, landmarks, padding=40)
                mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                view4 = frame.copy()
                cv2.addWeighted(mask_colored, 0.5, view4, 0.5, 0, view4)
                
                # Create grid
                grid = create_grid_layout(
                    [view1, view2, view3, view4],
                    ['1. Skeleton', '2. Result', '3. Control Points', '4. Mask'],
                    grid_size=(2, 2)
                )
                
                # Add overall info
                cv2.putText(grid, f"Frame: {frame_count} | Points: {len(src_pts)} | Press 'q' to quit",
                           (10, grid.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Fit Comparison', grid)
            else:
                # No control points
                msg = "Insufficient keypoints detected"
                cv2.putText(frame, msg, (10, frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow('Fit Comparison', frame)
        else:
            # No pose detected
            msg = "No pose detected - stand in front of camera"
            cv2.putText(frame, msg, (10, frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow('Fit Comparison', frame)
        
        # Handle input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Processed {frame_count} frames")
    print("Closed\n")


if __name__ == '__main__':
    main()
