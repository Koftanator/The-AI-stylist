#!/usr/bin/env python3
"""
Debug TPS Warping - Visual Inspection Tool

This script helps visualize:
1. Original garment with control points overlaid
2. Detected body keypoints
3. Source -> Destination control point mapping
4. Final warped result

Press 's' to save debug images.
"""

import cv2
import numpy as np
import os
import sys

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from pose_yolo import get_landmarks
from garment_mapping import GarmentMapping, denormalize_points, get_control_point_pairs
from tps_warp import TPSWarper


def draw_control_points_on_garment(garment_img, control_points_px, title="Garment Control Points"):
    """Draw control points on garment image."""
    vis = garment_img.copy()
    if vis.shape[2] == 4:
        vis = cv2.cvtColor(vis, cv2.COLOR_BGRA2BGR)
    
    # Draw points
    for i, (name, (x, y)) in enumerate(control_points_px.items()):
        color = (0, 255, 0)
        cv2.circle(vis, (int(x), int(y)), 8, color, -1)
        cv2.circle(vis, (int(x), int(y)), 10, (255, 255, 255), 2)
        # Label
        cv2.putText(vis, str(i), (int(x) + 12, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis, name[:8], (int(x) + 12, int(y) + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    
    # Add title
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return vis


def draw_keypoints_with_mapping(frame, landmarks, dst_points, src_names, title="Body Keypoints"):
    """Draw body keypoints with mapped control points."""
    vis = frame.copy()
    
    # Draw all landmarks first
    for idx, (x, y) in landmarks.items():
        cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 255), -1)
        cv2.putText(vis, str(idx), (int(x) + 8, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw destination points for garment mapping
    for i, (pt, name) in enumerate(zip(dst_points, src_names)):
        x, y = pt
        cv2.circle(vis, (int(x), int(y)), 10, (0, 255, 0), 2)
        cv2.putText(vis, f"{i}:{name[:6]}", (int(x) + 12, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(vis, f"{len(dst_points)} control points", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis


def draw_correspondences(garment_img, frame, src_points, dst_points, src_names):
    """Draw side-by-side visualization with correspondence lines."""
    h1, w1 = garment_img.shape[:2]
    h2, w2 = frame.shape[:2]
    
    # Resize to same height
    scale = h2 / h1
    new_w1 = int(w1 * scale)
    garment_resized = cv2.resize(garment_img, (new_w1, h2))
    
    if garment_resized.shape[2] == 4:
        garment_resized = cv2.cvtColor(garment_resized, cv2.COLOR_BGRA2BGR)
    
    # Create side-by-side canvas
    canvas = np.zeros((h2, new_w1 + w2, 3), dtype=np.uint8)
    canvas[:, :new_w1] = garment_resized
    canvas[:, new_w1:] = frame
    
    # Draw correspondence lines
    for i, (src, dst, name) in enumerate(zip(src_points, dst_points, src_names)):
        # Scale source point
        src_scaled = (int(src[0] * scale), int(src[1] * scale))
        dst_offset = (int(dst[0]) + new_w1, int(dst[1]))
        
        # Color by index
        color = tuple([int(c) for c in cv2.applyColorMap(np.array([[i * 15]], dtype=np.uint8), cv2.COLORMAP_HSV)[0, 0]])
        
        # Draw line
        cv2.line(canvas, src_scaled, dst_offset, color, 2)
        cv2.circle(canvas, src_scaled, 6, color, -1)
        cv2.circle(canvas, dst_offset, 6, color, -1)
    
    cv2.putText(canvas, "Garment -> Body Mapping", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return canvas


def main():
    print("\n" + "="*60)
    print("TPS Warping Debug Visualization")
    print("="*60)
    print("Controls:")
    print("  q: Quit")
    print("  s: Save debug images")
    print("  SPACE: Pause/Resume")
    print("="*60 + "\n")
    
    # Load garment
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    garment_path = os.path.join(project_root, 'assests', 'garments', 'front_seg.png')
    garment_rgba = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
    
    if garment_rgba is None:
        print(f"ERROR: Cannot load garment from {garment_path}")
        return
    
    print(f"Garment loaded: {garment_rgba.shape}")
    
    # Get garment mapping
    mapping = GarmentMapping.get_shirt_front_mapping()
    control_pts_norm = mapping['control_points_normalized']
    
    # Denormalize to pixel coordinates
    garment_h, garment_w = garment_rgba.shape[:2]
    control_pts_px = denormalize_points(control_pts_norm, (garment_h, garment_w))
    
    print(f"Control points: {len(control_pts_px)}")
    
    # Draw control points on garment
    garment_vis = draw_control_points_on_garment(garment_rgba, control_pts_px)
    cv2.imshow('1. Garment Control Points', garment_vis)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize TPS warper
    tps_warper = TPSWarper(use_gpu=True, downsample_factor=0.5)
    
    paused = False
    frame_cache = None
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            frame_cache = frame.copy()
        else:
            frame = frame_cache
        
        # Detect pose
        landmarks = get_landmarks(frame)
        
        if landmarks:
            # Get control point pairs
            src_points, dst_points = get_control_point_pairs(
                garment_rgba,
                landmarks,
                'shirt_front'
            )
            
            if src_points is not None and dst_points is not None:
                # Get source names for visualization
                mapping = GarmentMapping.get_shirt_front_mapping()
                src_names = list(mapping['control_points_normalized'].keys())[:len(src_points)]
                
                print(f"\r[DEBUG] Control points: {len(src_points)} | Landmarks: {len(landmarks)}", end='', flush=True)
                
                # Visualize body keypoints
                body_vis = draw_keypoints_with_mapping(frame, landmarks, dst_points, src_names)
                cv2.imshow('2. Body Keypoints + Mapping', body_vis)
                
                # Visualize correspondences
                corr_vis = draw_correspondences(garment_rgba, frame, src_points, dst_points, src_names)
                cv2.imshow('3. Correspondences', corr_vis)
                
                # Perform TPS warp
                warped = tps_warper.warp(garment_rgba, src_points, dst_points, frame.shape[:2])
                
                # Show warped result
                if warped is not None:
                    warped_vis = warped.copy()
                    if warped_vis.shape[2] == 4:
                        # Show alpha channel as overlay
                        alpha = warped_vis[:, :, 3:4] / 255.0
                        bg = frame.copy()
                        for c in range(3):
                            bg[:, :, c] = (1 - alpha[:, :, 0]) * bg[:, :, c] + alpha[:, :, 0] * warped_vis[:, :, c]
                        cv2.imshow('4. Warped Result', bg.astype(np.uint8))
                    else:
                        cv2.imshow('4. Warped Result', warped_vis)
            else:
                info = frame.copy()
                cv2.putText(info, 'Insufficient control points for TPS', 
                           (10, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('2. Body Keypoints + Mapping', info)
        else:
            info = frame.copy()
            cv2.putText(info, 'No pose detected - stand in front of camera', 
                       (10, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('2. Body Keypoints + Mapping', info)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save debug images
            if landmarks:
                output_dir = os.path.join(project_root, 'debug_output')
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(os.path.join(output_dir, '1_garment_control_points.png'), garment_vis)
                cv2.imwrite(os.path.join(output_dir, '2_body_keypoints.png'), body_vis)
                cv2.imwrite(os.path.join(output_dir, '3_correspondences.png'), corr_vis)
                if warped is not None:
                    cv2.imwrite(os.path.join(output_dir, '4_warped_result.png'), bg.astype(np.uint8))
                print(f"\n[SAVED] Debug images to {output_dir}/")
        elif key == ord(' '):
            paused = not paused
            print(f"\n[{'PAUSED' if paused else 'RESUMED'}]")
    
    print("\n")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
