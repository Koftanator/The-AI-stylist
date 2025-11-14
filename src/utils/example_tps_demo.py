#!/usr/bin/env python3
"""
Quick Start Example: TPS Warping Demo

Demonstrates TPS warping with a single test frame.
Useful for testing and understanding control point mapping.

Usage:
    python example_tps_demo.py
"""

import cv2
import numpy as np
import os
from typing import Dict, Tuple

# Import modules
try:
    from pose_yolo import get_landmarks
    print('[DEMO] Using YOLOv8 pose backend')
except Exception:
    from pose import get_landmarks
    print('[DEMO] Using MoveNet pose backend')

from tps_warp import create_tps_warper
from garment_mapping import (
    get_control_point_pairs,
    visualize_control_points,
    GarmentMapping
)
from overlay import overlay
from overlay_skeleton import draw_skeleton


def visualize_mapping(
    frame: np.ndarray,
    garment: np.ndarray,
    landmarks: Dict[int, Tuple[int, int]],
    garment_type: str = 'shirt_front'
):
    """
    Create visualization showing:
    1. Original frame with skeleton
    2. Garment with control points marked
    3. Body with destination points marked
    4. Final TPS-warped result
    """
    # Get control point pairs
    src_pts, dst_pts = get_control_point_pairs(garment, landmarks, garment_type)
    
    if src_pts is None or dst_pts is None:
        print("[DEMO] Could not generate control points (insufficient landmarks)")
        return None
    
    print(f"[DEMO] Generated {len(src_pts)} control point pairs")
    
    # Panel 1: Skeleton overlay
    panel1 = draw_skeleton(frame.copy(), landmarks)
    cv2.putText(panel1, '1. Detected Pose', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Panel 2: Garment with source control points
    if garment.shape[2] == 4:
        # Convert RGBA to BGR for visualization
        garment_vis = cv2.cvtColor(garment, cv2.COLOR_BGRA2BGR)
    else:
        garment_vis = garment.copy()
    
    panel2 = visualize_control_points(garment_vis, src_pts, color=(255, 0, 0), radius=6)
    # Resize to match frame size for side-by-side comparison
    panel2 = cv2.resize(panel2, (frame.shape[1], frame.shape[0]))
    cv2.putText(panel2, '2. Garment Control Points', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Panel 3: Body with destination control points
    panel3 = frame.copy()
    panel3 = visualize_control_points(panel3, dst_pts, color=(0, 255, 0), radius=8)
    cv2.putText(panel3, '3. Target Body Points', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Panel 4: TPS warped result
    print("[DEMO] Applying TPS warp...")
    warper = create_tps_warper(fast_mode=True, gpu_enabled=True)
    warped = warper.warp(
        garment,
        src_pts,
        dst_pts,
        output_shape=(frame.shape[0], frame.shape[1])
    )
    
    panel4 = overlay(frame.copy(), warped)
    cv2.putText(panel4, '4. TPS Warped Result', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Combine into 2x2 grid
    top_row = np.hstack([panel1, panel2])
    bottom_row = np.hstack([panel3, panel4])
    combined = np.vstack([top_row, bottom_row])
    
    # Resize for display if too large
    max_width = 1920
    max_height = 1080
    if combined.shape[1] > max_width or combined.shape[0] > max_height:
        scale = min(max_width / combined.shape[1], max_height / combined.shape[0])
        new_w = int(combined.shape[1] * scale)
        new_h = int(combined.shape[0] * scale)
        combined = cv2.resize(combined, (new_w, new_h))
    
    return combined


def demo_live():
    """Live webcam demo with step-by-step visualization."""
    print("\n" + "="*60)
    print("TPS WARPING DEMO - Live Webcam")
    print("="*60)
    print("Controls:")
    print("  SPACE: Pause and show detailed visualization")
    print("  q: Quit")
    print("="*60 + "\n")
    
    # Load garment with absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    garment_path = os.path.join(project_root, 'assests', 'garments', 'front_seg.png')
    
    garment = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
    if garment is None:
        print(f"ERROR: Could not load garment from {garment_path}")
        return
    
    cap = cv2.VideoCapture(0)
    paused = False
    saved_frame = None
    saved_landmarks = None
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = get_landmarks(frame)
            
            if landmarks:
                # Quick preview: just show warped result
                src_pts, dst_pts = get_control_point_pairs(
                    garment, landmarks, 'shirt_front'
                )
                
                if src_pts is not None and dst_pts is not None:
                    warper = create_tps_warper(fast_mode=True, gpu_enabled=True)
                    warped = warper.warp(
                        garment, src_pts, dst_pts,
                        output_shape=(frame.shape[0], frame.shape[1])
                    )
                    output = overlay(frame, warped)
                else:
                    output = frame
                
                cv2.putText(
                    output, 'Press SPACE to see details', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
                )
            else:
                output = frame
                cv2.putText(
                    output, 'No pose detected', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )
            
            cv2.imshow('TPS Demo - Press SPACE for details', output)
            saved_frame = frame.copy()
            saved_landmarks = landmarks
        
        else:
            # Show detailed visualization
            if saved_frame is not None and saved_landmarks is not None:
                vis = visualize_mapping(saved_frame, garment, saved_landmarks)
                if vis is not None:
                    cv2.imshow('TPS Demo - Detailed View (press any key to resume)', vis)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            if not paused and saved_landmarks:
                paused = True
                print("[DEMO] Paused - generating detailed visualization...")
            else:
                paused = False
                cv2.destroyWindow('TPS Demo - Detailed View (press any key to resume)')
                print("[DEMO] Resumed")
    
    cap.release()
    cv2.destroyAllWindows()


def demo_single_frame():
    """Capture and analyze a single frame."""
    print("\n" + "="*60)
    print("TPS WARPING DEMO - Single Frame Analysis")
    print("="*60 + "\n")
    
    # Load garment with absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    garment_path = os.path.join(project_root, 'assests', 'garments', 'front_seg.png')
    
    garment = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
    if garment is None:
        print(f"ERROR: Could not load garment from {garment_path}")
        return
    
    cap = cv2.VideoCapture(0)
    
    print("[DEMO] Position yourself in frame and press SPACE to capture...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.putText(
            frame, 'Press SPACE to capture', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow('Capture Frame', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            print("[DEMO] Frame captured! Processing...")
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    
    cap.release()
    cv2.destroyWindow('Capture Frame')
    
    # Process captured frame
    landmarks = get_landmarks(frame)
    
    if not landmarks:
        print("[DEMO] No pose detected in captured frame!")
        cv2.imshow('No Pose Detected', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    
    print(f"[DEMO] Detected {len([p for p in landmarks.values() if p is not None])} keypoints")
    
    # Generate visualization
    vis = visualize_mapping(frame, garment, landmarks)
    
    if vis is not None:
        print("[DEMO] Visualization complete!")
        print("        Top-left: Detected pose skeleton")
        print("        Top-right: Garment with source control points")
        print("        Bottom-left: Body with target control points")
        print("        Bottom-right: Final TPS-warped result")
        print("\nPress any key to close...")
        
        cv2.imshow('TPS Warping Analysis', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("[DEMO] Could not generate visualization")


def main():
    """Main entry point."""
    import sys
    
    print("\n" + "="*60)
    print("TPS WARPING DEMONSTRATION")
    print("="*60)
    print("\nChoose mode:")
    print("  1. Live demo (continuous warping)")
    print("  2. Single frame analysis (detailed visualization)")
    print("="*60)
    
    try:
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == '1':
            demo_live()
        elif choice == '2':
            demo_single_frame()
        else:
            print("Invalid choice")
    except KeyboardInterrupt:
        print("\n[DEMO] Interrupted")
    except Exception as e:
        print(f"[DEMO] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
