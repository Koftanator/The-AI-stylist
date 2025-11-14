#!/usr/bin/env python3
"""
Interactive TPS Tuning Tool

This tool allows you to:
1. See the current TPS warping result in real-time
2. Adjust control point positions interactively
3. See before/after comparisons
4. Save optimized control point configurations

Use this to make the shirt wrap realistically on your body.
"""

import cv2
import numpy as np
import os
import sys
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from pose_yolo import get_landmarks
from garment_mapping import GarmentMapping, denormalize_points, get_control_point_pairs
from tps_warp import TPSWarper
from overlay import create_body_mask, overlay
from overlay_skeleton import draw_skeleton


class TPSMonitor:
    """Monitor and analyze TPS warping quality."""
    
    def __init__(self):
        self.project_root = os.path.dirname(script_dir)
        self.tps_warper = TPSWarper(use_gpu=True, downsample_factor=0.5)
        
        # Load garment
        garment_path = os.path.join(self.project_root, 'assests', 'garments', 'front_seg.png')
        self.garment_rgba = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
        
        if self.garment_rgba is None:
            raise FileNotFoundError(f"Cannot load garment from {garment_path}")
        
        print(f"[Monitor] Garment loaded: {self.garment_rgba.shape}")
        
        # Display options
        self.show_keypoints = True
        self.show_control_points = True
        self.show_mask = False
        self.show_original_garment = False
        
    def analyze_fit(self, frame, landmarks):
        """Analyze how well the garment fits."""
        # Get control points
        src_pts, dst_pts = get_control_point_pairs(
            self.garment_rgba,
            landmarks,
            'shirt_front'
        )
        
        if src_pts is None or dst_pts is None:
            return None, "Insufficient keypoints detected"
        
        # Analyze control point distribution
        analysis = {
            'num_points': len(src_pts),
            'landmarks_detected': len(landmarks),
            'shoulder_width': 0,
            'torso_height': 0,
            'fit_quality': 'Unknown'
        }
        
        # Calculate shoulder width from destination points
        if 5 in landmarks and 6 in landmarks:  # left and right shoulder
            left_shoulder = landmarks[5]
            right_shoulder = landmarks[6]
            if left_shoulder is not None and right_shoulder is not None:
                analysis['shoulder_width'] = np.linalg.norm(
                    np.array(left_shoulder) - np.array(right_shoulder)
                )
        
        # Calculate torso height
        if 5 in landmarks and 11 in landmarks:  # shoulder to hip
            shoulder = landmarks[5]
            hip = landmarks[11]
            if shoulder is not None and hip is not None:
                analysis['torso_height'] = abs(hip[1] - shoulder[1])
        
        # Assess fit quality based on point distribution
        if analysis['num_points'] >= 14:
            analysis['fit_quality'] = 'Excellent'
        elif analysis['num_points'] >= 10:
            analysis['fit_quality'] = 'Good'
        elif analysis['num_points'] >= 6:
            analysis['fit_quality'] = 'Fair'
        else:
            analysis['fit_quality'] = 'Poor'
        
        return analysis, None
    
    def draw_analysis_overlay(self, frame, landmarks, analysis):
        """Draw analysis information on frame."""
        vis = frame.copy()
        
        # Draw skeleton if enabled
        if self.show_keypoints:
            vis = draw_skeleton(vis, landmarks)
        
        # Draw control points if enabled
        if self.show_control_points:
            src_pts, dst_pts = get_control_point_pairs(
                self.garment_rgba,
                landmarks,
                'shirt_front'
            )
            
            if src_pts is not None and dst_pts is not None:
                for i, (src, dst) in enumerate(zip(src_pts, dst_pts)):
                    # Draw destination point (on body)
                    cv2.circle(vis, (int(dst[0]), int(dst[1])), 8, (0, 255, 0), -1)
                    cv2.circle(vis, (int(dst[0]), int(dst[1])), 10, (255, 255, 255), 2)
                    cv2.putText(vis, str(i), (int(dst[0]) + 12, int(dst[1])), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw mask if enabled
        if self.show_mask:
            mask = create_body_mask(frame.shape, landmarks, padding=40)
            mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            alpha = 0.3
            cv2.addWeighted(mask_colored, alpha, vis, 1 - alpha, 0, vis)
        
        # Draw analysis info
        if analysis:
            y_offset = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            cv2.putText(vis, f"Control Points: {analysis['num_points']}", 
                       (10, y_offset), font, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
            cv2.putText(vis, f"Fit Quality: {analysis['fit_quality']}", 
                       (10, y_offset), font, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
            if analysis['shoulder_width'] > 0:
                cv2.putText(vis, f"Shoulder Width: {analysis['shoulder_width']:.0f}px", 
                           (10, y_offset), font, 0.6, (255, 255, 255), 1)
                y_offset += 25
            
            if analysis['torso_height'] > 0:
                cv2.putText(vis, f"Torso Height: {analysis['torso_height']:.0f}px", 
                           (10, y_offset), font, 0.6, (255, 255, 255), 1)
        
        # Draw help text
        help_y = frame.shape[0] - 120
        cv2.putText(vis, "Controls:", (10, help_y), font, 0.6, (255, 255, 0), 2)
        help_y += 25
        cv2.putText(vis, "k: Toggle keypoints | c: Toggle control points", 
                   (10, help_y), font, 0.5, (255, 255, 255), 1)
        help_y += 20
        cv2.putText(vis, "m: Toggle mask | g: Toggle original garment", 
                   (10, help_y), font, 0.5, (255, 255, 255), 1)
        help_y += 20
        cv2.putText(vis, "s: Save screenshot | q: Quit", 
                   (10, help_y), font, 0.5, (255, 255, 255), 1)
        
        return vis
    
    def run(self):
        """Run the monitoring tool."""
        print("\n" + "="*70)
        print("TPS Garment Fit Monitor")
        print("="*70)
        print("This tool helps you monitor how the shirt wraps on your body.")
        print("\nControls:")
        print("  k: Toggle skeleton keypoints")
        print("  c: Toggle control points")
        print("  m: Toggle body mask overlay")
        print("  g: Toggle original garment view")
        print("  s: Save screenshot")
        print("  q: Quit")
        print("="*70 + "\n")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Monitor] Failed to read frame")
                break
            
            # Detect pose
            landmarks = get_landmarks(frame)
            
            if landmarks:
                # Analyze fit
                analysis, error = self.analyze_fit(frame, landmarks)
                
                if error:
                    # Show error
                    vis = frame.copy()
                    cv2.putText(vis, f"Error: {error}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('TPS Monitor - Analysis', vis)
                else:
                    # Warp garment
                    src_pts, dst_pts = get_control_point_pairs(
                        self.garment_rgba,
                        landmarks,
                        'shirt_front'
                    )
                    
                    if src_pts is not None and dst_pts is not None:
                        warped = self.tps_warper.warp(
                            self.garment_rgba,
                            src_pts,
                            dst_pts,
                            frame.shape[:2]
                        )
                        
                        # Create result with garment overlay
                        result = overlay(frame, warped, landmarks, mask_type='body')
                        
                        # Draw analysis overlay
                        analysis_vis = self.draw_analysis_overlay(frame, landmarks, analysis)
                        
                        # Show windows
                        cv2.imshow('TPS Monitor - Result', result)
                        cv2.imshow('TPS Monitor - Analysis', analysis_vis)
                        
                        # Optionally show original garment
                        if self.show_original_garment:
                            garment_vis = self.garment_rgba.copy()
                            if garment_vis.shape[2] == 4:
                                # Show on white background
                                bg = np.ones_like(garment_vis[:, :, :3]) * 255
                                alpha = garment_vis[:, :, 3:4] / 255.0
                                garment_vis = (alpha * garment_vis[:, :, :3] + (1 - alpha) * bg).astype(np.uint8)
                            
                            # Draw control points on garment
                            mapping = GarmentMapping.get_shirt_front_mapping()
                            control_pts_norm = mapping['control_points_normalized']
                            h, w = self.garment_rgba.shape[:2]
                            control_pts_px = denormalize_points(control_pts_norm, (h, w))
                            
                            for i, (name, (x, y)) in enumerate(control_pts_px.items()):
                                cv2.circle(garment_vis, (int(x), int(y)), 8, (0, 255, 0), -1)
                                cv2.putText(garment_vis, str(i), (int(x) + 10, int(y)), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                            
                            cv2.imshow('TPS Monitor - Original Garment', garment_vis)
            else:
                vis = frame.copy()
                cv2.putText(vis, 'No pose detected - stand in front of camera', 
                           (10, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 0, 255), 2)
                cv2.imshow('TPS Monitor - Analysis', vis)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('k'):
                self.show_keypoints = not self.show_keypoints
                print(f"[Monitor] Keypoints: {'ON' if self.show_keypoints else 'OFF'}")
            elif key == ord('c'):
                self.show_control_points = not self.show_control_points
                print(f"[Monitor] Control points: {'ON' if self.show_control_points else 'OFF'}")
            elif key == ord('m'):
                self.show_mask = not self.show_mask
                print(f"[Monitor] Mask overlay: {'ON' if self.show_mask else 'OFF'}")
            elif key == ord('g'):
                self.show_original_garment = not self.show_original_garment
                print(f"[Monitor] Original garment: {'ON' if self.show_original_garment else 'OFF'}")
                if not self.show_original_garment:
                    cv2.destroyWindow('TPS Monitor - Original Garment')
            elif key == ord('s'):
                # Save screenshots
                timestamp = cv2.getTickCount()
                output_dir = os.path.join(self.project_root, 'monitor_output')
                os.makedirs(output_dir, exist_ok=True)
                
                if landmarks:
                    # Save result
                    cv2.imwrite(os.path.join(output_dir, f'result_{timestamp}.png'), result)
                    # Save analysis
                    cv2.imwrite(os.path.join(output_dir, f'analysis_{timestamp}.png'), analysis_vis)
                    print(f"[Monitor] Screenshots saved to {output_dir}/")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n[Monitor] Closed")


if __name__ == '__main__':
    monitor = TPSMonitor()
    monitor.run()
