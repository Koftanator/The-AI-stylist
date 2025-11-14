#!/usr/bin/env python3
"""
Garment Fit Analysis Tool

Analyzes the current TPS warping and identifies specific issues with the fit:
1. Shoulder alignment
2. Sleeve positioning
3. Torso coverage
4. Overall proportions

Generates a detailed report with recommendations for control point adjustments.
"""

import cv2
import numpy as np
import os
import sys
import json
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from pose_yolo import get_landmarks
from garment_mapping import get_control_point_pairs, GarmentMapping
from tps_warp import TPSWarper
from overlay import create_body_mask


class FitAnalyzer:
    """Analyze garment fit quality and generate recommendations."""
    
    def __init__(self):
        self.project_root = os.path.dirname(script_dir)
        self.tps_warper = TPSWarper(use_gpu=True, downsample_factor=0.5)
        
        # Load garment
        garment_path = os.path.join(self.project_root, 'assests', 'garments', 'front_seg.png')
        self.garment_rgba = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
        
        if self.garment_rgba is None:
            raise FileNotFoundError(f"Cannot load garment from {garment_path}")
        
        print(f"[Analyzer] Garment loaded: {self.garment_rgba.shape}")
        
    def analyze_shoulder_fit(self, landmarks, dst_pts):
        """Analyze if shoulders align properly."""
        issues = []
        
        if 5 not in landmarks or 6 not in landmarks:
            return ["Cannot analyze shoulders - keypoints not detected"]
        
        left_shoulder = np.array(landmarks[5])
        right_shoulder = np.array(landmarks[6])
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        
        # Find shoulder control points in destination
        # Typically the first few control points are neck and shoulders
        if len(dst_pts) >= 3:
            # Estimate garment shoulder width from control points
            garment_shoulder_pts = []
            for i, pt in enumerate(dst_pts[:8]):  # Check first 8 points
                # If point is near shoulder height
                shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                if abs(pt[1] - shoulder_y) < 50:  # Within 50px of shoulder
                    garment_shoulder_pts.append(pt)
            
            if len(garment_shoulder_pts) >= 2:
                xs = [pt[0] for pt in garment_shoulder_pts]
                garment_width = max(xs) - min(xs)
                
                width_ratio = garment_width / shoulder_width
                
                if width_ratio < 0.8:
                    issues.append(f"Shoulders too narrow ({width_ratio:.2f}x) - garment should be wider")
                elif width_ratio > 1.3:
                    issues.append(f"Shoulders too wide ({width_ratio:.2f}x) - garment should be narrower")
                else:
                    issues.append(f"✓ Shoulder width good ({width_ratio:.2f}x)")
        
        return issues
    
    def analyze_sleeve_fit(self, landmarks, dst_pts):
        """Analyze if sleeves align with arms."""
        issues = []
        
        # Check if arm keypoints are detected
        arm_keypoints = [5, 6, 7, 8, 9, 10]  # shoulders, elbows, wrists
        detected_arms = sum(1 for kp in arm_keypoints if kp in landmarks)
        
        if detected_arms < 4:
            return [f"Cannot analyze sleeves - only {detected_arms}/6 arm keypoints detected"]
        
        # Check left arm
        if 5 in landmarks and 7 in landmarks and 9 in landmarks:
            shoulder = np.array(landmarks[5])
            elbow = np.array(landmarks[7])
            wrist = np.array(landmarks[9])
            
            arm_length = np.linalg.norm(wrist - shoulder)
            upper_arm = np.linalg.norm(elbow - shoulder)
            
            issues.append(f"✓ Left arm detected (length: {arm_length:.0f}px)")
        
        # Check right arm
        if 6 in landmarks and 8 in landmarks and 10 in landmarks:
            shoulder = np.array(landmarks[6])
            elbow = np.array(landmarks[8])
            wrist = np.array(landmarks[10])
            
            arm_length = np.linalg.norm(wrist - shoulder)
            
            issues.append(f"✓ Right arm detected (length: {arm_length:.0f}px)")
        
        return issues
    
    def analyze_torso_fit(self, landmarks, dst_pts):
        """Analyze torso coverage."""
        issues = []
        
        if 5 not in landmarks or 11 not in landmarks:
            return ["Cannot analyze torso - shoulder/hip keypoints not detected"]
        
        shoulder = np.array(landmarks[5])
        hip = np.array(landmarks[11])
        torso_height = abs(hip[1] - shoulder[1])
        
        # Find topmost and bottommost destination points
        if len(dst_pts) > 0:
            ys = [pt[1] for pt in dst_pts]
            garment_height = max(ys) - min(ys)
            
            height_ratio = garment_height / torso_height
            
            if height_ratio < 0.7:
                issues.append(f"Torso coverage too short ({height_ratio:.2f}x) - garment should extend lower")
            elif height_ratio > 1.2:
                issues.append(f"Torso coverage too long ({height_ratio:.2f}x) - garment extending too far")
            else:
                issues.append(f"✓ Torso coverage good ({height_ratio:.2f}x)")
        
        issues.append(f"Torso height: {torso_height:.0f}px")
        
        return issues
    
    def analyze_control_point_distribution(self, src_pts, dst_pts):
        """Analyze if control points are well distributed."""
        issues = []
        
        issues.append(f"Total control points: {len(src_pts)}")
        
        # Check for clustering
        if len(dst_pts) >= 4:
            distances = []
            for i in range(len(dst_pts)):
                for j in range(i + 1, len(dst_pts)):
                    dist = np.linalg.norm(dst_pts[i] - dst_pts[j])
                    distances.append(dist)
            
            avg_dist = np.mean(distances)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            issues.append(f"Point spacing: avg={avg_dist:.1f}, min={min_dist:.1f}, max={max_dist:.1f}")
            
            if min_dist < avg_dist * 0.2:
                issues.append("⚠ Some points are too close together - may cause clustering")
            
            if max_dist > avg_dist * 3:
                issues.append("⚠ Some points are too far apart - may need more intermediate points")
        
        return issues
    
    def generate_report(self, frame, landmarks):
        """Generate comprehensive fit analysis report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'frame_size': frame.shape[:2],
            'landmarks_detected': len(landmarks),
            'issues': {}
        }
        
        # Get control points
        src_pts, dst_pts = get_control_point_pairs(
            self.garment_rgba,
            landmarks,
            'shirt_front'
        )
        
        if src_pts is None or dst_pts is None:
            report['error'] = 'Could not generate control points'
            return report
        
        # Run analyses
        report['issues']['shoulders'] = self.analyze_shoulder_fit(landmarks, dst_pts)
        report['issues']['sleeves'] = self.analyze_sleeve_fit(landmarks, dst_pts)
        report['issues']['torso'] = self.analyze_torso_fit(landmarks, dst_pts)
        report['issues']['distribution'] = self.analyze_control_point_distribution(src_pts, dst_pts)
        
        return report
    
    def print_report(self, report):
        """Print report to console."""
        print("\n" + "="*70)
        print("GARMENT FIT ANALYSIS REPORT")
        print("="*70)
        print(f"Time: {report['timestamp']}")
        print(f"Frame size: {report['frame_size']}")
        print(f"Landmarks detected: {report['landmarks_detected']}")
        
        if 'error' in report:
            print(f"\n❌ ERROR: {report['error']}")
            return
        
        for category, issues in report['issues'].items():
            print(f"\n{category.upper()}:")
            for issue in issues:
                if issue.startswith('✓'):
                    print(f"  {issue}")
                elif issue.startswith('⚠'):
                    print(f"  {issue}")
                else:
                    print(f"  • {issue}")
        
        print("\n" + "="*70)
    
    def save_report(self, report, filename='fit_report.json'):
        """Save report to file."""
        output_dir = os.path.join(self.project_root, 'monitor_output')
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n[Analyzer] Report saved to: {filepath}")
    
    def run_interactive(self):
        """Run interactive analysis with live camera."""
        print("\n" + "="*70)
        print("Interactive Garment Fit Analyzer")
        print("="*70)
        print("Press SPACE to capture and analyze")
        print("Press 'q' to quit")
        print("="*70 + "\n")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Show live feed
            display = frame.copy()
            cv2.putText(display, "Press SPACE to analyze fit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Fit Analyzer', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space bar
                # Analyze current frame
                landmarks = get_landmarks(frame)
                
                if landmarks:
                    report = self.generate_report(frame, landmarks)
                    self.print_report(report)
                    self.save_report(report)
                    
                    # Wait for user
                    print("\nPress any key to continue...")
                    cv2.waitKey(0)
                else:
                    print("\n⚠ No pose detected in frame")
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    analyzer = FitAnalyzer()
    analyzer.run_interactive()
