"""
Debug Visualization Tool for TPS Virtual Try-On
Helps diagnose misalignment issues in garment fitting

Features:
- Keypoint confidence visualization with color-coding
- Control point source→destination mapping overlay
- TPS weight matrix validation
- Garment segmentation quality checks
- Real-time alignment metrics

Usage:
    from debug_visualizer import DebugVisualizer
    
    debugger = DebugVisualizer()
    debug_frame = debugger.visualize_all(frame, landmarks, src_pts, dst_pts)
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
import warnings


class DebugVisualizer:
    """Visual debugging tools for TPS alignment issues."""
    
    # COCO keypoint names for display
    KEYPOINT_NAMES = {
        0: 'nose', 1: 'left_eye', 2: 'right_eye',
        3: 'left_ear', 4: 'right_ear',
        5: 'left_shoulder', 6: 'right_shoulder',
        7: 'left_elbow', 8: 'right_elbow',
        9: 'left_wrist', 10: 'right_wrist',
        11: 'left_hip', 12: 'right_hip',
        13: 'left_knee', 14: 'right_knee',
        15: 'left_ankle', 16: 'right_ankle'
    }
    
    # Critical keypoints for shirt fitting (should have high confidence)
    CRITICAL_KEYPOINTS = [5, 6, 7, 8, 11, 12]  # shoulders, elbows, hips
    
    def __init__(self, min_confidence=0.5, show_labels=True):
        """
        Initialize debug visualizer.
        
        Args:
            min_confidence: Minimum confidence threshold for keypoints
            show_labels: Show keypoint labels on visualization
        """
        self.min_confidence = min_confidence
        self.show_labels = show_labels
        
        # Color schemes
        self.color_high_conf = (0, 255, 0)      # Green = high confidence
        self.color_med_conf = (0, 255, 255)     # Yellow = medium confidence
        self.color_low_conf = (0, 0, 255)       # Red = low confidence
        self.color_missing = (128, 128, 128)    # Gray = missing
        self.color_control_pt = (255, 0, 255)   # Magenta = control points
    
    def visualize_keypoints_with_confidence(
        self,
        frame: np.ndarray,
        landmarks: Dict[int, Optional[Tuple[int, int]]],
        confidences: Optional[Dict[int, float]] = None
    ) -> np.ndarray:
        """
        Draw keypoints with color-coded confidence scores.
        
        Args:
            frame: Input image
            landmarks: Detected keypoints {idx: (x, y)}
            confidences: Confidence scores {idx: float}
            
        Returns:
            Frame with keypoint visualization
        """
        output = frame.copy()
        
        for idx, point in landmarks.items():
            if point is None:
                continue
            
            x, y = point
            
            # Determine color based on confidence
            if confidences and idx in confidences:
                conf = confidences[idx]
                if conf >= 0.7:
                    color = self.color_high_conf
                elif conf >= 0.5:
                    color = self.color_med_conf
                else:
                    color = self.color_low_conf
                
                # Draw confidence text
                conf_text = f"{conf:.2f}"
                cv2.putText(output, conf_text, (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
                color = self.color_high_conf  # Default if no confidence
            
            # Draw keypoint
            cv2.circle(output, (x, y), 6, color, -1)
            cv2.circle(output, (x, y), 8, (255, 255, 255), 2)
            
            # Draw keypoint label
            if self.show_labels and idx in self.KEYPOINT_NAMES:
                label = self.KEYPOINT_NAMES[idx]
                cv2.putText(output, label, (x + 15, y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return output
    
    def visualize_control_point_mapping(
        self,
        frame: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        garment_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Draw control point source→destination mappings.
        
        Args:
            frame: Input image
            src_pts: Source control points (N, 2) - garment coordinates
            dst_pts: Destination control points (N, 2) - body coordinates
            garment_shape: Optional (H, W) to show garment overlay
            
        Returns:
            Frame with control point visualization
        """
        output = frame.copy()
        
        if src_pts is None or dst_pts is None:
            cv2.putText(output, "No control points available", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return output
        
        # Draw garment bounding box (if shape provided)
        if garment_shape is not None:
            gh, gw = garment_shape
            # Show garment size reference
            cv2.putText(output, f"Garment: {gw}x{gh}px", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw source points (small, garment coords are relative)
        for i, (src, dst) in enumerate(zip(src_pts, dst_pts)):
            # Destination points (on body)
            dst_x, dst_y = int(dst[0]), int(dst[1])
            
            # Draw destination point (large)
            cv2.circle(output, (dst_x, dst_y), 8, self.color_control_pt, -1)
            cv2.circle(output, (dst_x, dst_y), 10, (255, 255, 255), 2)
            
            # Draw point index
            cv2.putText(output, str(i), (dst_x + 12, dst_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw lines connecting nearby points to show structure
        if len(dst_pts) > 1:
            # Connect shoulder line
            for i in range(len(dst_pts) - 1):
                pt1 = tuple(dst_pts[i].astype(int))
                pt2 = tuple(dst_pts[i + 1].astype(int))
                
                # Only draw line if points are close (part of same structure)
                dist = np.linalg.norm(dst_pts[i] - dst_pts[i + 1])
                if dist < 200:  # Adjust threshold as needed
                    cv2.line(output, pt1, pt2, self.color_control_pt, 1)
        
        # Show control point count
        cv2.putText(output, f"Control Points: {len(src_pts)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color_control_pt, 2)
        
        return output
    
    def validate_tps_condition(
        self,
        src_pts: np.ndarray,
        min_spacing: float = 10.0,
        max_condition: float = 1e10
    ) -> Dict[str, Any]:
        """
        Validate TPS numerical stability.
        
        Args:
            src_pts: Source control points (N, 2)
            min_spacing: Minimum distance between points
            max_condition: Maximum acceptable condition number
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'num_points': len(src_pts),
            'min_distance': None,
            'condition_number': None,
            'warnings': [],
            'errors': []
        }
        
        # Check number of points
        if len(src_pts) < 3:
            results['valid'] = False
            results['errors'].append(f"Too few control points: {len(src_pts)} (need ≥3)")
            return results
        
        # Check point spacing
        from scipy.spatial.distance import pdist
        distances = pdist(src_pts)
        min_dist = np.min(distances)
        results['min_distance'] = float(min_dist)
        
        if min_dist < min_spacing:
            results['warnings'].append(
                f"Points too close: min={min_dist:.1f}px (recommend ≥{min_spacing}px)"
            )
        
        # Check TPS matrix condition number
        try:
            from scipy.spatial.distance import cdist
            
            n = len(src_pts)
            
            # Build TPS kernel matrix
            r = cdist(src_pts, src_pts)
            r2 = r ** 2
            K = np.where(r > 0, r2 * np.log(r2), 0)
            
            # Build full TPS matrix
            P = np.hstack([np.ones((n, 1)), src_pts])
            A = np.zeros((n + 3, n + 3))
            A[:n, :n] = K
            A[:n, n:] = P
            A[n:, :n] = P.T
            
            # Compute condition number
            cond = np.linalg.cond(A)
            results['condition_number'] = float(cond)
            
            if cond > max_condition:
                results['warnings'].append(
                    f"Poor conditioning: {cond:.2e} (recommend <{max_condition:.0e})"
                )
            
        except Exception as e:
            results['warnings'].append(f"Could not compute condition number: {e}")
        
        # Final validation
        if results['errors']:
            results['valid'] = False
        
        return results
    
    def check_keypoint_reliability(
        self,
        landmarks: Dict[int, Optional[Tuple[int, int]]],
        confidences: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """
        Check reliability of detected keypoints.
        
        Args:
            landmarks: Detected keypoints
            confidences: Confidence scores
            
        Returns:
            Reliability report
        """
        report = {
            'total_detected': 0,
            'critical_detected': 0,
            'critical_missing': [],
            'low_confidence': [],
            'reliable_for_fitting': False
        }
        
        # Count detections
        detected = [idx for idx, pt in landmarks.items() if pt is not None]
        report['total_detected'] = len(detected)
        
        # Check critical keypoints
        for idx in self.CRITICAL_KEYPOINTS:
            if idx in landmarks and landmarks[idx] is not None:
                report['critical_detected'] += 1
                
                # Check confidence
                if confidences and idx in confidences:
                    if confidences[idx] < self.min_confidence:
                        report['low_confidence'].append(
                            (idx, self.KEYPOINT_NAMES[idx], confidences[idx])
                        )
            else:
                report['critical_missing'].append((idx, self.KEYPOINT_NAMES[idx]))
        
        # Determine if reliable for fitting
        # Need at least: both shoulders, both hips, both elbows
        required = [5, 6, 11, 12, 7, 8]  # shoulders, hips, elbows
        has_required = all(
            idx in landmarks and landmarks[idx] is not None 
            for idx in required
        )
        report['reliable_for_fitting'] = has_required
        
        return report
    
    def draw_alignment_metrics(
        self,
        frame: np.ndarray,
        validation_results: Dict[str, Any],
        reliability_report: Dict[str, Any]
    ) -> np.ndarray:
        """
        Draw alignment quality metrics on frame.
        
        Args:
            frame: Input image
            validation_results: TPS validation results
            reliability_report: Keypoint reliability report
            
        Returns:
            Frame with metrics overlay
        """
        output = frame.copy()
        
        # Position for text overlay
        x, y = 10, frame.shape[0] - 150
        line_height = 25
        
        # Background for text
        cv2.rectangle(output, (5, y - 10), (400, frame.shape[0] - 5), 
                     (0, 0, 0), -1)
        cv2.rectangle(output, (5, y - 10), (400, frame.shape[0] - 5), 
                     (255, 255, 255), 2)
        
        # Title
        cv2.putText(output, "Alignment Diagnostics", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += line_height
        
        # Keypoint reliability
        detected = reliability_report['total_detected']
        critical = reliability_report['critical_detected']
        reliable = reliability_report['reliable_for_fitting']
        
        status_color = self.color_high_conf if reliable else self.color_low_conf
        cv2.putText(output, f"Keypoints: {detected}/17, Critical: {critical}/6", 
                   (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        y += line_height
        
        # TPS validation
        if validation_results['valid']:
            cv2.putText(output, "TPS: Valid", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_high_conf, 1)
        else:
            cv2.putText(output, "TPS: INVALID", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_low_conf, 2)
        y += line_height
        
        # Control point info
        if validation_results['min_distance'] is not None:
            min_dist = validation_results['min_distance']
            cv2.putText(output, f"Min spacing: {min_dist:.1f}px", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += line_height
        
        # Warnings
        if validation_results['warnings']:
            cv2.putText(output, f"Warnings: {len(validation_results['warnings'])}", 
                       (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return output
    
    def visualize_all(
        self,
        frame: np.ndarray,
        landmarks: Dict[int, Optional[Tuple[int, int]]],
        src_pts: Optional[np.ndarray] = None,
        dst_pts: Optional[np.ndarray] = None,
        confidences: Optional[Dict[int, float]] = None,
        garment_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Create comprehensive debug visualization.
        
        Args:
            frame: Input image
            landmarks: Detected keypoints
            src_pts: Source control points
            dst_pts: Destination control points
            confidences: Keypoint confidence scores
            garment_shape: Garment dimensions
            
        Returns:
            Frame with all debug overlays
        """
        output = frame.copy()
        
        # 1. Draw keypoints with confidence
        output = self.visualize_keypoints_with_confidence(output, landmarks, confidences)
        
        # 2. Draw control point mapping
        if src_pts is not None and dst_pts is not None:
            output = self.visualize_control_point_mapping(
                output, src_pts, dst_pts, garment_shape
            )
        
        # 3. Validate and draw metrics
        if src_pts is not None:
            validation_results = self.validate_tps_condition(src_pts)
            reliability_report = self.check_keypoint_reliability(landmarks, confidences)
            output = self.draw_alignment_metrics(output, validation_results, reliability_report)
        
        return output
    
    def print_diagnostics(
        self,
        landmarks: Dict[int, Optional[Tuple[int, int]]],
        src_pts: Optional[np.ndarray] = None,
        dst_pts: Optional[np.ndarray] = None,
        confidences: Optional[Dict[int, float]] = None
    ):
        """
        Print detailed diagnostics to console.
        
        Args:
            landmarks: Detected keypoints
            src_pts: Source control points
            dst_pts: Destination control points
            confidences: Keypoint confidence scores
        """
        print("\n" + "="*60)
        print("TPS ALIGNMENT DIAGNOSTICS")
        print("="*60)
        
        # Keypoint reliability
        report = self.check_keypoint_reliability(landmarks, confidences)
        print(f"\nKeypoint Detection:")
        print(f"  Total detected: {report['total_detected']}/17")
        print(f"  Critical detected: {report['critical_detected']}/6")
        print(f"  Reliable for fitting: {report['reliable_for_fitting']}")
        
        if report['critical_missing']:
            print(f"\n  Missing critical keypoints:")
            for idx, name in report['critical_missing']:
                print(f"    - {name} (#{idx})")
        
        if report['low_confidence']:
            print(f"\n  Low confidence keypoints:")
            for idx, name, conf in report['low_confidence']:
                print(f"    - {name} (#{idx}): {conf:.3f}")
        
        # TPS validation
        if src_pts is not None:
            print(f"\nTPS Validation:")
            validation = self.validate_tps_condition(src_pts)
            print(f"  Valid: {validation['valid']}")
            print(f"  Control points: {validation['num_points']}")
            if validation['min_distance'] is not None:
                print(f"  Min point spacing: {validation['min_distance']:.1f}px")
            if validation['condition_number'] is not None:
                print(f"  Condition number: {validation['condition_number']:.2e}")
            
            if validation['warnings']:
                print(f"\n  Warnings:")
                for warning in validation['warnings']:
                    print(f"    ⚠ {warning}")
            
            if validation['errors']:
                print(f"\n  Errors:")
                for error in validation['errors']:
                    print(f"    ✗ {error}")
        
        print("="*60 + "\n")


def add_debug_mode_to_main(main_loop_code: str) -> str:
    """
    Generates code to add debug mode to main.py.
    
    Returns:
        Code snippet to integrate into main.py
    """
    code = """
# Add to imports at top of main.py:
from debug_visualizer import DebugVisualizer

# Add after FPSCounter initialization:
debug_mode = False  # Toggle with 'd' key
debugger = DebugVisualizer(min_confidence=0.5, show_labels=True)

# In main loop, after getting landmarks:
if debug_mode and landmarks:
    # Get control points for debugging
    from garment_mapping import get_control_point_pairs
    garment = front_rgba if view_mode == 'front' else back_rgba
    garment_type = 'shirt_front' if view_mode == 'front' else 'shirt_back'
    
    if garment is not None:
        src_pts, dst_pts = get_control_point_pairs(
            garment, landmarks, garment_type
        )
        
        # Create debug visualization
        display = debugger.visualize_all(
            frame=display,
            landmarks=landmarks,
            src_pts=src_pts,
            dst_pts=dst_pts,
            confidences=None,  # Add if available from pose detector
            garment_shape=garment.shape[:2]
        )
        
        # Print diagnostics every 30 frames
        if frame_count % 30 == 0:
            debugger.print_diagnostics(landmarks, src_pts, dst_pts)

# Add keyboard control:
elif key == ord('d'):
    debug_mode = not debug_mode
    print(f"[DEBUG] Debug mode: {'ON' if debug_mode else 'OFF'}")
"""
    return code


if __name__ == '__main__':
    print("Debug Visualizer Tool")
    print("="*60)
    print("\nThis module provides visual debugging for TPS alignment issues.")
    print("\nIntegration example:")
    print(add_debug_mode_to_main(""))
    print("\nPress 'd' during runtime to toggle debug visualization.")
