#!/usr/bin/env python3
"""
Analyze garment structure and suggest better control point placements.

This tool analyzes the actual pixel distribution in the garment image
and suggests where anatomical features (neck, shoulders, hips) likely are.
"""

import cv2
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)


def analyze_garment_structure(garment_rgba):
    """Analyze where the shirt features actually are in the image."""
    h, w = garment_rgba.shape[:2]
    alpha = garment_rgba[:, :, 3]
    
    # Find garment bounds
    nonzero_mask = alpha > 128
    rows, cols = np.where(nonzero_mask)
    
    if len(rows) == 0:
        print("ERROR: No garment pixels found!")
        return None
    
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    
    # Calculate width profile (how wide the garment is at each row)
    width_profile = []
    row_positions = []
    
    for row in range(min_row, max_row + 1):
        row_mask = nonzero_mask[row, :]
        if np.any(row_mask):
            row_cols = np.where(row_mask)[0]
            if len(row_cols) > 0:
                width = row_cols.max() - row_cols.min()
                center = (row_cols.min() + row_cols.max()) / 2
                width_profile.append((row, width, center, row_cols.min(), row_cols.max()))
                row_positions.append(row)
    
    width_profile = np.array(width_profile)
    
    # Analyze structure
    total_height = max_row - min_row
    
    # Collar/neck (top 5-10%)
    neck_row_idx = int(len(width_profile) * 0.07)
    neck_row = int(width_profile[neck_row_idx, 0])
    neck_center = width_profile[neck_row_idx, 2]
    neck_left = width_profile[neck_row_idx, 3]
    neck_right = width_profile[neck_row_idx, 4]
    
    # Shoulders (top 15-20% - usually widest in upper region)
    shoulder_start = int(len(width_profile) * 0.10)
    shoulder_end = int(len(width_profile) * 0.25)
    shoulder_region = width_profile[shoulder_start:shoulder_end]
    shoulder_idx = shoulder_start + np.argmax(shoulder_region[:, 1])
    shoulder_row = int(width_profile[shoulder_idx, 0])
    shoulder_left = width_profile[shoulder_idx, 3]
    shoulder_right = width_profile[shoulder_idx, 4]
    
    # Chest (around 30-40%)
    chest_idx = int(len(width_profile) * 0.35)
    chest_row = int(width_profile[chest_idx, 0])
    chest_left = width_profile[chest_idx, 3]
    chest_right = width_profile[chest_idx, 4]
    
    # Waist (around 55-65%)
    waist_idx = int(len(width_profile) * 0.60)
    waist_row = int(width_profile[waist_idx, 0])
    waist_left = width_profile[waist_idx, 3]
    waist_right = width_profile[waist_idx, 4]
    
    # Hip/bottom (90-95%)
    hip_idx = int(len(width_profile) * 0.92)
    hip_row = int(width_profile[hip_idx, 0])
    hip_left = width_profile[hip_idx, 3]
    hip_center = width_profile[hip_idx, 2]
    hip_right = width_profile[hip_idx, 4]
    
    # Sleeve analysis (look for pixels extending to sides at mid-height)
    sleeve_row_idx = int(len(width_profile) * 0.30)
    sleeve_row = int(width_profile[sleeve_row_idx, 0])
    
    analysis = {
        'bounds': {'min_row': min_row, 'max_row': max_row, 'min_col': min_col, 'max_col': max_col},
        'height': h,
        'width': w,
        'total_height': total_height,
        'neck': {'row': neck_row, 'center': neck_center, 'left': neck_left, 'right': neck_right},
        'shoulders': {'row': shoulder_row, 'left': shoulder_left, 'right': shoulder_right},
        'chest': {'row': chest_row, 'left': chest_left, 'right': chest_right},
        'waist': {'row': waist_row, 'left': waist_left, 'right': waist_right},
        'hip': {'row': hip_row, 'left': hip_left, 'center': hip_center, 'right': hip_right},
        'sleeve_height': sleeve_row,
    }
    
    return analysis


def suggest_control_points(analysis):
    """Generate suggested control point coordinates based on garment structure."""
    h, w = analysis['height'], analysis['width']
    
    # Convert to normalized coordinates (0-1)
    suggested = {
        # Neck region
        'neck_center': (analysis['neck']['center'] / w, analysis['neck']['row'] / h),
        'neck_left': (analysis['neck']['left'] / w, analysis['neck']['row'] / h),
        'neck_right': (analysis['neck']['right'] / w, analysis['neck']['row'] / h),
        
        # Shoulders
        'shoulder_left': (analysis['shoulders']['left'] / w, analysis['shoulders']['row'] / h),
        'shoulder_right': (analysis['shoulders']['right'] / w, analysis['shoulders']['row'] / h),
        
        # Chest
        'chest_left': (analysis['chest']['left'] / w, analysis['chest']['row'] / h),
        'chest_right': (analysis['chest']['right'] / w, analysis['chest']['row'] / h),
        
        # Waist
        'waist_left': (analysis['waist']['left'] / w, analysis['waist']['row'] / h),
        'waist_right': (analysis['waist']['right'] / w, analysis['waist']['row'] / h),
        
        # Hips
        'hip_left': (analysis['hip']['left'] / w, analysis['hip']['row'] / h),
        'hip_center': (analysis['hip']['center'] / w, analysis['hip']['row'] / h),
        'hip_right': (analysis['hip']['right'] / w, analysis['hip']['row'] / h),
        
        # Sleeves (extend to edges at sleeve height)
        'sleeve_left_mid': (analysis['shoulders']['left'] / w, analysis['sleeve_height'] / h),
        'sleeve_right_mid': (analysis['shoulders']['right'] / w, analysis['sleeve_height'] / h),
        'sleeve_left_end': (min(analysis['shoulders']['left'] - 50, 5) / w, 
                            (analysis['sleeve_height'] + 100) / h),
        'sleeve_right_end': (max(analysis['shoulders']['right'] + 50, w - 5) / w, 
                             (analysis['sleeve_height'] + 100) / h),
    }
    
    return suggested


def main():
    print("\n" + "="*70)
    print("Garment Structure Analysis")
    print("="*70)
    
    # Load garment
    garment_path = os.path.join(project_root, 'assests', 'garments', 'front_seg.png')
    garment_rgba = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
    
    if garment_rgba is None:
        print(f"ERROR: Cannot load garment from {garment_path}")
        return
    
    print(f"\nGarment image: {garment_rgba.shape}")
    
    # Analyze structure
    analysis = analyze_garment_structure(garment_rgba)
    
    if analysis is None:
        return
    
    # Print analysis
    print(f"\nGarment bounds:")
    print(f"  Rows: {analysis['bounds']['min_row']} - {analysis['bounds']['max_row']}")
    print(f"  Cols: {analysis['bounds']['min_col']} - {analysis['bounds']['max_col']}")
    print(f"  Total height: {analysis['total_height']} pixels")
    
    print(f"\nDetected features:")
    print(f"  Neck:      row {analysis['neck']['row']:3d}  ({analysis['neck']['row']/analysis['height']:.3f})")
    print(f"  Shoulders: row {analysis['shoulders']['row']:3d}  ({analysis['shoulders']['row']/analysis['height']:.3f})")
    print(f"  Chest:     row {analysis['chest']['row']:3d}  ({analysis['chest']['row']/analysis['height']:.3f})")
    print(f"  Waist:     row {analysis['waist']['row']:3d}  ({analysis['waist']['row']/analysis['height']:.3f})")
    print(f"  Hip:       row {analysis['hip']['row']:3d}  ({analysis['hip']['row']/analysis['height']:.3f})")
    
    # Generate suggested control points
    suggested = suggest_control_points(analysis)
    
    print(f"\nSuggested control points (normalized 0-1 coordinates):")
    print("="*70)
    for name, (x, y) in suggested.items():
        print(f"    '{name}': ({x:.3f}, {y:.3f}),")
    print("="*70)
    
    # Visualize
    vis = garment_rgba.copy()
    if vis.shape[2] == 4:
        bg = np.ones_like(vis[:, :, :3]) * 255
        alpha = vis[:, :, 3:4] / 255.0
        vis = (alpha * vis[:, :, :3] + (1 - alpha) * bg).astype(np.uint8)
    
    h, w = vis.shape[:2]
    
    # Draw suggested points
    for i, (name, (x_norm, y_norm)) in enumerate(suggested.items()):
        x, y = int(x_norm * w), int(y_norm * h)
        
        # Color by region
        if 'neck' in name:
            color = (255, 0, 0)  # Blue
        elif 'shoulder' in name:
            color = (0, 255, 0)  # Green
        elif 'chest' in name or 'waist' in name:
            color = (0, 0, 255)  # Red
        elif 'hip' in name:
            color = (255, 255, 0)  # Cyan
        else:
            color = (255, 0, 255)  # Magenta
        
        cv2.circle(vis, (x, y), 10, color, -1)
        cv2.circle(vis, (x, y), 12, (255, 255, 255), 2)
        cv2.putText(vis, str(i), (x - 6, y + 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw feature lines
    cv2.line(vis, (0, analysis['neck']['row']), (w, analysis['neck']['row']), (255, 0, 0), 2)
    cv2.line(vis, (0, analysis['shoulders']['row']), (w, analysis['shoulders']['row']), (0, 255, 0), 2)
    cv2.line(vis, (0, analysis['chest']['row']), (w, analysis['chest']['row']), (0, 0, 255), 1)
    cv2.line(vis, (0, analysis['waist']['row']), (w, analysis['waist']['row']), (0, 0, 255), 1)
    cv2.line(vis, (0, analysis['hip']['row']), (w, analysis['hip']['row']), (255, 255, 0), 2)
    
    cv2.putText(vis, "Auto-detected Control Points", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    cv2.putText(vis, "Auto-detected Control Points", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    output_path = os.path.join(project_root, 'garment_analysis.png')
    cv2.imwrite(output_path, vis)
    print(f"\nVisualization saved to: {output_path}")
    
    cv2.imshow('Garment Analysis', vis)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
