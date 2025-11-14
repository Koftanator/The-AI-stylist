#!/usr/bin/env python3
"""
Simple tool to visualize the garment image and its control points
"""

import cv2
import numpy as np
import os
import sys

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from garment_mapping import GarmentMapping, denormalize_points

def main():
    # Load garment
    project_root = os.path.dirname(script_dir)
    garment_path = os.path.join(project_root, 'assests', 'garments', 'front_seg.png')
    garment_rgba = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
    
    if garment_rgba is None:
        print(f"ERROR: Cannot load garment from {garment_path}")
        return
    
    print(f"Garment shape: {garment_rgba.shape}")
    print(f"Alpha channel: {garment_rgba.shape[2] == 4}")
    
    # Get mapping
    mapping = GarmentMapping.get_shirt_front_mapping()
    control_pts_norm = mapping['control_points_normalized']
    
    # Denormalize to pixels
    garment_h, garment_w = garment_rgba.shape[:2]
    control_pts_px = denormalize_points(control_pts_norm, (garment_h, garment_w))
    
    print(f"\nControl points ({len(control_pts_px)}):")
    for i, (name, (x, y)) in enumerate(control_pts_px.items()):
        print(f"  {i:2d}. {name:20s} -> ({x:4d}, {y:4d})")
    
    # Create visualization
    vis = garment_rgba.copy()
    
    # Convert RGBA to BGR for display
    if vis.shape[2] == 4:
        # Create white background
        bg = np.ones((garment_h, garment_w, 3), dtype=np.uint8) * 255
        alpha = vis[:, :, 3:4] / 255.0
        for c in range(3):
            bg[:, :, c] = (1 - alpha[:, :, 0]) * bg[:, :, c] + alpha[:, :, 0] * vis[:, :, c]
        vis = bg
    
    # Draw control points
    colors = [
        (255, 0, 0),    # Blue - neck
        (0, 255, 0),    # Green - shoulders
        (0, 0, 255),    # Red - body
        (255, 255, 0),  # Cyan - hips
        (255, 0, 255),  # Magenta - sleeves
    ]
    
    for i, (name, (x, y)) in enumerate(control_pts_px.items()):
        # Color by region
        if 'neck' in name:
            color = colors[0]
        elif 'shoulder' in name:
            color = colors[1]
        elif 'chest' in name or 'waist' in name:
            color = colors[2]
        elif 'hip' in name:
            color = colors[3]
        else:  # sleeves
            color = colors[4]
        
        # Draw point
        cv2.circle(vis, (int(x), int(y)), 12, color, -1)
        cv2.circle(vis, (int(x), int(y)), 14, (255, 255, 255), 2)
        
        # Draw label
        cv2.putText(vis, str(i), (int(x) - 8, int(y) + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add legend
    y_offset = 30
    cv2.putText(vis, "Control Points on Garment Template", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    cv2.putText(vis, "Control Points on Garment Template", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    y_offset += 40
    legend_items = [
        ("Neck", colors[0]),
        ("Shoulders", colors[1]),
        ("Body", colors[2]),
        ("Hips", colors[3]),
        ("Sleeves", colors[4])
    ]
    
    for text, color in legend_items:
        cv2.circle(vis, (20, y_offset), 8, color, -1)
        cv2.putText(vis, text, (35, y_offset + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y_offset += 30
    
    # Show
    cv2.imshow('Garment Template with Control Points', vis)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save
    output_path = os.path.join(project_root, 'garment_control_points_visualization.png')
    cv2.imwrite(output_path, vis)
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()
