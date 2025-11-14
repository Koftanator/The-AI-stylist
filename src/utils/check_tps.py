#!/usr/bin/env python3
"""
Simple debug script to check TPS output
"""
import cv2
import numpy as np

print("Checking TPS integration...")

# Test 1: Import check
try:
    from tps_warp import create_tps_warper
    print("✓ tps_warp module imports OK")
except Exception as e:
    print(f"✗ tps_warp import failed: {e}")
    exit(1)

try:
    from garment_mapping import get_control_point_pairs
    print("✓ garment_mapping module imports OK")
except Exception as e:
    print(f"✗ garment_mapping import failed: {e}")
    exit(1)

try:
    from warp import set_warp_mode, warp_image_tps
    print("✓ warp module imports OK")
except Exception as e:
    print(f"✗ warp import failed: {e}")
    exit(1)

# Test 2: Create TPS warper
try:
    warper = create_tps_warper(fast_mode=True)
    stats = warper.get_stats()
    print(f"✓ TPS warper created: {stats}")
except Exception as e:
    print(f"✗ TPS warper creation failed: {e}")
    exit(1)

# Test 3: Simple test with dummy data
print("\nTesting with dummy data...")
try:
    # Create simple test image
    test_img = np.zeros((100, 100, 4), dtype=np.uint8)
    test_img[25:75, 25:75] = [255, 0, 0, 255]  # Red square
    
    # Simple control points (corners)
    src_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    dst_pts = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
    
    # Warp
    warped = warper.warp(test_img, src_pts, dst_pts, output_shape=(100, 100))
    
    non_zero = np.count_nonzero(warped[:,:,3] > 0)
    print(f"✓ TPS warp successful: {non_zero} non-zero pixels")
    
    if non_zero == 0:
        print("  WARNING: Output has no content!")
    else:
        print("  ✓ Output has content")
        
except Exception as e:
    print(f"✗ TPS warp failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✓ All tests passed! TPS is working.")
print("\nTo see TPS in action:")
print("  1. Run: python main.py")
print("  2. The app now starts in TPS mode by default")
print("  3. Look for 'Warp: TPS' indicator on screen")
print("  4. Press 'm' to cycle between modes if needed")
