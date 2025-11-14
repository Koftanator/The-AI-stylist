"""
Real-time 2D Virtual Try-On with TPS Warping
Optimized for NVIDIA RTX 4060 GPU

Features:
- YOLOv8-pose skeleton detection
- Multiple warping modes: perspective, affine, TPS
- GPU-accelerated TPS with intelligent caching
- Real-time FPS monitoring
- Threaded video processing for smooth performance

Controls:
    q: Quit
    t: Toggle front/back view
    m: Cycle warp modes (Perspective -> Affine -> TPS)
    r: Reset TPS cache
    f: Toggle FPS display
"""

import cv2
import time
import os
from collections import deque
from warp import warp_image, set_warp_mode, get_warp_mode, reset_tps_cache
from overlay import overlay
from overlay_skeleton import draw_skeleton
from debug_visualizer import DebugVisualizer
from garment_mapping import get_control_point_pairs

# Prefer YOLOv8 pose backend if available, otherwise fall back to existing pose
try:
    from pose_yolo import get_landmarks
    print('[MAIN] Using YOLOv8 pose backend')
except Exception:
    from pose import get_landmarks
    print('[MAIN] Using MoveNet pose backend')


class FPSCounter:
    """Simple FPS counter with exponential moving average."""
    
    def __init__(self, window_size=30):
        self.timestamps = deque(maxlen=window_size)
        self.fps = 0.0
    
    def update(self):
        """Call once per frame to update FPS."""
        self.timestamps.append(time.time())
        if len(self.timestamps) >= 2:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            self.fps = (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self.fps


def main():
    """Main application loop with mode switching and FPS monitoring."""
    
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from src/
    
    # Preload segmented garments using absolute paths
    garment_dir = os.path.join(project_root, 'assests', 'garments')
    front_path = os.path.join(garment_dir, 'front_seg.png')
    back_path = os.path.join(garment_dir, 'back_seg.png')
    
    front_rgba = cv2.imread(front_path, cv2.IMREAD_UNCHANGED)
    back_rgba = cv2.imread(back_path, cv2.IMREAD_UNCHANGED)

    # Validate loaded garments
    if front_rgba is None:
        print(f"WARNING: front_seg.png not found at: {front_path}")
        print(f"         Please ensure the file exists in: {garment_dir}")
    else:
        print(f"[MAIN] Front garment loaded: {front_rgba.shape}")
        
    if back_rgba is None:
        print(f"WARNING: back_seg.png not found at: {back_path}")
        print(f"         Please ensure the file exists in: {garment_dir}")
    else:
        print(f"[MAIN] Back garment loaded: {back_rgba.shape}")

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for optimal performance on RTX 4060
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Application state
    view_mode = 'front'  # 'front' or 'back'
    warp_modes = ['perspective', 'affine', 'tps']
    current_warp_idx = 2  # Start with TPS mode for best quality
    set_warp_mode(warp_modes[current_warp_idx])
    
    # Mask mode state
    mask_modes = ['body', 'torso', 'head']  # body=shoulders/elbows/wrists/hips, torso=shoulders/hips/knees, head=nose/eyes
    current_mask_idx = 0  # Start with body mask (best for shirts with sleeves)
    
    show_fps = True
    fps_counter = FPSCounter()
    
    # Debug mode initialization
    debugger = DebugVisualizer()
    debug_mode = False
    
    print("\n" + "="*60)
    print("2D Virtual Try-On with TPS Warping")
    print("="*60)
    print(f"Starting in: {warp_modes[current_warp_idx].upper()} mode")
    print("Controls:")
    print("  q: Quit")
    print("  t: Toggle front/back view")
    print("  m: Cycle warp modes (Perspective -> Affine -> TPS)")
    print("  b: Cycle mask modes (Body -> Torso -> Head)")
    print("  r: Reset TPS cache")
    print("  f: Toggle FPS display")
    print("  d: Toggle debug mode (keypoint + control point visualization)")
    print("="*60)
    print("Mask Modes:")
    print("  Body:  Uses shoulders, elbows, wrists, hips (full upper body)")
    print("  Torso: Uses shoulders, hips, knees (torso only)")
    print("  Head:  Uses nose, eyes (face/head region)")
    print("="*60)
    print("NOTE: TPS mode + Body mask = Best garment fit!")
    print("="*60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[MAIN] Failed to read frame, exiting")
            break

        # Update FPS counter
        fps_counter.update()

        # Detect pose landmarks
        landmarks = get_landmarks(frame)
        display = frame.copy()

        if landmarks:
            # 1. Draw skeleton overlay
            display = draw_skeleton(display, landmarks)

            # 2. Warp chosen garment
            garment = front_rgba if view_mode == 'front' else back_rgba
            garment_type = 'shirt_front' if view_mode == 'front' else 'shirt_back'
            
            if garment is not None:
                warped = warp_image(display, garment, landmarks, garment_type)
            else:
                warped = None

            # 3. Overlay garment onto frame (with body masking)
            if warped is not None:
                current_mask = mask_modes[current_mask_idx]
                output = overlay(display, warped, landmarks, mask_type=current_mask)
            else:
                output = display
                cv2.putText(
                    output, 'Garment not loaded!', 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )
            
            # 4. Add debug visualization if enabled
            if debug_mode and garment is not None:
                src_pts, dst_pts = get_control_point_pairs(garment, landmarks, garment_type)
                if src_pts is not None and dst_pts is not None:
                    output = debugger.visualize_all(
                        output, landmarks, src_pts, dst_pts,
                        garment_shape=garment.shape[:2]
                    )
        else:
            output = display
            # Show why no pose was applied
            cv2.putText(
                output, 'Pose disabled or not detected', 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

        # Display UI elements
        y_offset = 30
        
        # View mode indicator
        cv2.putText(
            output, f'View: {view_mode.upper()}', 
            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        y_offset += 30
        
        # Warp mode indicator
        current_warp = get_warp_mode()
        warp_color = (0, 255, 0) if current_warp == 'tps' else (0, 255, 255)
        cv2.putText(
            output, f'Warp: {current_warp.upper()}', 
            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warp_color, 2
        )
        # Add indicator for TPS mode
        if current_warp == 'tps':
            cv2.putText(
                output, '(TPS Active)', 
                (220, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        y_offset += 30
        
        # Mask mode indicator
        current_mask = mask_modes[current_mask_idx]
        mask_color = (0, 255, 255) if current_mask != 'none' else (100, 100, 100)
        cv2.putText(
            output, f'Mask: {current_mask.upper()}', 
            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mask_color, 2
        )
        y_offset += 30
        
        # FPS counter
        if show_fps:
            fps_text = f'FPS: {fps_counter.get_fps():.1f}'
            cv2.putText(
                output, fps_text, 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
        
        # Display output
        cv2.imshow('2D Try-On with TPS', output)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("[MAIN] Quit requested")
            break
        
        elif key == ord('t'):
            # Toggle front/back view
            view_mode = 'back' if view_mode == 'front' else 'front'
            print(f"[MAIN] Switched to {view_mode} view")
            # Reset TPS cache when changing views
            reset_tps_cache()
        
        elif key == ord('m'):
            # Cycle through warp modes
            current_warp_idx = (current_warp_idx + 1) % len(warp_modes)
            new_mode = warp_modes[current_warp_idx]
            set_warp_mode(new_mode)
            print(f"[MAIN] Warp mode: {new_mode.upper()}")
        
        elif key == ord('r'):
            # Reset TPS cache
            reset_tps_cache()
            print("[MAIN] TPS cache reset")
        
        elif key == ord('f'):
            # Toggle FPS display
            show_fps = not show_fps
            print(f"[MAIN] FPS display: {'ON' if show_fps else 'OFF'}")
        
        elif key == ord('b'):
            # Cycle through mask modes
            current_mask_idx = (current_mask_idx + 1) % len(mask_modes)
            new_mask = mask_modes[current_mask_idx]
            print(f"[MAIN] Mask mode: {new_mask.upper()}")
        
        elif key == ord('d'):
            # Toggle debug mode
            debug_mode = not debug_mode
            print(f"[MAIN] Debug mode: {'ON' if debug_mode else 'OFF'}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[MAIN] Application closed")


if __name__ == '__main__':
    main()
