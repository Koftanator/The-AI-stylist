"""
Performance Profiling Utility for Virtual Try-On System

Benchmarks different warping methods and identifies bottlenecks.
Useful for tuning parameters on RTX 4060 and other hardware.

Usage:
    python profiler.py [--mode perspective|affine|tps|all] [--frames 100]
"""

import cv2
import time
import numpy as np
import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple

# Import core modules
try:
    from pose_yolo import get_landmarks
    print('[PROFILER] Using YOLOv8 pose backend')
except Exception:
    from pose import get_landmarks
    print('[PROFILER] Using MoveNet pose backend')

from warp import (
    warp_image_perspective, 
    warp_image_affine, 
    warp_image_tps,
    set_warp_mode
)
from overlay import overlay
from overlay_skeleton import draw_skeleton


class Profiler:
    """
    Performance profiling tool for measuring frame timings.
    """
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.frame_count = 0
    
    def start(self, label: str):
        """Start timing a section."""
        return time.perf_counter()
    
    def end(self, label: str, start_time: float):
        """End timing a section and record."""
        elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
        self.timings[label].append(elapsed)
    
    def record_frame(self):
        """Increment frame counter."""
        self.frame_count += 1
    
    def get_stats(self, label: str) -> Dict[str, float]:
        """Get statistics for a specific timing label."""
        if label not in self.timings or not self.timings[label]:
            return {}
        
        times = self.timings[label]
        return {
            'mean': np.mean(times),
            'median': np.median(times),
            'min': np.min(times),
            'max': np.max(times),
            'std': np.std(times),
            'count': len(times)
        }
    
    def print_report(self):
        """Print comprehensive profiling report."""
        print("\n" + "="*70)
        print("PERFORMANCE PROFILING REPORT")
        print("="*70)
        print(f"Total Frames Processed: {self.frame_count}")
        print("-"*70)
        
        # Sort by mean time (slowest first)
        sorted_labels = sorted(
            self.timings.keys(),
            key=lambda l: np.mean(self.timings[l]),
            reverse=True
        )
        
        for label in sorted_labels:
            stats = self.get_stats(label)
            if not stats:
                continue
            
            print(f"\n{label}:")
            print(f"  Mean:   {stats['mean']:7.2f} ms")
            print(f"  Median: {stats['median']:7.2f} ms")
            print(f"  Min:    {stats['min']:7.2f} ms")
            print(f"  Max:    {stats['max']:7.2f} ms")
            print(f"  Std:    {stats['std']:7.2f} ms")
            
            # Calculate FPS if this were the bottleneck
            fps = 1000.0 / stats['mean'] if stats['mean'] > 0 else 0
            print(f"  → Equivalent FPS: {fps:.1f}")
        
        # Overall pipeline FPS
        if 'total_frame' in self.timings:
            total_stats = self.get_stats('total_frame')
            overall_fps = 1000.0 / total_stats['mean'] if total_stats['mean'] > 0 else 0
            print("\n" + "-"*70)
            print(f"OVERALL PIPELINE FPS: {overall_fps:.1f}")
        
        print("="*70 + "\n")


def benchmark_warp_mode(
    mode: str,
    num_frames: int = 100,
    garment_rgba=None,
    video_source: int = 0
) -> Profiler:
    """
    Benchmark a specific warp mode.
    
    Args:
        mode: 'perspective', 'affine', or 'tps'
        num_frames: Number of frames to process
        garment_rgba: Preloaded garment image
        video_source: Video source index
        
    Returns:
        Profiler object with collected statistics
    """
    print(f"\n[PROFILER] Benchmarking {mode.upper()} mode for {num_frames} frames...")
    
    cap = cv2.VideoCapture(video_source)
    profiler = Profiler()
    
    # Warm-up: process a few frames to initialize caches
    print("[PROFILER] Warming up (10 frames)...")
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = get_landmarks(frame)
        if landmarks and garment_rgba is not None:
            if mode == 'perspective':
                _ = warp_image_perspective(frame, garment_rgba, landmarks)
            elif mode == 'affine':
                _ = warp_image_affine(frame, garment_rgba, landmarks)
            elif mode == 'tps':
                _ = warp_image_tps(frame, garment_rgba, landmarks, 'shirt_front')
    
    # Actual benchmark
    print(f"[PROFILER] Running benchmark...")
    frames_processed = 0
    
    while frames_processed < num_frames:
        # Total frame time
        t_frame = profiler.start('total_frame')
        
        # 1. Capture
        t_cap = profiler.start('capture')
        ret, frame = cap.read()
        profiler.end('capture', t_cap)
        
        if not ret:
            print("[PROFILER] End of video stream")
            break
        
        # 2. Pose detection
        t_pose = profiler.start('pose_detection')
        landmarks = get_landmarks(frame)
        profiler.end('pose_detection', t_pose)
        
        if landmarks and garment_rgba is not None:
            # 3. Skeleton drawing
            t_skel = profiler.start('skeleton_draw')
            display = draw_skeleton(frame.copy(), landmarks)
            profiler.end('skeleton_draw', t_skel)
            
            # 4. Warping (the main focus)
            t_warp = profiler.start(f'warp_{mode}')
            if mode == 'perspective':
                warped = warp_image_perspective(display, garment_rgba, landmarks)
            elif mode == 'affine':
                warped = warp_image_affine(display, garment_rgba, landmarks)
            elif mode == 'tps':
                warped = warp_image_tps(display, garment_rgba, landmarks, 'shirt_front')
            profiler.end(f'warp_{mode}', t_warp)
            
            # 5. Overlay
            t_overlay = profiler.start('overlay')
            output = overlay(display, warped)
            profiler.end('overlay', t_overlay)
        
        profiler.end('total_frame', t_frame)
        profiler.record_frame()
        frames_processed += 1
        
        # Progress indicator
        if frames_processed % 10 == 0:
            print(f"  Processed {frames_processed}/{num_frames} frames...")
    
    cap.release()
    print(f"[PROFILER] Benchmark complete: {frames_processed} frames processed")
    
    return profiler


def compare_modes(
    num_frames: int = 100,
    garment_rgba=None,
    video_source: int = 0
):
    """
    Compare all warp modes side-by-side.
    
    Args:
        num_frames: Frames to process per mode
        garment_rgba: Preloaded garment
        video_source: Video source
    """
    modes = ['perspective', 'affine', 'tps']
    results = {}
    
    print("\n" + "="*70)
    print("COMPARING WARP MODES")
    print("="*70)
    
    for mode in modes:
        profiler = benchmark_warp_mode(mode, num_frames, garment_rgba, video_source)
        results[mode] = profiler
        profiler.print_report()
    
    # Comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Mode':<15} {'Mean (ms)':<12} {'FPS':<10} {'Speedup':<10}")
    print("-"*70)
    
    baseline_time = None
    for mode in modes:
        warp_stats = results[mode].get_stats(f'warp_{mode}')
        if warp_stats:
            mean_time = warp_stats['mean']
            fps = 1000.0 / mean_time if mean_time > 0 else 0
            
            if baseline_time is None:
                baseline_time = mean_time
                speedup = 1.0
            else:
                speedup = baseline_time / mean_time if mean_time > 0 else 0
            
            print(f"{mode:<15} {mean_time:<12.2f} {fps:<10.1f} {speedup:<10.2f}x")
    
    print("="*70 + "\n")


def main():
    """Main profiling entry point."""
    parser = argparse.ArgumentParser(description='Profile Virtual Try-On performance')
    parser.add_argument(
        '--mode', 
        type=str, 
        default='all',
        choices=['perspective', 'affine', 'tps', 'all'],
        help='Warp mode to benchmark'
    )
    parser.add_argument(
        '--frames',
        type=int,
        default=100,
        help='Number of frames to process'
    )
    parser.add_argument(
        '--video',
        type=int,
        default=0,
        help='Video source index (0 for default webcam)'
    )
    parser.add_argument(
        '--garment',
        type=str,
        default=None,
        help='Path to garment image (defaults to front_seg.png in project)'
    )
    
    args = parser.parse_args()
    
    # Determine garment path
    if args.garment:
        garment_path = args.garment
    else:
        # Use default path relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        garment_path = os.path.join(project_root, 'assests', 'garments', 'front_seg.png')
    
    # Load garment
    print(f"[PROFILER] Loading garment: {garment_path}")
    garment_rgba = cv2.imread(args.garment, cv2.IMREAD_UNCHANGED)
    if garment_rgba is None:
        print(f"ERROR: Could not load garment from {args.garment}")
        return
    
    # Run benchmark
    if args.mode == 'all':
        compare_modes(args.frames, garment_rgba, args.video)
    else:
        profiler = benchmark_warp_mode(args.mode, args.frames, garment_rgba, args.video)
        profiler.print_report()
    
    print("\n[PROFILER] Profiling complete!")
    print("\nOptimization Tips for RTX 4060:")
    print("- TPS mode benefits most from GPU (CuPy)")
    print("- If pose_detection is slow, consider reducing YOLOv8 input size")
    print("- If warp_tps is slow, try increasing downsample_factor in tps_warp.py")
    print("- For best real-time performance, aim for <33ms total_frame time (30 FPS)")


if __name__ == '__main__':
    main()
