"""
GPU-Accelerated Thin-Plate Spline (TPS) Warping Module
Optimized for NVIDIA RTX 4060 GPU with real-time performance

This module implements TPS warping with the following optimizations:
1. CuPy/CUDA acceleration for matrix operations
2. Intelligent caching: recompute TPS coefficients only when keypoints shift significantly
3. Resolution downscaling for warping, then upscaling to display resolution
4. Vectorized operations to maximize GPU throughput
5. Memory-efficient computation reusing GPU buffers

TPS Theory:
Thin-Plate Spline provides smooth, non-linear warping that minimizes bending energy.
Given source control points and target control points, TPS computes a transformation
that smoothly interpolates between them, ideal for organic garment fitting.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
import warnings

# Try to import CuPy for CUDA acceleration
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    CUPY_AVAILABLE = True
    print("[TPS] CuPy detected - GPU acceleration enabled")
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
    print("[TPS] CuPy not available - falling back to CPU (install cupy-cuda12x for GPU acceleration)")

from scipy.spatial.distance import cdist
from scipy.linalg import lstsq


class TPSWarper:
    """
    Thin-Plate Spline warping with GPU acceleration and intelligent caching.
    
    Key Features for RTX 4060 Performance:
    - CUDA-accelerated grid transformations via CuPy
    - Cached TPS coefficients (recompute only when keypoints change)
    - Downscale warping resolution to reduce computation
    - Reusable GPU memory buffers
    """
    
    def __init__(
        self,
        downsample_factor: float = 0.5,
        movement_threshold: float = 5.0,
        use_gpu: bool = True,
        regularization: float = 0.0
    ):
        """
        Initialize TPS Warper.
        
        Args:
            downsample_factor: Scale factor for warping (0.5 = half resolution, 2x speedup)
            movement_threshold: Min pixel distance to trigger TPS recomputation (px)
            use_gpu: Enable GPU acceleration if CuPy available
            regularization: TPS regularization parameter (0.0 = exact interpolation)
        """
        self.downsample_factor = downsample_factor
        self.movement_threshold = movement_threshold
        self.regularization = regularization
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        
        # Cache for TPS coefficients to avoid redundant computation
        self._cached_src_pts: Optional[np.ndarray] = None
        self._cached_dst_pts: Optional[np.ndarray] = None
        self._cached_weights: Optional[np.ndarray] = None
        
        # GPU memory buffers (allocated once, reused)
        if self.use_gpu:
            self._gpu_grid_x: Optional[cp.ndarray] = None
            self._gpu_grid_y: Optional[cp.ndarray] = None
            self._gpu_map_x: Optional[cp.ndarray] = None
            self._gpu_map_y: Optional[cp.ndarray] = None
    
    def _compute_tps_weights(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray
    ) -> np.ndarray:
        """
        Compute TPS transformation weights (CPU-based, but only called when needed).
        
        This is the mathematical core of TPS but runs infrequently due to caching.
        
        Args:
            src_pts: Source control points (N, 2)
            dst_pts: Destination control points (N, 2)
            
        Returns:
            Weights array for TPS transformation
        """
        n = src_pts.shape[0]
        
        # Build TPS kernel matrix K
        # K[i,j] = U(||p_i - p_j||) where U(r) = r^2 * log(r^2) for r > 0
        K = cdist(src_pts, src_pts, metric='euclidean')
        
        # Apply TPS radial basis function
        # Avoid log(0) by masking diagonal
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            K = np.where(K == 0, 0, K**2 * np.log(K**2 + 1e-10))
        
        # Add regularization to diagonal (smoothing parameter)
        if self.regularization > 0:
            K += self.regularization * np.eye(n)
        
        # Build P matrix [1, x, y] for affine component
        P = np.hstack([np.ones((n, 1)), src_pts])
        
        # Construct full system matrix (n+3) x (n+3)
        zeros = np.zeros((3, 3))
        L = np.vstack([
            np.hstack([K, P]),
            np.hstack([P.T, zeros])
        ])
        
        # Right-hand side: destination coordinates + zero constraints
        V = np.vstack([dst_pts, np.zeros((3, 2))])
        
        # Solve linear system L * weights = V
        # This computes weights for both x and y coordinates
        weights, _, _, _ = lstsq(L, V)
        
        return weights
    
    def _should_recompute(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray
    ) -> bool:
        """
        Determine if TPS weights need recomputation based on keypoint movement.
        
        Optimization: Only recompute when keypoints move beyond threshold.
        This is critical for real-time performance - TPS weight computation is
        the most expensive operation (~20-50ms on CPU), but can be amortized
        over many frames when the user is relatively still.
        
        Args:
            src_pts: Current source control points
            dst_pts: Current destination control points
            
        Returns:
            True if recomputation needed
        """
        if self._cached_weights is None:
            return True
        
        if self._cached_src_pts is None or self._cached_dst_pts is None:
            return True
        
        # Check if points have moved significantly
        src_diff = np.max(np.abs(src_pts - self._cached_src_pts))
        dst_diff = np.max(np.abs(dst_pts - self._cached_dst_pts))
        
        return (src_diff > self.movement_threshold or 
                dst_diff > self.movement_threshold)
    
    def _apply_tps_transform(
        self,
        grid_shape: Tuple[int, int],
        src_pts: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply TPS transformation to generate dense warp grid.
        
        This is GPU-accelerated when CuPy is available.
        
        RTX 4060 Optimization:
        - Uses CuPy for parallel distance computations across all grid points
        - Vectorized operations leverage CUDA cores (3072 on RTX 4060)
        - Grid is downsampled to reduce computations
        
        Args:
            grid_shape: (height, width) of output grid
            src_pts: Source control points used in weight computation
            weights: TPS weights from _compute_tps_weights
            
        Returns:
            (map_x, map_y) remap grids for cv2.remap
        """
        h, w = grid_shape
        n_pts = src_pts.shape[0]
        
        if self.use_gpu:
            # GPU path using CuPy
            # Allocate or reuse GPU grid buffers
            if (self._gpu_grid_x is None or 
                self._gpu_grid_x.shape != (h, w)):
                self._gpu_grid_x = cp.arange(w, dtype=cp.float32)
                self._gpu_grid_y = cp.arange(h, dtype=cp.float32)
                self._gpu_grid_x, self._gpu_grid_y = cp.meshgrid(
                    self._gpu_grid_x, self._gpu_grid_y
                )
            
            grid_x = self._gpu_grid_x
            grid_y = self._gpu_grid_y
            
            # Flatten grid for vectorized operations
            grid_pts = cp.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
            
            # Transfer source points and weights to GPU
            src_pts_gpu = cp.asarray(src_pts, dtype=cp.float32)
            weights_gpu = cp.asarray(weights, dtype=cp.float32)
            
            # Compute distances from each grid point to each control point
            # Shape: (h*w, n_pts)
            dists = cp.linalg.norm(
                grid_pts[:, cp.newaxis, :] - src_pts_gpu[cp.newaxis, :, :],
                axis=2
            )
            
            # Apply TPS radial basis function
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                U = cp.where(dists == 0, 0, dists**2 * cp.log(dists**2 + 1e-10))
            
            # Add affine component [1, x, y]
            affine = cp.hstack([
                cp.ones((grid_pts.shape[0], 1), dtype=cp.float32),
                grid_pts
            ])
            
            # Combine TPS and affine: [U | 1 x y] * weights
            basis = cp.hstack([U, affine])
            
            # Matrix multiply to get warped coordinates
            warped = cp.dot(basis, weights_gpu)
            
            # Reshape back to grid
            map_x = warped[:, 0].reshape(h, w)
            map_y = warped[:, 1].reshape(h, w)
            
            # Transfer back to CPU for cv2.remap (OpenCV doesn't support CuPy arrays directly)
            # For ultimate performance, could use cv2.cuda.remap, but requires UMat conversion
            map_x = cp.asnumpy(map_x).astype(np.float32)
            map_y = cp.asnumpy(map_y).astype(np.float32)
            
        else:
            # CPU fallback path
            grid_x, grid_y = np.meshgrid(
                np.arange(w, dtype=np.float32),
                np.arange(h, dtype=np.float32)
            )
            grid_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
            
            # Compute distances
            dists = cdist(grid_pts, src_pts, metric='euclidean')
            
            # TPS radial basis
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                U = np.where(dists == 0, 0, dists**2 * np.log(dists**2 + 1e-10))
            
            # Affine component
            affine = np.hstack([np.ones((grid_pts.shape[0], 1)), grid_pts])
            
            # Combine and transform
            basis = np.hstack([U, affine])
            warped = np.dot(basis, weights)
            
            map_x = warped[:, 0].reshape(h, w).astype(np.float32)
            map_y = warped[:, 1].reshape(h, w).astype(np.float32)
        
        return map_x, map_y
    
    def warp(
        self,
        image: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        output_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Warp image using Thin-Plate Spline transformation.
        
        Main entry point for TPS warping with all optimizations enabled.
        
        Args:
            image: Input image (H, W, C) - typically RGBA garment
            src_pts: Source control points (N, 2) - garment keypoints
            dst_pts: Target control points (N, 2) - user body keypoints
            output_shape: Output image shape (H, W), defaults to input shape
            
        Returns:
            Warped image with same shape as output_shape or input
        """
        if image is None or src_pts is None or dst_pts is None:
            return image if image is not None else np.zeros((100, 100, 4), dtype=np.uint8)
        
        if len(src_pts) < 3 or len(dst_pts) < 3:
            # Need at least 3 points for TPS
            return image
        
        # Validate point arrays
        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)
        
        if src_pts.shape != dst_pts.shape:
            raise ValueError("Source and destination points must have same shape")
        
        # Output shape defaults to input image shape
        if output_shape is None:
            output_shape = (image.shape[0], image.shape[1])
        
        # Optimization 1: Downscale for warping (major speedup on RTX 4060)
        # Compute at lower resolution, then upscale
        warp_h = int(output_shape[0] * self.downsample_factor)
        warp_w = int(output_shape[1] * self.downsample_factor)
        
        # Scale control points to match downsampled grid
        scale_x = warp_w / output_shape[1]
        scale_y = warp_h / output_shape[0]
        dst_pts_scaled = dst_pts.copy()
        dst_pts_scaled[:, 0] *= scale_x
        dst_pts_scaled[:, 1] *= scale_y
        
        # Scale garment dimensions
        src_h, src_w = image.shape[:2]
        src_pts_scaled = src_pts.copy()
        src_pts_scaled[:, 0] *= (warp_w / output_shape[1])
        src_pts_scaled[:, 1] *= (warp_h / output_shape[0])
        
        # Optimization 2: Cache TPS weights - only recompute when needed
        if self._should_recompute(src_pts_scaled, dst_pts_scaled):
            self._cached_weights = self._compute_tps_weights(
                src_pts_scaled, dst_pts_scaled
            )
            self._cached_src_pts = src_pts_scaled.copy()
            self._cached_dst_pts = dst_pts_scaled.copy()
        
        # Generate dense warp grid using cached weights
        map_x, map_y = self._apply_tps_transform(
            (warp_h, warp_w),
            src_pts_scaled,
            self._cached_weights
        )
        
        # Map back to original image coordinates
        map_x *= (image.shape[1] / warp_w)
        map_y *= (image.shape[0] / warp_h)
        
        # Apply warping using OpenCV's efficient remap
        warped = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0) if image.shape[2] == 4 else (0, 0, 0)
        )
        
        # Upscale to target resolution if needed
        if (warped.shape[0] != output_shape[0] or 
            warped.shape[1] != output_shape[1]):
            warped = cv2.resize(
                warped,
                (output_shape[1], output_shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        return warped
    
    def reset_cache(self):
        """Clear cached TPS weights. Call when switching garments or major scene change."""
        self._cached_src_pts = None
        self._cached_dst_pts = None
        self._cached_weights = None
    
    def get_stats(self) -> dict:
        """Return current warper statistics for profiling."""
        return {
            'gpu_enabled': self.use_gpu,
            'downsample_factor': self.downsample_factor,
            'cache_valid': self._cached_weights is not None,
            'movement_threshold': self.movement_threshold
        }


def create_tps_warper(
    fast_mode: bool = True,
    gpu_enabled: bool = True
) -> TPSWarper:
    """
    Factory function to create optimally configured TPS warper for RTX 4060.
    
    Args:
        fast_mode: If True, use aggressive optimizations (lower resolution)
        gpu_enabled: Enable GPU acceleration
        
    Returns:
        Configured TPSWarper instance
    """
    if fast_mode:
        # Optimized for 30+ FPS on RTX 4060
        return TPSWarper(
            downsample_factor=0.5,  # Half resolution = 4x speedup
            movement_threshold=3.0,  # Sensitive to movement
            use_gpu=gpu_enabled,
            regularization=0.001  # Slight smoothing
        )
    else:
        # Higher quality, still real-time
        return TPSWarper(
            downsample_factor=0.75,
            movement_threshold=5.0,
            use_gpu=gpu_enabled,
            regularization=0.0
        )
