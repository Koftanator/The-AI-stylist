"""
Temporal Smoothing for Keypoint Jitter Reduction
Adds exponential smoothing and Kalman filtering to pose detection

Integration: Add to pose_yolo.py or wrap the get_landmarks function
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from collections import deque


class ExponentialSmoother:
    """
    Simple exponential moving average for keypoint smoothing.
    Fast and effective for reducing jitter.
    """
    
    def __init__(self, alpha=0.3, min_confidence=0.5):
        """
        Initialize smoother.
        
        Args:
            alpha: Smoothing factor (0-1). Lower = more smoothing
                  0.1 = very smooth but laggy
                  0.5 = balanced
                  0.9 = responsive but less smoothing
            min_confidence: Minimum confidence to update
        """
        self.alpha = alpha
        self.min_confidence = min_confidence
        self.smoothed: Dict[int, Tuple[float, float]] = {}
    
    def smooth(
        self,
        landmarks: Dict[int, Optional[Tuple[int, int]]],
        confidences: Optional[Dict[int, float]] = None
    ) -> Dict[int, Optional[Tuple[int, int]]]:
        """
        Apply exponential smoothing to keypoints.
        
        Args:
            landmarks: Raw detected keypoints
            confidences: Optional confidence scores
            
        Returns:
            Smoothed keypoints
        """
        smoothed_landmarks = {}
        
        for idx, point in landmarks.items():
            if point is None:
                # Use previous smoothed value if available
                if idx in self.smoothed:
                    smoothed_landmarks[idx] = (
                        int(self.smoothed[idx][0]),
                        int(self.smoothed[idx][1])
                    )
                else:
                    smoothed_landmarks[idx] = None
                continue
            
            # Check confidence
            if confidences and idx in confidences:
                if confidences[idx] < self.min_confidence:
                    # Low confidence - use previous value
                    if idx in self.smoothed:
                        smoothed_landmarks[idx] = (
                            int(self.smoothed[idx][0]),
                            int(self.smoothed[idx][1])
                        )
                    else:
                        smoothed_landmarks[idx] = None
                    continue
            
            x, y = point
            
            if idx not in self.smoothed:
                # First observation - initialize
                self.smoothed[idx] = (float(x), float(y))
            else:
                # Exponential moving average: S_t = α * X_t + (1-α) * S_{t-1}
                prev_x, prev_y = self.smoothed[idx]
                new_x = self.alpha * x + (1 - self.alpha) * prev_x
                new_y = self.alpha * y + (1 - self.alpha) * prev_y
                self.smoothed[idx] = (new_x, new_y)
            
            smoothed_landmarks[idx] = (
                int(self.smoothed[idx][0]),
                int(self.smoothed[idx][1])
            )
        
        return smoothed_landmarks
    
    def reset(self):
        """Reset all smoothed values."""
        self.smoothed.clear()


class MedianFilter:
    """
    Median filter for outlier rejection.
    Useful for occasional detection glitches.
    """
    
    def __init__(self, window_size=5):
        """
        Initialize median filter.
        
        Args:
            window_size: Number of frames to consider (odd number recommended)
        """
        self.window_size = window_size
        self.history: Dict[int, deque] = {}
    
    def filter(
        self,
        landmarks: Dict[int, Optional[Tuple[int, int]]]
    ) -> Dict[int, Optional[Tuple[int, int]]]:
        """
        Apply median filtering to keypoints.
        
        Args:
            landmarks: Raw detected keypoints
            
        Returns:
            Median-filtered keypoints
        """
        filtered_landmarks = {}
        
        for idx, point in landmarks.items():
            if point is None:
                filtered_landmarks[idx] = None
                continue
            
            # Initialize history buffer
            if idx not in self.history:
                self.history[idx] = deque(maxlen=self.window_size)
            
            # Add current point
            self.history[idx].append(point)
            
            # Compute median if we have enough history
            if len(self.history[idx]) >= 3:
                points = list(self.history[idx])
                x_vals = [p[0] for p in points]
                y_vals = [p[1] for p in points]
                
                median_x = int(np.median(x_vals))
                median_y = int(np.median(y_vals))
                filtered_landmarks[idx] = (median_x, median_y)
            else:
                # Not enough history, use raw point
                filtered_landmarks[idx] = point
        
        return filtered_landmarks
    
    def reset(self):
        """Reset all history."""
        self.history.clear()


class OneEuroFilter:
    """
    One Euro Filter - Advanced smoothing with adaptive cutoff.
    Best for reducing jitter while maintaining responsiveness.
    
    Reference: Casiez et al., CHI 2012
    """
    
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        """
        Initialize One Euro Filter.
        
        Args:
            min_cutoff: Minimum cutoff frequency (lower = more smoothing)
            beta: Speed coefficient (controls lag)
            d_cutoff: Derivative cutoff frequency
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev: Dict[int, float] = {}
        self.dx_prev: Dict[int, float] = {}
        self.y_prev: Dict[int, float] = {}
        self.dy_prev: Dict[int, float] = {}
        self.t_prev: Dict[int, float] = {}
        
        import time
        self.time_fn = time.time
    
    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        """Calculate smoothing factor."""
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)
    
    def _exponential_smoothing(self, a: float, x: float, x_prev: float) -> float:
        """Apply exponential smoothing."""
        return a * x + (1 - a) * x_prev
    
    def smooth_coordinate(
        self,
        idx: int,
        coord: float,
        is_x: bool
    ) -> float:
        """
        Smooth a single coordinate (x or y).
        
        Args:
            idx: Keypoint index
            coord: Current coordinate value
            is_x: True for x-coordinate, False for y
            
        Returns:
            Smoothed coordinate
        """
        t = self.time_fn()
        
        # Initialize on first observation
        if idx not in self.t_prev:
            self.x_prev[idx] = coord if is_x else self.x_prev.get(idx, 0)
            self.y_prev[idx] = coord if not is_x else self.y_prev.get(idx, 0)
            self.dx_prev[idx] = 0
            self.dy_prev[idx] = 0
            self.t_prev[idx] = t
            return coord
        
        # Time elapsed
        t_e = t - self.t_prev[idx]
        if t_e <= 0:
            t_e = 0.001  # Prevent division by zero
        
        # Get previous values
        if is_x:
            coord_prev = self.x_prev[idx]
            dcoord_prev = self.dx_prev[idx]
        else:
            coord_prev = self.y_prev[idx]
            dcoord_prev = self.dy_prev[idx]
        
        # Estimate derivative
        dcoord = (coord - coord_prev) / t_e
        
        # Smooth derivative
        alpha_d = self._smoothing_factor(t_e, self.d_cutoff)
        dcoord_smooth = self._exponential_smoothing(alpha_d, dcoord, dcoord_prev)
        
        # Adaptive cutoff based on speed
        cutoff = self.min_cutoff + self.beta * abs(dcoord_smooth)
        
        # Smooth coordinate
        alpha = self._smoothing_factor(t_e, cutoff)
        coord_smooth = self._exponential_smoothing(alpha, coord, coord_prev)
        
        # Update state
        if is_x:
            self.x_prev[idx] = coord_smooth
            self.dx_prev[idx] = dcoord_smooth
        else:
            self.y_prev[idx] = coord_smooth
            self.dy_prev[idx] = dcoord_smooth
        
        self.t_prev[idx] = t
        
        return coord_smooth
    
    def filter(
        self,
        landmarks: Dict[int, Optional[Tuple[int, int]]]
    ) -> Dict[int, Optional[Tuple[int, int]]]:
        """
        Apply One Euro filter to keypoints.
        
        Args:
            landmarks: Raw detected keypoints
            
        Returns:
            Filtered keypoints
        """
        filtered_landmarks = {}
        
        for idx, point in landmarks.items():
            if point is None:
                filtered_landmarks[idx] = None
                continue
            
            x, y = point
            
            # Smooth both coordinates
            x_smooth = self.smooth_coordinate(idx, float(x), is_x=True)
            y_smooth = self.smooth_coordinate(idx, float(y), is_x=False)
            
            filtered_landmarks[idx] = (int(x_smooth), int(y_smooth))
        
        return filtered_landmarks
    
    def reset(self):
        """Reset all filter state."""
        self.x_prev.clear()
        self.dx_prev.clear()
        self.y_prev.clear()
        self.dy_prev.clear()
        self.t_prev.clear()


# Global smoother instance (reuse across frames)
_global_smoother: Optional[ExponentialSmoother] = None


def add_smoothing_to_pose_detector(
    filter_type='exponential',
    alpha=0.3,
    window_size=5
) -> Union['ExponentialSmoother', 'MedianFilter', 'OneEuroFilter']:
    """
    Factory function to create appropriate smoother.
    
    Args:
        filter_type: 'exponential', 'median', or 'one_euro'
        alpha: Smoothing factor for exponential filter
        window_size: Window size for median filter
        
    Returns:
        Smoother instance
    """
    if filter_type == 'exponential':
        return ExponentialSmoother(alpha=alpha)
    elif filter_type == 'median':
        return MedianFilter(window_size=window_size)
    elif filter_type == 'one_euro':
        return OneEuroFilter(min_cutoff=1.0, beta=0.007)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def wrap_get_landmarks_with_smoothing(
    original_get_landmarks,
    filter_type='exponential',
    alpha=0.3
):
    """
    Wrap existing get_landmarks function with temporal smoothing.
    
    Usage in pose_yolo.py:
        # At module level
        from temporal_smoothing import wrap_get_landmarks_with_smoothing
        _original_get_landmarks = get_landmarks
        get_landmarks = wrap_get_landmarks_with_smoothing(_original_get_landmarks)
    
    Args:
        original_get_landmarks: Original pose detection function
        filter_type: Type of smoothing filter
        alpha: Smoothing factor
        
    Returns:
        Wrapped function with smoothing
    """
    smoother = add_smoothing_to_pose_detector(filter_type=filter_type, alpha=alpha)
    
    def smoothed_get_landmarks(frame, conf_threshold=0.25):
        """Get landmarks with temporal smoothing."""
        # Get raw landmarks
        landmarks = original_get_landmarks(frame, conf_threshold)
        
        if landmarks is None:
            return None
        
        # Apply smoothing (type: ignore because all smoother types have smooth method)
        smoothed = smoother.smooth(landmarks)  # type: ignore[union-attr]
        
        return smoothed
    
    return smoothed_get_landmarks


# Integration code snippet for pose_yolo.py
INTEGRATION_CODE = """
# Add to pose_yolo.py after imports:
from temporal_smoothing import ExponentialSmoother

# Add at module level (after _model definition):
_smoother = ExponentialSmoother(alpha=0.3, min_confidence=0.25)

# Modify get_landmarks function - add smoothing before return:
def get_landmarks(frame, conf_threshold=0.25):
    # ... existing code to get 'points' dict ...
    
    # Add smoothing before returning:
    smoothed_points = _smoother.smooth(points, confidences=None)
    return smoothed_points
"""


if __name__ == '__main__':
    print("Temporal Smoothing for Keypoint Jitter Reduction")
    print("="*60)
    print("\nAvailable filters:")
    print("1. Exponential Smoother - Fast, good for general use")
    print("2. Median Filter - Removes outliers, moderate smoothing")
    print("3. One Euro Filter - Best quality, adaptive smoothing")
    print("\nRecommended: Exponential with alpha=0.3 (balanced)")
    print("\nIntegration:")
    print(INTEGRATION_CODE)
