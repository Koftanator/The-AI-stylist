# 🎽 Virtual Try-On System with TPS Warping

A real-time 2D virtual try-on application using YOLOv8 pose detection and GPU-accelerated Thin-Plate Spline (TPS) warping for realistic garment fitting.

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.13-green)
![CUDA](https://img.shields.io/badge/CUDA-12.x-orange)
![Status](https://img.shields.io/badge/status-production-success)

---

## ✨ Features

- **Real-time Pose Detection**: YOLOv8-pose with 17 COCO keypoints
- **GPU-Accelerated Warping**: CuPy-based TPS transformation (~30-35 FPS)
- **Multiple Warp Modes**: Perspective, Affine, and TPS
- **Body Masking**: Smart garment overlay with multiple mask modes
- **Temporal Smoothing**: Reduces keypoint jitter for stable rendering
- **Debug Visualization**: Real-time diagnostic overlay with confidence scoring
- **Enhanced Control Points**: 31 points for front view, 24 for back view

---

## 🎯 Key Improvements (v2.0)

✅ **Temporal Smoothing** - Exponential filter reduces frame-to-frame jitter  
✅ **Debug Mode** - Press 'd' to visualize keypoints and control points  
✅ **31 Control Points** - Enhanced from 15 for better garment fit  
✅ **Stable Sleeve Mapping** - Sleeves map to elbows (not unstable wrists)  
✅ **Comprehensive Docs** - Troubleshooting guide and integration instructions

---

## 📁 Project Structure

```
test2-1/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── yolov8n-pose.pt             # YOLOv8 pose model weights
│
├── docs/                        # Documentation
│   ├── FIXES_APPLIED.md         # Summary of v2.0 improvements
│   ├── TROUBLESHOOTING_GUIDE.md # Symptom-to-fix mappings
│   ├── INTEGRATION_INSTRUCTIONS.md # Step-by-step setup guide
│   ├── ARCHITECTURE.md          # System architecture
│   ├── INSTALLATION.md          # Installation guide
│   └── ... (other docs)
│
├── assests/                     # Garment images
│   └── garments/
│       ├── front.png           # Front-view garment (RGB)
│       ├── front_seg.png       # Front-view with alpha channel
│       ├── back.png            # Back-view garment (RGB)
│       └── back_seg.png        # Back-view with alpha channel
│
└── src/                        # Source code
    ├── main.py                 # Main application entry point
    │
    ├── pose_yolo.py           # YOLOv8 pose detection (with smoothing)
    ├── pose.py                # Fallback pose detector
    │
    ├── warp.py                # Warping interface (TPS/Affine/Perspective)
    ├── tps_warp.py            # GPU-accelerated TPS implementation
    │
    ├── garment_mapping.py     # Control point mappings (31 points)
    ├── overlay.py             # Garment overlay with body masking
    ├── overlay_skeleton.py    # Skeleton visualization
    ├── skeleton.py            # Skeleton drawing utilities
    │
    ├── debug_visualizer.py    # Debug mode visualization
    ├── temporal_smoothing.py  # Keypoint smoothing filters
    │
    ├── segment.py             # Garment segmentation
    │
    ├── utils/                 # Utility scripts
    │   ├── analyze_fit.py     # Fit analysis tools
    │   ├── analyze_garment.py # Garment analysis
    │   ├── check_tps.py       # TPS validation
    │   ├── debug_tps_visual.py # TPS debugging
    │   ├── test_tps.py        # TPS unit tests
    │   ├── profiler.py        # Performance profiling
    │   └── ...
    │
    └── tflite-2.18.0/         # TFLite library (optional)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10+ (tested on 3.13)
- **GPU**: NVIDIA GPU with CUDA 12.x (RTX 4060 recommended)
- **OS**: Linux (tested on Ubuntu)
- **RAM**: 8GB+ recommended
- **Webcam**: For real-time try-on

### Installation

```bash
# Clone or navigate to project
cd /path/to/test2-1

# Install dependencies
pip install -r requirements.txt

# Verify YOLOv8 model exists
ls yolov8n-pose.pt  # Should exist in project root
```

### Run Application

```bash
cd src
python main.py
```

**Expected Output**:
```
[MAIN] Using YOLOv8 pose backend
============================================================
2D Virtual Try-On with TPS Warping
============================================================
Starting in: TPS mode
Controls:
  q: Quit
  t: Toggle front/back view
  m: Cycle warp modes (Perspective -> Affine -> TPS)
  b: Cycle mask modes (Body -> Torso -> Head)
  r: Reset TPS cache
  f: Toggle FPS display
  d: Toggle debug mode (keypoint + control point visualization)
...
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `t` | Toggle front/back garment view |
| `m` | Cycle warp modes (Perspective → Affine → TPS) |
| `b` | Cycle mask modes (Body → Torso → Head) |
| `r` | Reset TPS cache (if garment glitches) |
| `f` | Toggle FPS counter display |
| **`d`** | **Toggle debug mode (NEW!)** |

### Debug Mode Features

Press `d` to enable debug visualization:

- 🟢 **Green circles**: High confidence keypoints (> 0.7)
- 🟡 **Yellow circles**: Medium confidence (0.5-0.7)
- 🔴 **Red circles**: Low confidence (< 0.5)
- 🔵 **Blue lines**: Control point mappings (garment → body)
- 📊 **Text overlay**: TPS metrics, point counts, condition number
- 💬 **Console output**: Detailed diagnostics

---

## 🎽 Adding Custom Garments

### Step 1: Prepare Garment Images

Required format:
- **Resolution**: 512x512 or higher (square recommended)
- **Format**: PNG with alpha channel (RGBA)
- **Background**: Transparent (alpha = 0)
- **Orientation**: Flat, front-facing or back-facing

### Step 2: Add to Assets

```bash
# Place your garment images in:
assests/garments/
├── my_shirt_front_seg.png  # Front view with transparency
└── my_shirt_back_seg.png   # Back view with transparency
```

### Step 3: Update main.py

```python
# In main.py, update garment paths:
front_rgba = cv2.imread('../assests/garments/my_shirt_front_seg.png', cv2.IMREAD_UNCHANGED)
back_rgba = cv2.imread('../assests/garments/my_shirt_back_seg.png', cv2.IMREAD_UNCHANGED)
```

### Step 4: (Optional) Adjust Control Points

If your garment has different proportions, adjust normalized coordinates in `garment_mapping.py`:

```python
control_points_normalized = {
    'neck_center': (0.475, 0.086),  # x, y in 0-1 range
    'shoulder_left': (0.071, 0.255),
    # ... customize as needed
}
```

---

## ⚙️ Configuration

### Temporal Smoothing

Adjust responsiveness in `pose_yolo.py`:

```python
_smoother = ExponentialSmoother(alpha=0.3, min_confidence=0.25)
# alpha=0.2: More smooth, higher lag
# alpha=0.3: Balanced (recommended)
# alpha=0.5: More responsive, less smooth
```

### TPS Parameters

Modify TPS behavior in `tps_warp.py`:

```python
warper = TPSWarper(
    regularization=0.0,       # 0.0 = exact interpolation, 0.001 = smoother
    movement_threshold=10.0,  # Cache invalidation threshold (pixels)
    downsample_factor=2       # Reduce resolution for speed (1=full res)
)
```

### Body Masking

Choose mask mode for different garment types:
- **Body**: Full upper body (best for t-shirts)
- **Torso**: Shoulders + hips only (good for tank tops)
- **Head**: Face region (for hats/accessories)

---

## 📊 Performance

### Benchmarks (RTX 4060, 1080p webcam)

| Configuration | FPS | Quality |
|---------------|-----|---------|
| Perspective mode | ~45-50 | ⭐⭐ |
| Affine mode | ~40-45 | ⭐⭐⭐ |
| **TPS mode** | **~30-35** | ⭐⭐⭐⭐⭐ |
| TPS + Debug mode | ~25-30 | ⭐⭐⭐⭐⭐ |

### Optimization Tips

1. **Lower resolution**: Reduce webcam resolution to 720p
2. **Increase downsample**: Set `downsample_factor=4` in TPS
3. **Disable debug mode**: Only use 'd' for diagnostics
4. **Reduce control points**: Use fewer points for simpler garments

---

## 🐛 Troubleshooting

### Common Issues

#### "No pose detected"
- **Cause**: Person not visible or too far from camera
- **Fix**: Move closer, ensure good lighting, check webcam

#### "Sleeves don't follow arms"
- **Cause**: Low elbow keypoint confidence
- **Fix**: Improve lighting, keep arms visible, check debug mode (press 'd')
- **Reference**: See `docs/TROUBLESHOOTING_GUIDE.md` section "Sleeves Don't Follow Arms"

#### "Garment appears distorted"
- **Cause**: Poor TPS conditioning, too few control points
- **Fix**: Check condition number in debug mode (should be < 1e10)
- **Reference**: See `docs/TROUBLESHOOTING_GUIDE.md` section "TPS Warping Produces Distorted Results"

#### "Application slow/laggy"
- **Cause**: High resolution, too many control points
- **Fix**: Reduce downsample_factor, lower webcam resolution
- **Performance Guide**: See `docs/FIXES_APPLIED.md` section "Performance Impact"

### Debug Checklist

Run debug mode and verify:

```bash
# 1. Check keypoints detected
python main.py
# Press 'd' and observe:
# - All 6 critical keypoints (shoulders, elbows, hips) should be green/yellow
# - Confidence values should be > 0.5

# 2. Validate TPS condition
# In debug mode, check console output:
# - Condition number should be < 1e10
# - Min distance between points should be > 10px

# 3. Test temporal smoothing
# Observe garment stability:
# - No excessive jitter or jumping
# - Smooth transitions when moving
```

**Full troubleshooting guide**: See `docs/TROUBLESHOOTING_GUIDE.md`

---

## 📚 Documentation

Comprehensive documentation available in `docs/`:

- **FIXES_APPLIED.md** - Complete changelog for v2.0
- **TROUBLESHOOTING_GUIDE.md** - Symptom-to-fix mappings with code examples
- **INTEGRATION_INSTRUCTIONS.md** - Step-by-step integration guide
- **ARCHITECTURE.md** - System architecture and design decisions
- **INSTALLATION.md** - Detailed installation instructions

---

## 🔧 Development

### Running Tests

```bash
cd src/utils

# Test TPS warping
python test_tps.py

# Validate TPS numerics
python check_tps.py

# Analyze garment fit
python analyze_fit.py
```

### Profiling Performance

```bash
cd src/utils
python profiler.py
```

### Debugging TPS

```bash
cd src/utils
python debug_tps_visual.py  # Visual TPS debugging
```

---

## 🤝 Contributing

To add features or fix bugs:

1. **Test thoroughly**: Use debug mode to validate changes
2. **Update docs**: Reflect changes in relevant .md files
3. **Profile performance**: Ensure FPS remains > 25
4. **Check alignment**: Press 'd' to verify control points

---

## 📝 Technical Details

### Pose Detection
- **Model**: YOLOv8n-pose (nano variant)
- **Keypoints**: 17 COCO format (0=nose, 5/6=shoulders, 7/8=elbows, 11/12=hips)
- **Confidence threshold**: 0.25 (adjustable)
- **Smoothing**: Exponential filter (alpha=0.3)

### TPS Warping
- **Algorithm**: Thin-Plate Spline interpolation
- **Implementation**: GPU-accelerated with CuPy
- **Control points**: 31 (front), 24 (back)
- **Caching**: Movement threshold = 10px
- **Regularization**: 0.0 (exact interpolation)

### Control Point Mapping
- **Collar/Neck**: Interpolated from shoulders + nose
- **Shoulders**: Direct mapping to COCO keypoints 5, 6
- **Chest**: Interpolated (30% down from shoulders)
- **Waist**: Interpolated (65% down from shoulders)
- **Hips**: Direct mapping to COCO keypoints 11, 12
- **Sleeves**: **Map to ELBOWS (7, 8), NOT wrists (9, 10)**

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics YOLO for pose detection
- **CuPy**: GPU acceleration library
- **OpenCV**: Computer vision operations

---

## 📞 Support

For issues or questions:

1. Check `docs/TROUBLESHOOTING_GUIDE.md`
2. Enable debug mode (`d` key) and check console output
3. Review `docs/FIXES_APPLIED.md` for known issues

---

## 🎉 Quick Reference

```bash
# Start application
python src/main.py

# Enable debug mode
Press 'd' during runtime

# Switch garment view
Press 't' for front/back

# Change warp mode
Press 'm' to cycle (recommend TPS for best quality)

# Reset if glitchy
Press 'r' to reset TPS cache
```

**Enjoy your virtual try-on system! 🎽✨**

---

**Version**: 2.0  
**Last Updated**: November 14, 2025  
**Status**: ✅ Production Ready
