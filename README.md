Virtual Try-On System
A Python-based virtual try-on application that uses pose estimation and image warping to overlay garments onto a person's body in real-time or from static images.

Features
Pose Detection: Uses YOLOv8 pose estimation model to detect human body keypoints
Garment Overlay: Warps and overlays garments (front/back views) onto detected body pose
Image Segmentation: Removes backgrounds from garment images for clean overlay
Skeleton Visualization: Option to display detected pose skeleton for debugging
Perspective Transformation: Accurately maps garments to body proportions using shoulder and hip landmarks
