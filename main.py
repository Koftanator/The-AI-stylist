import cv2
from warp import warp_image
from overlay import overlay
from overlay_skeleton import draw_skeleton

# Prefer YOLOv8 pose backend if available, otherwise fall back to existing pose
try:
    from pose_yolo import get_landmarks
    print('Using YOLOv8 pose backend')
except Exception:
    from pose import get_landmarks
    print('Using MoveNet pose backend')

# Preload segmented garments (note: repository folder is `assests/garments`)
front_rgba = cv2.imread('../assests/garments/front_seg.png', cv2.IMREAD_UNCHANGED)
back_rgba  = cv2.imread('../assests/garments/back_seg.png',  cv2.IMREAD_UNCHANGED)

# Validate loaded garments
if front_rgba is None:
    print("WARNING: front_seg.png not found or failed to load at 'assests/garments/front_seg.png'")
if back_rgba is None:
    print("WARNING: back_seg.png not found or failed to load at 'assests/garments/back_seg.png'")

cap = cv2.VideoCapture(0)
mode = 'front'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = get_landmarks(frame)
    display = frame.copy()

    if landmarks:
        # 1. Draw skeleton
        display = draw_skeleton(display, landmarks)

        # 2. Warp chosen garment
        garment = front_rgba if mode=='front' else back_rgba
        warped = warp_image(display, garment, landmarks)

        # 3. Overlay garment
        output = overlay(display, warped)
    else:
        output = display
        # show why no pose was applied
        cv2.putText(output, 'Pose disabled or not detected', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.putText(output, f'Viewing: {mode}', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow('2D Try-On with Skeleton', output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('t'):
        mode = 'back' if mode=='front' else 'front'

cap.release()
cv2.destroyAllWindows()
