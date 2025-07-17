from ultralytics import YOLO
import cv2
import time
from collections import deque
import numpy as np
import logging

# --- Configuration Constants ---
CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.7
IMG_SIZE = 1440

# Define relevant object classes that block landing zone
OBSTACLE_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'bench', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
    # Added more general objects that could be obstacles
]

ZONE_TOP_LEFT = (100, 150)
ZONE_BOTTOM_RIGHT = (540, 400)
STABLE_FRAMES_REQUIRED = 5 # Number of consistent frames for streak before starting timer
CONFIRMATION_DELAY_SECONDS = 5.0 # How long the status must be consistent before changing

# --- Global Flags for Toggles ---
thermal_mode_active = False
ir_mode_active = False
rotate_display_active = False
detection_active = True # Toggle for enabling/disabling detection

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def put_text_shadow(img, text, pos, font_scale, color, thickness):
    """Draws text on an image with a shadow effect for better visibility."""
    # Ensure position coordinates are integers
    cv2.putText(img, text, (int(pos[0] + 2), int(pos[1] + 2)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (int(pos[0]), int(pos[1])), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def check_significant_overlap(box_coords, zone_tl, zone_br, min_overlap_percentage=0.5):
    """
    Checks if a bounding box significantly overlaps with a defined zone.
    Corrected calculation for intersection area.
    """
    bx1, by1, bx2, by2 = box_coords
    zx1, zy1 = zone_tl
    zx2, zy2 = zone_br

    # Calculate intersection coordinates
    ix1 = max(bx1, zx1)
    iy1 = max(by1, zy1)
    ix2 = min(bx2, zx2)
    iy2 = min(by2, zy2)

    # Calculate intersection area
    intersection_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    
    # Calculate object area
    object_area = (bx2 - bx1) * (by2 - by1)
    
    if object_area == 0:  # Avoid division by zero if object has no area
        return False

    overlap_ratio = intersection_area / object_area
    return overlap_ratio >= min_overlap_percentage

def main():
    print("Loading YOLO model...")
    model = YOLO(r"C:\Users\adity\Desktop\OBJDETECT\yolov8x.pt")
    print("YOLO model loaded.")

    video_path = r"C:\Users\adity\Desktop\OBJDETECT\data\landing_notclearr.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"Error: Cannot open video at {video_path}")
        return

    display_size = (640, 480)
    
    # State variables for stable status detection
    last_immediate_status = None # "CLEAR" or "BLOCKED" for the current frame
    stable_status_streak = 0 # How many consecutive frames the immediate status has been the same
    status_change_start_time = None # Timestamp when the immediate status *first* became stable
    
    final_status = "INITIALIZING..." # Displayed status, changes after delay
    status_color = (0, 165, 255) # Orange for initializing

    frame_times = deque(maxlen=10)

    window_name = "Landing Zone Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Increased height for more text, removed extra space for trackbars
    cv2.resizeWindow(window_name, display_size[0], display_size[1] + 120) 

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("Video ended or error reading frame. Restarting video.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            status_change_start_time = None # Reset timer on video restart
            last_immediate_status = None
            final_status = "INITIALIZING..."
            status_color = (0, 165, 255)
            continue

        orig_h, orig_w = frame.shape[:2]
        
        # Access global flags directly
        global detection_active, thermal_mode_active, ir_mode_active, rotate_display_active

        display_frame = cv2.resize(frame, display_size)

        # Apply display modes based on toggles
        if thermal_mode_active:
            gray_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            display_frame = cv2.applyColorMap(gray_frame, cv2.COLORMAP_JET)
            put_text_shadow(display_frame, "THERMAL MODE (T)", (display_size[0] - 250, 70), 0.7, (255, 255, 255), 1)
        elif ir_mode_active:
            display_frame = cv2.bitwise_not(display_frame)
            put_text_shadow(display_frame, "IR MODE (I)", (display_size[0] - 180, 70), 0.7, (255, 255, 255), 1)

        if rotate_display_active:
            display_frame = cv2.rotate(display_frame, cv2.ROTATE_90_CLOCKWISE)
            display_size_for_text = (display_frame.shape[1], display_frame.shape[0])
            put_text_shadow(display_frame, "ROTATED (R)", (display_size_for_text[0] - 200, 25), 0.7, (255, 255, 255), 1)
        else:
            display_size_for_text = display_size # Use original display_size for text positioning

        objects_in_zone_current_frame = 0
        current_detected_obstacle_names = []
        immediate_status = "CLEAR" # Assume clear initially for the current frame

        if detection_active:
            results = model.predict(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=NMS_IOU_THRESHOLD,
                imgsz=IMG_SIZE,
                device='cpu',
                verbose=False,
                half=False
            )

            scale_x = display_frame.shape[1] / orig_w
            scale_y = display_frame.shape[0] / orig_h

            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = results[0].names[class_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dx1, dy1 = int(x1 * scale_x), int(y1 * scale_y)
                dx2, dy2 = int(x2 * scale_x), int(y2 * scale_y)

                if class_name in OBSTACLE_CLASSES:
                    scaled_zone_tl = (int(ZONE_TOP_LEFT[0] * scale_x), int(ZONE_TOP_LEFT[1] * scale_y))
                    scaled_zone_br = (int(ZONE_BOTTOM_RIGHT[0] * scale_x), int(ZONE_BOTTOM_RIGHT[1] * scale_y))

                    if check_significant_overlap((dx1, dy1, dx2, dy2), scaled_zone_tl, scaled_zone_br):
                        objects_in_zone_current_frame += 1
                        current_detected_obstacle_names.append(class_name.upper())
                        color = (0, 0, 255) # Red for obstacles in zone
                    else:
                        color = (0, 255, 0) # Green for obstacles outside zone
                else:
                    color = (255, 255, 0) # Yellow for other classes

                cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), color, 2)
                put_text_shadow(display_frame, class_name, (dx1, dy1 - 10), 0.6, color, 1)

            if objects_in_zone_current_frame > 0:
                immediate_status = "BLOCKED"
            else:
                immediate_status = "CLEAR"
        else:
            final_status = "DETECTION INACTIVE (D)"
            status_color = (128, 128, 128) # Grey
            last_immediate_status = None # Reset state when detection is off
            stable_status_streak = 0
            status_change_start_time = None

        # --- Stable State Logic ---
        if detection_active: # Only run stable state logic if detection is on
            if last_immediate_status is None:
                last_immediate_status = immediate_status
                stable_status_streak = 1
                status_change_start_time = time.time() # Start timer
            elif immediate_status == last_immediate_status:
                stable_status_streak += 1
            else: # Immediate status has changed
                last_immediate_status = immediate_status
                stable_status_streak = 1
                status_change_start_time = time.time() # Reset timer

            # Check if the status has been stable for long enough
            if stable_status_streak >= STABLE_FRAMES_REQUIRED and \
               (time.time() - status_change_start_time) >= CONFIRMATION_DELAY_SECONDS:
                
                # Check if the final_status is actually changing
                if final_status != immediate_status:
                    final_status = immediate_status
                    if final_status == "CLEAR":
                        status_color = (0, 255, 0) # Green
                        logger.info(f"LANDING ZONE STATUS CHANGE: Zone is now CLEAR.")
                    else:
                        status_color = (0, 0, 255) # Red
                        unique_obstacles = sorted(list(set(current_detected_obstacle_names)))
                        logger.warning(f"!!! ALERT: LANDING ZONE BLOCKED !!! Detected Objects: {', '.join(unique_obstacles)}")
            else:
                # If not yet stable, show a "Pending" or "Checking" state if it's different from current final_status
                if final_status != immediate_status:
                    if immediate_status == "BLOCKED":
                        status_color = (0, 165, 255) # Orange for pending blocked
                        final_status = "CHECKING BLOCKED..."
                    else:
                        status_color = (255, 165, 0) # Light orange for pending clear
                        final_status = "CHECKING CLEAR..."
        
        # Draw the landing zone rectangle
        cv2.rectangle(display_frame, ZONE_TOP_LEFT, ZONE_BOTTOM_RIGHT, status_color, 3)

        # Add a semi-transparent overlay to the landing zone
        overlay_zone = display_frame.copy()
        cv2.rectangle(overlay_zone, ZONE_TOP_LEFT, ZONE_BOTTOM_RIGHT, status_color, -1)
        alpha = 0.15
        cv2.addWeighted(overlay_zone, alpha, display_frame, 1 - alpha, 0, display_frame)

        # Display the overall landing zone status
        put_text_shadow(display_frame, final_status, (20, 40), 1, status_color, 2)

        # Display detected obstacles if the zone is blocked (final status)
        if final_status == "BLOCKED": # Use "BLOCKED" as the string for consistency
            unique_obstacles = sorted(list(set(current_detected_obstacle_names)))
            obstacle_text = "Obstacles: " + ", ".join(unique_obstacles)
            put_text_shadow(display_frame, obstacle_text, (20, 70), 0.7, (0, 0, 255), 1)
            put_text_shadow(display_frame, "!!! ALERT !!!", (20, 100), 0.8, (0, 0, 255), 1)

        # Instructions for keyboard controls (adjusted vertical positions)
        text_start_y = display_size_for_text[1] - 100 # Adjusted starting Y for controls
        put_text_shadow(display_frame, "Controls:", (20, text_start_y), 0.6, (255, 255, 0), 1)
        put_text_shadow(display_frame, "D: Toggle Detection", (20, text_start_y + 20), 0.6, (255, 255, 0), 1)
        put_text_shadow(display_frame, "T: Toggle Thermal Mode", (20, text_start_y + 40), 0.6, (255, 255, 0), 1)
        put_text_shadow(display_frame, "I: Toggle IR Mode", (20, text_start_y + 60), 0.6, (255, 255, 0), 1)
        
        # FIXED: Ensure these coordinates are integers
        put_text_shadow(display_frame, "R: Toggle Rotate Display", (int(display_size_for_text[0] / 2), text_start_y + 20), 0.6, (255, 255, 0), 1)
        put_text_shadow(display_frame, "Q: Quit", (int(display_size_for_text[0] / 2), text_start_y + 40), 0.6, (255, 255, 0), 1)


        # Calculate and display FPS
        curr_time = time.time()
        frame_times.append(curr_time)
        if len(frame_times) > 1:
            avg_frame_time = (frame_times[-1] - frame_times[0]) / (len(frame_times) - 1)
            fps = 1 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            fps = 0

        put_text_shadow(display_frame, f"FPS: {fps:.1f}", (display_size_for_text[0] - 120, 25), 0.7, (255, 255, 255), 1)
        current_datetime_str = time.strftime("%H:%M:%S", time.localtime())
        put_text_shadow(display_frame, current_datetime_str, (display_size_for_text[0] - 130, 45), 0.6, (255, 255, 255), 1)


        cv2.imshow(window_name, display_frame)

        # --- Keyboard Input Handling ---
        key = cv2.waitKey(1) & 0xFF 
        if key == ord('q'):
            logger.info("Exiting...")
            break
        elif key == ord('d'):
            detection_active = not detection_active
            logger.info(f"Keyboard Control: Object Detection Toggled {'ON' if detection_active else 'OFF'}")
        elif key == ord('t'):
            thermal_mode_active = not thermal_mode_active
            # Ensure only one display mode is active
            if thermal_mode_active:
                ir_mode_active = False
            logger.info(f"Keyboard Control: Thermal Mode Toggled {'ON' if thermal_mode_active else 'OFF'}")
        elif key == ord('i'):
            ir_mode_active = not ir_mode_active
            # Ensure only one display mode is active
            if ir_mode_active:
                thermal_mode_active = False
            logger.info(f"Keyboard Control: IR Mode Toggled {'ON' if ir_mode_active else 'OFF'}")
        elif key == ord('r'):
            rotate_display_active = not rotate_display_active
            logger.info(f"Keyboard Control: Rotate Display Toggled {'ON' if rotate_display_active else 'OFF'}")


    cap.release()
    cv2.destroyAllWindows()
    logger.info("Application finished.")

if __name__ == "__main__":
    main()