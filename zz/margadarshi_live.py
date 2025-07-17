from ultralytics import YOLO
import cv2
import time
import torch
import numpy as np
import os
import threading
import queue

# Try importing playsound, handle if not installed
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    print("Warning: 'playsound' library not found. Audible alerts will be disabled.")
    print("To enable audible alerts, install it: pip install playsound")
    PLAYSOUND_AVAILABLE = False

# --- Configuration ---
CONFIG = {
    "CONFIDENCE_THRESHOLD": 0.4, # Lower for more detections, higher for fewer false positives
    "NMS_IOU_THRESHOLD": 0.5,    # NMS IOU for YOLO detections
    "IMG_SIZE": 640,             # Recommended for YOLOv8. Higher might be slower. This is the input size for the YOLO model.
    "DISPLAY_WIDTH": 1920,       # Desired width of the display window
    "DISPLAY_HEIGHT": 1080,      # Desired height of the display window

    # Runway Zone as Proportions of the Frame (0.0 to 1.0)
    # This allows the zone to scale with the frame size.
    # Example: (0.2, 0.3) means 20% from left, 30% from top
    "ZONE_PROPORTIONAL_TOP_LEFT": [0.2, 0.3],    # (x_prop, y_prop) - Made list to be mutable
    "ZONE_PROPORTIONAL_BOTTOM_RIGHT": [0.8, 0.7], # (x_prop, y_prop) - Made list to be mutable

    # Zone Adjustment Step Size (as a proportion of the frame)
    "ZONE_ADJUST_STEP": 0.01, # 1% of the frame width/height per key press
    "ZONE_GROW_STEP_UNIFORM": 0.02, # 2% uniform growth for the 'g' key

    "IP_STREAM_URL": "http://10.236.237.207:8080/video", # Change this to your IP camera URL or 0 for webcam
    "DESIRED_CAMERA_FPS": 60,    # Attempt to set camera FPS to a higher value
    "LOG_INTERVAL_NOT_CLEAR_SEC": 3, # How often to log "not clear" message AND save screenshot
    "MODEL_PATH": r'C:\Users\adity\Desktop\OBJDETECT\yolov8n.pt', # Path to your YOLOv8 nano model

    # --- UPDATED TARGET_CLASSES ---
    # COCO classes:
    # 0: person (human)
    # 2: car, 3: motorcycle, 5: bus, 7: truck, 4: airplane (various vehicles)
    # 56: chair
    # 60: dining table (often used for desk)
    # 39: bottle
    # 24: backpack, 26: handbag, 28: suitcase, 73: book, 41: cup, 45: bowl, 63: laptop, 67: cell phone (examples for "all the stuff")
    "TARGET_CLASSES": [
        0,  # person (human)
        2,  # car
        3,  # motorcycle
        4,  # airplane
        5,  # bus
        7,  # truck
        56, # chair
        60, # dining table (as a proxy for desk)
        39, # bottle
        # Add more "stuff" here if you want specific common items
        24, # backpack
        26, # handbag
        28, # suitcase
        73, # book
        41, # cup
        45, # bowl
        63, # laptop
        67  # cell phone
    ],
    # If you truly want *all* detectible objects, set TARGET_CLASSES to None:
    # "TARGET_CLASSES": None,

    "MAX_PIXEL_DISTANCE_FOR_TRACK": 100, # Max pixels a centroid can move to be considered same object
    "TRACK_EXPIRY_FRAMES": 30,           # How many frames to keep a track if not seen
    "ALERT_SOUND_PATH": "alert.wav",     # Path to your alert sound file (e.g., a simple beep or alarm sound)
    "ALERT_SCREENSHOT_DIR": "runway_alerts" # Directory to save automatic screenshots
}

# --- Create alert screenshot directory if it doesn't exist ---
if not os.path.exists(CONFIG["ALERT_SCREENSHOT_DIR"]):
    os.makedirs(CONFIG["ALERT_SCREENSHOT_DIR"])
    print(f"Created directory for alerts: {CONFIG['ALERT_SCREENSHOT_DIR']}")

# --- Check for alert sound file ---
if PLAYSOUND_AVAILABLE and not os.path.exists(CONFIG["ALERT_SOUND_PATH"]):
    print(f"Warning: Alert sound file '{CONFIG['ALERT_SOUND_PATH']}' not found. Please ensure it's a valid .wav file for audio alerts.")

# Global flag to control the alert sound thread
g_run_sound_thread = False
g_sound_thread_running = False

def alert_sound_thread_func():
    """Function to be run in a separate thread for continuous sound."""
    global g_run_sound_thread, g_sound_thread_running
    g_sound_thread_running = True
    print("Alert sound thread started.")
    while g_run_sound_thread:
        if PLAYSOUND_AVAILABLE and os.path.exists(CONFIG["ALERT_SOUND_PATH"]):
            try:
                playsound(CONFIG["ALERT_SOUND_PATH"], block=True)
            except Exception as e:
                print(f"Error in sound thread playing sound: {e}")
                g_run_sound_thread = False
        else:
            time.sleep(0.5)
    print("Alert sound thread stopped.")
    g_sound_thread_running = False

# --- Multithreaded Camera Class ---
class VideoStream:
    def __init__(self, src=0, desired_fps=30, width=None, height=None):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise IOError(f"Cannot open video stream at {src}")

        self.stream.set(cv2.CAP_PROP_FPS, desired_fps)
        if width is not None:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.queue = queue.Queue(maxsize=1)

    def start(self):
        if self.started:
            print("Video stream already started.")
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        print(f"Camera stream started, actual FPS: {self.stream.get(cv2.CAP_PROP_FPS):.2f}")
        print(f"Camera resolution: {int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.stream.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            if not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put(frame)

    def read(self):
        if not self.queue.empty():
            return True, self.queue.get()
        return False, None

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join(timeout=1)
        self.stream.release()
        print("Video stream stopped.")

def put_text(img, text, pos, color=(0, 255, 0), font_scale=0.7, thickness=2):
    """Helper function to draw text on the image."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

# --- Simple Object Tracking without DeepSORT ---
class SimpleTracker:
    def __init__(self, max_distance=50, expiry_frames=30):
        self.tracks = {}  # {track_id: {'bbox': [x1, y1, x2, y2], 'centroid': (cx, cy), 'frames_since_last_seen': 0, 'class': cls}}
        self.next_id = 0
        self.max_distance = max_distance
        self.expiry_frames = expiry_frames

    def update(self, detections, current_frame_number):
        # detections: list of ([x1, y1, x2, y2], confidence, class_id)

        updated_tracks = {}
        matched_detection_indices = set()

        # Step 1: Try to match current detections to existing tracks
        for detection_idx, (bbox, conf, cls) in enumerate(detections):
            det_cx = (bbox[0] + bbox[2]) // 2
            det_cy = (bbox[1] + bbox[3]) // 2

            best_match_id = -1
            min_dist = float('inf')

            for track_id, track_data in self.tracks.items():
                track_cx, track_cy = track_data['centroid']
                distance = np.sqrt((det_cx - track_cx)**2 + (det_cy - track_cy)**2)

                if distance < self.max_distance and distance < min_dist:
                    min_dist = distance
                    best_match_id = track_id

            if best_match_id != -1:
                # Update existing track
                updated_tracks[best_match_id] = {
                    'bbox': bbox,
                    'centroid': (det_cx, det_cy),
                    'frames_since_last_seen': 0,
                    'class': cls,
                    'track_id': best_match_id # Add track_id for easier drawing
                }
                matched_detection_indices.add(detection_idx)

        # Step 2: Create new tracks for unmatched detections
        for detection_idx, (bbox, conf, cls) in enumerate(detections):
            if detection_idx not in matched_detection_indices:
                det_cx = (bbox[0] + bbox[2]) // 2
                det_cy = (bbox[1] + bbox[3]) // 2
                self.tracks[self.next_id] = {
                    'bbox': bbox,
                    'centroid': (det_cx, det_cy),
                    'frames_since_last_seen': 0,
                    'class': cls,
                    'track_id': self.next_id
                }
                updated_tracks[self.next_id] = self.tracks[self.next_id] # Add to updated list immediately
                self.next_id += 1

        # Step 3: Increment 'frames_since_last_seen' for unmatched tracks and remove expired ones
        tracks_to_delete = []
        for track_id, track_data in self.tracks.items():
            if track_id not in updated_tracks: # If this track was not updated in this frame
                track_data['frames_since_last_seen'] += 1
                if track_data['frames_since_last_seen'] > self.expiry_frames:
                    tracks_to_delete.append(track_id)
                else:
                    updated_tracks[track_id] = track_data # Keep non-expired unmatched tracks

        for track_id in tracks_to_delete:
            del self.tracks[track_id]

        self.tracks = updated_tracks # Update the main tracks dictionary

        return list(self.tracks.values()) # Return confirmed tracks for drawing

def main():
    global g_run_sound_thread, g_sound_thread_running

    # --- System Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(CONFIG["MODEL_PATH"])
    model.to(device)
    print("YOLO Model loaded.")

    # Initialize Simple Tracker
    tracker = SimpleTracker(
        max_distance=CONFIG["MAX_PIXEL_DISTANCE_FOR_TRACK"],
        expiry_frames=CONFIG["TRACK_EXPIRY_FRAMES"]
    )
    print("Simple Tracker initialized.")

    # --- Video Stream Setup (using the new threaded class) ---
    ip_stream_url = CONFIG["IP_STREAM_URL"]
    cap = VideoStream(src=ip_stream_url,
                      desired_fps=CONFIG["DESIRED_CAMERA_FPS"],
                      width=CONFIG["DISPLAY_WIDTH"],
                      height=CONFIG["DISPLAY_HEIGHT"]).start()

    time.sleep(1)

    if not cap.grabbed:
        print(f"ERROR: Could not open stream at {ip_stream_url}. Please check the URL or camera connection.")
        cap.stop()
        return

    # --- Zone and UI Setup ---
    cv2.namedWindow("Runway Clearance", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Runway Clearance", CONFIG["DISPLAY_WIDTH"], CONFIG["DISPLAY_HEIGHT"])

    zone_top_left_prop = CONFIG["ZONE_PROPORTIONAL_TOP_LEFT"]
    zone_bottom_right_prop = CONFIG["ZONE_PROPORTIONAL_BOTTOM_RIGHT"]
    zone_adjust_step = CONFIG["ZONE_ADJUST_STEP"]
    zone_grow_step_uniform = CONFIG["ZONE_GROW_STEP_UNIFORM"]

    # --- State Variables ---
    last_frame_time = time.perf_counter()
    last_log_time = time.perf_counter()
    ir_mode = False
    rotate_state = 0
    was_runway_clear = True
    frame_count = 0

    print("\nPress 'q' to quit, 'i' to toggle IR mode, 'r' to rotate, 's' to save screenshot manually.")
    print("Adjust Zone: [UP/DOWN]=Height  [LEFT/RIGHT]=Width")
    print("Toggle Zone Size (Uniform): [g]=Grow")

    # --- Main Loop ---
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("WARNING: No new frame available. Waiting for next frame...")
            time.sleep(0.01)
            continue

        frame_count += 1

        # --- Frame Pre-processing for Display and YOLO ---
        if rotate_state == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotate_state == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotate_state == 3:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        original_h, original_w = frame.shape[:2]
        display_w, display_h = CONFIG["DISPLAY_WIDTH"], CONFIG["DISPLAY_HEIGHT"]

        scale = min(display_w / original_w, display_h / original_h)
        scaled_w = int(original_w * scale)
        scaled_h = int(original_h * scale)

        display_frame = cv2.resize(frame, (scaled_w, scaled_h))

        canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        x_offset = (display_w - scaled_w) // 2
        y_offset = (display_h - scaled_h) // 2
        canvas[y_offset:y_offset+scaled_h, x_offset:x_offset+scaled_w] = display_frame.copy()

        display_frame = canvas

        if ir_mode:
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            fake_ir = cv2.bitwise_not(gray)
            display_frame = cv2.cvtColor(fake_ir, cv2.COLOR_GRAY2BGR)

        # --- Dynamic Runway Zone Calculation ---
        zone_top_left_x = int(zone_top_left_prop[0] * scaled_w) + x_offset
        zone_top_left_y = int(zone_top_left_prop[1] * scaled_h) + y_offset
        zone_bottom_right_x = int(zone_bottom_right_prop[0] * scaled_w) + x_offset
        zone_bottom_right_y = int(zone_bottom_right_prop[1] * scaled_h) + y_offset

        zone_top_left = (zone_top_left_x, zone_top_left_y)
        zone_bottom_right = (zone_bottom_right_x, zone_bottom_right_y)

        # --- YOLO Detection ---
        yolo_input_frame = cv2.resize(frame, (CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]))

        results = model.predict(
            yolo_input_frame,
            conf=CONFIG["CONFIDENCE_THRESHOLD"],
            iou=CONFIG["NMS_IOU_THRESHOLD"],
            imgsz=CONFIG["IMG_SIZE"],
            verbose=False,
            device=device,
            classes=CONFIG["TARGET_CLASSES"]
        )

        current_detections = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                scaled_x1 = x1 * (original_w / CONFIG["IMG_SIZE"])
                scaled_y1 = y1 * (original_h / CONFIG["IMG_SIZE"])
                scaled_x2 = x2 * (original_w / CONFIG["IMG_SIZE"])
                scaled_y2 = y2 * (original_h / CONFIG["IMG_SIZE"])

                final_x1 = int(scaled_x1 * (scaled_w / original_w) + x_offset)
                final_y1 = int(scaled_y1 * (scaled_h / original_h) + y_offset)
                final_x2 = int(scaled_x2 * (scaled_w / original_w) + x_offset)
                final_y2 = int(scaled_y2 * (scaled_h / original_h) + y_offset)

                current_detections.append(([final_x1, final_y1, final_x2, final_y2], conf, cls))

        # --- Simple Object Tracking Update ---
        tracked_objects = tracker.update(current_detections, frame_count)

        objects_in_zone_tracked = 0
        annotated_frame = display_frame.copy()

        # Iterate through confirmed tracks and draw them
        for track_data in tracked_objects:
            x1, y1, x2, y2 = map(int, track_data['bbox'])
            track_id = track_data['track_id']
            detection_class = track_data['class']

            track_color = (255, 0, 0) # Blue for tracked objects
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), track_color, 2)
            put_text(annotated_frame, f"ID: {track_id} ({model.names[detection_class]})", (x1, y1 - 10), track_color, font_scale=0.6)

            # Check if object is in the defined zone
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if zone_top_left[0] <= cx <= zone_bottom_right[0] and zone_top_left[1] <= cy <= zone_bottom_right[1]:
                objects_in_zone_tracked += 1

        # --- Runway Clearance Logic & Alerts ---
        runway_clear = (objects_in_zone_tracked == 0)
        zone_color = (0, 255, 0) if runway_clear else (0, 0, 255)
        text_status = "RUNWAY CLEAR" if runway_clear else "RUNWAY NOT CLEAR"

        if not runway_clear:
            if not g_sound_thread_running:
                g_run_sound_thread = True
                threading.Thread(target=alert_sound_thread_func, daemon=True).start()
        else:
            if g_sound_thread_running:
                g_run_sound_thread = False

        if not runway_clear and was_runway_clear:
            print("ALERT: Runway status changed to NOT CLEAR!")
        elif runway_clear and not was_runway_clear:
            print("INFO: Runway is now CLEAR.")

        was_runway_clear = runway_clear

        # Draw zone
        cv2.rectangle(annotated_frame, zone_top_left, zone_bottom_right, zone_color, 2)

        if not runway_clear:
            if int(time.perf_counter() * 5) % 2 == 0:
                put_text(annotated_frame, text_status, (20, 40), zone_color, font_scale=1.2, thickness=4)
            else:
                put_text(annotated_frame, text_status, (20, 40), (255, 255, 255), font_scale=1.2, thickness=4)
        else:
            put_text(annotated_frame, text_status, (20, 40), zone_color, font_scale=0.9, thickness=3)

        put_text(annotated_frame, f"Objects in Zone: {objects_in_zone_tracked}", (20, 70), zone_color)

        current_time = time.perf_counter()
        fps = 1 / (current_time - last_frame_time) if (current_time - last_frame_time) > 0 else 0
        last_frame_time = current_time
        put_text(annotated_frame, f"FPS: {fps:.1f}", (annotated_frame.shape[1] - 150, 30), (255, 255, 255))

        # --- Guide Text ---
        guide_text_general = "Keys: [q]=Quit  [i]=Toggle IR  [r]=Rotate  [s]=Save Frame manually"
        put_text(annotated_frame, guide_text_general, (20, annotated_frame.shape[0] - 20), (0, 255, 255))
        guide_text_zone_directional = "Zone (Directional): [UP/DOWN]=Height  [LEFT/RIGHT]=Width"
        put_text(annotated_frame, guide_text_zone_directional, (20, annotated_frame.shape[0] - 50), (0, 255, 255))
        guide_text_zone_uniform = "Zone (Uniform): [g]=Grow" # New guide text
        put_text(annotated_frame, guide_text_zone_uniform, (20, annotated_frame.shape[0] - 80), (0, 255, 255))


        # --- Display Frame ---
        cv2.imshow("Runway Clearance", annotated_frame)

        # --- Periodic Logging & AUTOMATIC SCREENSHOT ---
        if not runway_clear and (time.perf_counter() - last_log_time >= CONFIG["LOG_INTERVAL_NOT_CLEAR_SEC"]):
            log_message = f"LOG: Runway not clear - {objects_in_zone_tracked} object(s) detected/tracked in zone."
            print(log_message)

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            screenshot_filename = os.path.join(CONFIG["ALERT_SCREENSHOT_DIR"], f"runway_alert_{timestamp}.png")
            try:
                cv2.imwrite(screenshot_filename, annotated_frame)
                print(f"Automatic screenshot saved: {screenshot_filename}")
            except Exception as e:
                print(f"ERROR: Could not save automatic screenshot {screenshot_filename}: {e}")

            last_log_time = time.perf_counter()

        # --- Key Handling ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting application.")
            break
        elif key == ord('i'):
            ir_mode = not ir_mode
            print("IR mode toggled:", "ON" if ir_mode else "OFF")
        elif key == ord('r'):
            rotate_state = (rotate_state + 1) % 4
            print(f"Rotation: {rotate_state * 90}°")
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            manual_screenshot_filename = os.path.join(CONFIG["ALERT_SCREENSHOT_DIR"], f"manual_screenshot_{timestamp}.png")
            try:
                cv2.imwrite(manual_screenshot_filename, annotated_frame)
                print(f"Manual screenshot saved: {manual_screenshot_filename}")
            except Exception as e:
                print(f"ERROR: Could not save manual screenshot {manual_screenshot_filename}: {e}")
        # --- Zone Adjustment Keys (Directional) ---
        elif key == 82: # Up arrow key (increase height)
            zone_top_left_prop[1] = max(0.0, zone_top_left_prop[1] - zone_adjust_step)
            zone_bottom_right_prop[1] = min(1.0, zone_bottom_right_prop[1] + zone_adjust_step)
            print(f"Zone Height Increased. Current TL: {zone_top_left_prop}, BR: {zone_bottom_right_prop}")
        elif key == 84: # Down arrow key (decrease height)
            if zone_bottom_right_prop[1] - zone_top_left_prop[1] > (2 * zone_adjust_step):
                zone_top_left_prop[1] = min(1.0, zone_top_left_prop[1] + zone_adjust_step)
                zone_bottom_right_prop[1] = max(0.0, zone_bottom_right_prop[1] - zone_adjust_step)
                print(f"Zone Height Decreased. Current TL: {zone_top_left_prop}, BR: {zone_bottom_right_prop}")
        elif key == 81: # Left arrow key (decrease width)
            if zone_bottom_right_prop[0] - zone_top_left_prop[0] > (2 * zone_adjust_step):
                zone_top_left_prop[0] = min(1.0, zone_top_left_prop[0] + zone_adjust_step)
                zone_bottom_right_prop[0] = max(0.0, zone_bottom_right_prop[0] - zone_adjust_step)
                print(f"Zone Width Decreased. Current TL: {zone_top_left_prop}, BR: {zone_bottom_right_prop}")
        elif key == 83: # Right arrow key (increase width)
            zone_top_left_prop[0] = max(0.0, zone_top_left_prop[0] - zone_adjust_step)
            zone_bottom_right_prop[0] = min(1.0, zone_bottom_right_prop[0] + zone_adjust_step)
            print(f"Zone Width Increased. Current TL: {zone_top_left_prop}, BR: {zone_bottom_right_prop}")
        # --- New Key for Uniform Zone Growth ---
        elif key == ord('g'): # 'g' key for "grow"
            # Increase width and height uniformly from the center
            # Decrease top-left x,y and increase bottom-right x,y
            zone_top_left_prop[0] = max(0.0, zone_top_left_prop[0] - zone_grow_step_uniform)
            zone_top_left_prop[1] = max(0.0, zone_top_left_prop[1] - zone_grow_step_uniform)
            zone_bottom_right_prop[0] = min(1.0, zone_bottom_right_prop[0] + zone_grow_step_uniform)
            zone_bottom_right_prop[1] = min(1.0, zone_bottom_right_prop[1] + zone_grow_step_uniform)
            print(f"Zone Size Increased Uniformly. Current TL: {zone_top_left_prop}, BR: {zone_bottom_right_prop}")


    # --- Cleanup ---
    cap.stop()
    g_run_sound_thread = False
    time.sleep(0.5)
    cv2.destroyAllWindows()
    print("Application terminated successfully.")

if __name__ == "__main__":
    main()