import warnings
warnings.simplefilter(action='ignore', category=Warning)
import cv2
import torch
import numpy as np
import os
import threading
import queue
from datetime import datetime
import time

# Try importing playsound, handle if not installed
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    print("Warning: 'playsound' library not found. Audible alerts will be disabled for Margadarshi.")
    print("To enable audible alerts, install it: pip install playsound")
    print("Command: pip install playsound")
    PLAYSOUND_AVAILABLE = False

# Try importing pyttsx3, handle if not installed
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    print("Warning: 'pyttsx3' library not found. Voice alerts will be disabled for Margadarshi.")
    print("To enable voice alerts, install it: pip install pyttsx3")
    print("Command: pip install pyttsx3")
    PYTTSX3_AVAILABLE = False


from ultralytics import YOLO

# --- Configuration Constants ---
# Define relevant object classes that block landing zone by name
# These names must correspond to the classes recognized by your YOLO model (e.g., COCO dataset classes for yolov8n.pt)
OBSTACLE_CLASS_NAMES = [
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
]

# --- Global Configuration (Defaults - can be overridden by instance config) ---
# These are the default values if no config is passed to MargadarshiCamera
DEFAULT_MARGADARSHI_CONFIG = {
    "CONFIDENCE_THRESHOLD": 0.25, # Adjusted for better detection sensitivity
    "NMS_IOU_THRESHOLD": 0.5,
    "IMG_SIZE": 640, # Image size for YOLO model input
    "DISPLAY_WIDTH": 1280, # Desired width for the output video stream
    "DISPLAY_HEIGHT": 720, # Desired height for the output video stream
    "ZONE_PROPORTIONAL_TOP_LEFT": [0.2, 0.3], # Proportional coordinates for the exclusion zone
    "ZONE_PROPORTIONAL_BOTTOM_RIGHT": [0.8, 0.7],
    "ZONE_ADJUST_STEP": 0.01, # Step size for zone adjustment (not used in current UI, but kept for potential future use)
    "ZONE_GROW_STEP_UNIFORM": 0.02, # Step size for zone growth (not used in current UI)
    "IP_STREAM_URL": "0", # Default to webcam (0) or a video file path/IP camera URL
    "DESIRED_CAMERA_FPS": 30, # Desired frames per second from the camera
    "LOG_INTERVAL_NOT_CLEAR_SEC": 3, # Interval in seconds for logging "not clear" status and screenshots
    "MODEL_PATH": r'yolov8_models/yolov8n.pt', # Path to your YOLOv8 nano model file
    "TARGET_CLASSES": [], # This will be dynamically populated at runtime based on OBSTACLE_CLASS_NAMES
    "MAX_PIXEL_DISTANCE_FOR_TRACK": 100, # Max pixel distance for object tracking association
    "TRACK_EXPIRY_FRAMES": 30, # Number of frames before an untracked object expires
    "ALERT_SOUND_PATH": "alert.wav", # Path to the alert sound file
    "ALERT_SCREENSHOT_DIR": "static/runway_alerts", # Directory to save alert screenshots
    "VOICE_ALERTS_ENABLED": True, # Toggle for voice alerts
    "VISUAL_ALERT_ENABLED": True, # Toggle for visual alerts (pulsing red box)
    "AUTO_SCREENSHOT_ENABLED": True, # Toggle for automatic screenshots on alert
    "ALERT_SOUND_ENABLED": True, # Toggle for audible alert sound
    "IR_MODE": False # Toggle for IR visual mode
}

# Global flags and queues for Flask integration
margadarshi_log_queue = queue.Queue() # Queue for sending log messages to the frontend
margadarshi_alert_active_flag = False # Flag to indicate if an alert is active
margadarshi_alert_lock = threading.Lock() # Lock for thread-safe access to alert_active_flag
_tts_lock_margadarshi = threading.Lock() # Lock for thread-safe access to pyttsx3 engine
_playsound_lock_margadarshi = threading.Lock() # Lock for thread-safe access to playsound

def set_margadarshi_alert_active(state: bool):
    """Sets the global alert active flag."""
    global margadarshi_alert_active_flag
    with margadarshi_alert_lock:
        margadarshi_alert_active_flag = state

def get_margadarshi_alert_active() -> bool:
    """Gets the current state of the global alert active flag."""
    with margadarshi_alert_lock:
        return margadarshi_alert_active_flag

def get_margadarshi_log_queue():
    """Returns the log queue."""
    return margadarshi_log_queue

# --- Helper function for drawing text on OpenCV images ---
def put_text(img, text, pos, color=(0, 255, 0), font_scale=0.7, thickness=2):
    """Draws text on an OpenCV image."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

# --- Simple Object Tracking without DeepSORT ---
class SimpleTracker:
    """
    A basic object tracker that associates new detections with existing tracks
    based on proximity of centroids.
    """
    def __init__(self, max_distance=50, expiry_frames=30):
        self.tracks = {} # Stores active tracks: {track_id: {bbox, centroid, frames_since_last_seen, class, track_id}}
        self.next_id = 0 # Next available track ID
        self.max_distance = max_distance # Maximum pixel distance to associate a detection with a track
        self.expiry_frames = expiry_frames # Number of frames without detection before a track is removed

    def update(self, detections, current_frame_number):
        """
        Updates existing tracks with new detections and creates new tracks for unmatched detections.
        detections: List of (bbox, conf, cls) tuples for current frame.
        current_frame_number: Current frame count.
        Returns a list of updated track data.
        """
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
                    'frames_since_last_seen': 0, # Reset counter
                    'class': cls,
                    'track_id': best_match_id
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
                updated_tracks[self.next_id] = self.tracks[self.next_id]
                self.next_id += 1

        # Step 3: Increment 'frames_since_last_seen' for unmatched tracks and remove expired ones
        tracks_to_delete = []
        for track_id, track_data in self.tracks.items():
            if track_id not in updated_tracks: # If track was not matched in this frame
                track_data['frames_since_last_seen'] += 1
                if track_data['frames_since_last_seen'] > self.expiry_frames:
                    tracks_to_delete.append(track_id)
                else:
                    updated_tracks[track_id] = track_data # Keep non-expired, unmatched tracks

        for track_id in tracks_to_delete:
            del self.tracks[track_id]

        self.tracks = updated_tracks # Update the main tracks dictionary

        return list(self.tracks.values())

# --- Helper function for checking significant overlap ---
def check_significant_overlap(box_coords, zone_tl, zone_br, min_overlap_percentage=0.2):
    """
    Checks if a bounding box significantly overlaps with a defined zone.
    This is more robust than just checking if the centroid is in the zone.
    box_coords: (x1, y1, x2, y2) of the bounding box
    zone_tl: (x1, y1) of the zone top-left
    zone_br: (x2, y2) of the zone bottom-right
    min_overlap_percentage: minimum percentage of object's area that must overlap with the zone
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


# --- Main Margadarshi Camera Class ---
class MargadarshiCamera:
    """
    Manages the video stream, YOLO detection, object tracking, and runway clearance logic
    for the Margadarshi system.
    """
    def __init__(self, config: dict = None):
        self.config = DEFAULT_MARGADARSHI_CONFIG.copy()
        if config:
            self.config.update(config)
        
        print(f"Initializing Margadarshi with config: {self.config}")

        # Initialize zone properties early to prevent AttributeError
        self.zone_top_left_prop = list(self.config["ZONE_PROPORTIONAL_TOP_LEFT"])
        self.zone_bottom_right_prop = list(self.config["ZONE_PROPORTIONAL_BOTTOM_RIGHT"])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Margadarshi using device: {self.device}")

        self._is_ready = False # Flag to indicate if the instance is fully initialized and ready to stream

        # Initialize sound thread instance to None
        self.sound_thread_instance = None 

        # Load YOLO model
        print(f"Loading YOLO model from {self.config['MODEL_PATH']}...")
        try:
            # Ensure model path is absolute or relative to the project root
            model_abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", self.config['MODEL_PATH'])
            self.model = YOLO(model_abs_path)
            self.model.to(self.device)
            print("YOLO Model loaded.")
            self._add_log_to_queue(f"YOLO model names: {self.model.names}") # Log model names

            # Dynamically populate TARGET_CLASSES based on OBSTACLE_CLASS_NAMES
            if not self.model.names:
                self._add_log_to_queue("ERROR: YOLO model names are empty. Cannot populate TARGET_CLASSES. Model might not have loaded correctly.")
                # Do not set _is_ready = True if model names are missing
            else:
                self.config["TARGET_CLASSES"] = []
                for class_name in OBSTACLE_CLASS_NAMES:
                    if class_name in self.model.names.values():
                        # Find the key (ID) for the value (name)
                        class_id = [k for k, v in self.model.names.items() if v == class_name][0]
                        self.config["TARGET_CLASSES"].append(class_id)
                self._add_log_to_queue(f"Target classes for detection (IDs): {self.config['TARGET_CLASSES']}")
                if not self.config["TARGET_CLASSES"]:
                    self._add_log_to_queue("WARNING: No target classes found from OBSTACLE_CLASS_NAMES in the loaded model. Detection might not work as expected.")

        except Exception as e:
            print(f"Error loading YOLO model from {self.config['MODEL_PATH']}: {e}")
            self.model = None
            self._add_log_to_queue(f"ERROR: Failed to load YOLO model: {e}")
            # Do not set _is_ready = True on model load failure

        # Video capture
        self.cap = None
        self._init_video_stream(self.config["IP_STREAM_URL"])

        # Set _is_ready only if model is loaded, video stream is opened, and target classes are populated
        if self.model and self.cap and self.cap.isOpened() and self.config["TARGET_CLASSES"]:
            self._is_ready = True
            self._add_log_to_queue("Margadarshi instance is now ready.")
        else:
            self._add_log_to_queue("Margadarshi instance failed to become ready due to model, video, or target class initialization issues.")

        # Initialize Simple Tracker
        self.tracker = SimpleTracker(
            max_distance=self.config["MAX_PIXEL_DISTANCE_FOR_TRACK"],
            expiry_frames=self.config["TRACK_EXPIRY_FRAMES"]
        )
        print("Simple Tracker initialized.")

        # State Variables
        self.last_frame_time = time.perf_counter()
        self.last_log_time = time.perf_counter()
        self.last_audio_alert_time = time.perf_counter() # Initialize this attribute here
        self.ir_mode = self.config.get("IR_MODE", False) # Default to False if not in config
        self.rotate_state = self.config.get("ROTATION_STATE", 0) # Default to 0
        self.was_runway_clear = True # Tracks previous state for status change logging
        self.frame_count = 0 # Frame counter for FPS calculation
        self.running = True # Control flag for the generator loop

        print("Margadarshi IDS initialization sequence complete.")

    @property
    def is_ready(self):
        """Property to check if the Margadarshi instance is fully ready for streaming."""
        return self._is_ready

    def _init_video_stream(self, src):
        """Initializes or re-initializes the video stream."""
        if self.cap:
            self.cap.release() # Release existing stream if any
            self.cap = None

        try:
            if isinstance(src, str) and src.isdigit():
                self.cap = cv2.VideoCapture(int(src))
            else:
                self.cap = cv2.VideoCapture(src)
            
            if not self.cap.isOpened():
                raise IOError(f"Cannot open video stream at {src}")

            # Attempt to set camera properties
            # These might not be strictly adhered to by all cameras/video files
            self.cap.set(cv2.CAP_PROP_FPS, self.config["DESIRED_CAMERA_FPS"])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["DISPLAY_WIDTH"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["DISPLAY_HEIGHT"])

            # Read first frame to get actual dimensions
            ret, frame = self.cap.read()
            if not ret:
                raise IOError("Could not read the first frame from the video source.")
            
            self.frame_height, self.frame_width, _ = frame.shape
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset to beginning if it's a file

            print(f"Margadarshi video source opened successfully. Resolution: {self.frame_width}x{self.frame_height}")
            self._add_log_to_queue(f"Video source set to: {src}")

        except Exception as e:
            print(f"Error opening video source {src}: {e}")
            self.cap = None
            self._add_log_to_queue(f"ERROR: Failed to open video source: {e}")

    def _play_alert_sound(self):
        """Plays the alert sound in a separate thread."""
        if not PLAYSOUND_AVAILABLE or not self.config["ALERT_SOUND_ENABLED"]:
            return
        
        with _playsound_lock_margadarshi:
            # Only start a new sound thread if one isn't already running
            if self.sound_thread_instance and self.sound_thread_instance.is_alive():
                return
            
            self.run_sound_thread = True # Flag to keep the sound loop running
            self.sound_thread_instance = threading.Thread(target=self._alert_sound_loop, daemon=True)
            self.sound_thread_instance.start()
            self._add_log_to_queue("Alert sound started.")

    def _stop_alert_sound(self):
        """Stops the alert sound thread."""
        with _playsound_lock_margadarshi:
            if self.sound_thread_instance and self.sound_thread_instance.is_alive():
                self.run_sound_thread = False # Set flag to stop the loop
                # No need to join here, as it might block the main video processing loop.
                # The daemon thread will exit when the main program exits.
                self._add_log_to_queue("Alert sound stopped.")

    def _alert_sound_loop(self):
        """Loop for playing alert sound continuously until `run_sound_thread` is False."""
        while self.run_sound_thread:
            if PLAYSOUND_AVAILABLE and os.path.exists(self.config["ALERT_SOUND_PATH"]):
                try:
                    playsound(self.config["ALERT_SOUND_PATH"], block=True) # block=True means it waits for sound to finish
                except Exception as e:
                    print(f"Error in sound thread playing sound: {e}")
                    self.run_sound_thread = False # Stop thread on error
            else:
                time.sleep(0.5) # Wait if playsound not available or file missing

    def _add_log_to_queue(self, message: str):
        """Adds a log message to the thread-safe queue for frontend display."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        margadarshi_log_queue.put(f"[{timestamp}] {message}")

    def _trigger_alert_actions(self, annotated_frame: np.ndarray, objects_in_zone: int):
        """Handles automatic screenshots and logging when runway is not clear."""
        current_time = time.perf_counter()

        # Check for logging and screenshot cooldown
        if self.config["AUTO_SCREENSHOT_ENABLED"] and (current_time - self.last_log_time >= self.config["LOG_INTERVAL_NOT_CLEAR_SEC"]):
            log_message = f"LOG: Runway not clear - {objects_in_zone} object(s) detected/tracked in zone."
            self._add_log_to_queue(log_message)

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            screenshot_filename = os.path.join(self.config["ALERT_SCREENSHOT_DIR"], f"runway_alert_{timestamp}.png")
            try:
                cv2.imwrite(screenshot_filename, annotated_frame)
                self._add_log_to_queue(f"Automatic screenshot saved: {os.path.basename(screenshot_filename)}")
            except Exception as e:
                self._add_log_to_queue(f"ERROR: Could not save automatic screenshot {os.path.basename(screenshot_filename)}: {e}")
            self.last_log_time = current_time

        # Voice alert (using pyttsx3)
        if PYTTSX3_AVAILABLE and self.config["VOICE_ALERTS_ENABLED"] and (current_time - self.last_audio_alert_time >= self.config["LOG_INTERVAL_NOT_CLEAR_SEC"]): # Use same cooldown for voice
            alert_text = "Warning! Runway not clear."
            # Run TTS in a separate thread to avoid blocking the main frame processing loop
            threading.Thread(target=self._play_voice_alert, args=(alert_text,)).start()
            self.last_audio_alert_time = current_time # Update time even if PYTTSX3 is not available to prevent repeated alerts if it becomes available

    def _play_voice_alert(self, text):
        """Converts text to speech and plays it directly on the device using pyttsx3."""
        with _tts_lock_margadarshi: # Use a lock to prevent multiple TTS instances from clashing
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150) # Set speech rate
                engine.say(text)
                engine.runAndWait() # Blocks until speech is finished
            except Exception as e:
                print(f"Failed to play voice alert on device: {e}")
                self._add_log_to_queue(f"Voice alert failed: {e}")

    def generate_frames(self):
        """
        Generator function to yield processed frames for Flask streaming.
        This function continuously reads from the video source, performs detection,
        and overlays information before yielding JPEG-encoded frames.
        """
        # Wait for the instance to be ready before starting frame generation
        if not self.is_ready:
            self._add_log_to_queue("Waiting for Margadarshi instance to be ready before generating frames...")
            start_wait = time.perf_counter()
            while not self.is_ready:
                time.sleep(0.1) # Wait briefly
                if time.perf_counter() - start_wait > 60: # Max wait for readiness (e.g., 60 seconds)
                    self._add_log_to_queue("Margadarshi instance not ready after 60 seconds, stopping frame generation attempt.")
                    return # Exit generator if not ready after long wait
            self._add_log_to_queue("Margadarshi instance is ready, starting frame generation.")


        fps_start_time = time.perf_counter()
        self.frame_count = 0 # Reset frame count for new stream session

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self._add_log_to_queue("Video capture not available or stream ended. Attempting to re-open...")
                time.sleep(1) # Wait before retrying to avoid busy-looping
                self._init_video_stream(self.config["IP_STREAM_URL"]) # Try to re-initialize
                if self.cap is None or not self.cap.isOpened():
                    self._add_log_to_queue("Failed to re-open video source. Stopping stream.")
                    self.running = False # Stop the generator if stream cannot be opened
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self._add_log_to_queue("Error: Could not read frame from video source. Stream might have ended or disconnected. Resetting video if it's a file.")
                # If it's a video file, try to loop it
                if not (isinstance(self.config["IP_STREAM_URL"], str) and self.config["IP_STREAM_URL"].isdigit()):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video file to beginning
                    continue
                else: # If it's a live camera and it fails, stop
                    self.running = False
                    continue
            
            self.frame_count += 1

            # --- Frame Pre-processing for Display and YOLO ---
            current_frame_for_display = frame.copy() # Work on a copy for display

            # Apply rotation if configured
            if self.rotate_state == 1:
                current_frame_for_display = cv2.rotate(current_frame_for_display, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotate_state == 2:
                current_frame_for_display = cv2.rotate(current_frame_for_display, cv2.ROTATE_180)
            elif self.rotate_state == 3:
                current_frame_for_display = cv2.rotate(current_frame_for_display, cv2.ROTATE_90_COUNTERCLOCKWISE)

            original_h, original_w = current_frame_for_display.shape[:2]
            display_w, display_h = self.config["DISPLAY_WIDTH"], self.config["DISPLAY_HEIGHT"]

            # Calculate scaling factor to fit original frame into display dimensions while maintaining aspect ratio
            scale = min(display_w / original_w, display_h / original_h)
            scaled_w = int(original_w * scale)
            scaled_h = int(original_h * scale)

            display_frame_resized = cv2.resize(current_frame_for_display, (scaled_w, scaled_h))

            # Create a black canvas and paste the resized frame onto it (for padding)
            canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
            x_offset = (display_w - scaled_w) // 2
            y_offset = (display_h - scaled_h) // 2
            canvas[y_offset:y_offset+scaled_h, x_offset:x_offset+scaled_w] = display_frame_resized

            annotated_frame = canvas # This will be the frame we draw on and send to the frontend

            # Apply IR mode if enabled
            if self.ir_mode:
                gray = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2GRAY)
                # Apply JET colormap for a thermal/IR effect
                annotated_frame = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

            # --- Dynamic Runway Zone Calculation ---
            # Calculate zone coordinates based on the *scaled* frame dimensions and proportional settings
            zone_top_left_x = int(self.zone_top_left_prop[0] * scaled_w) + x_offset
            zone_top_left_y = int(self.zone_top_left_prop[1] * scaled_h) + y_offset
            zone_bottom_right_x = int(self.zone_bottom_right_prop[0] * scaled_w) + x_offset
            zone_bottom_right_y = int(self.zone_bottom_right_prop[1] * scaled_h) + y_offset

            zone_top_left_coords = (zone_top_left_x, zone_top_left_y)
            zone_bottom_right_coords = (zone_bottom_right_x, zone_bottom_right_y)
            # Log these coordinates for debugging if needed, but not every frame
            # self._add_log_to_queue(f"Zone pixel coords: TL({zone_top_left_coords}), BR({zone_bottom_right_coords})")


            # --- YOLO Detection ---
            current_detections = []
            # Only attempt prediction if model is loaded and target classes are defined
            if self.model and self.config["TARGET_CLASSES"]:
                # YOLO model expects an image in the correct format (e.g., RGB numpy array) and size
                # Resize original frame for YOLO input (IMG_SIZE x IMG_SIZE)
                yolo_input_frame = cv2.resize(frame, (self.config["IMG_SIZE"], self.config["IMG_SIZE"]))
                
                results = self.model.predict(
                    yolo_input_frame,
                    conf=self.config["CONFIDENCE_THRESHOLD"],
                    iou=self.config["NMS_IOU_THRESHOLD"],
                    imgsz=self.config["IMG_SIZE"],
                    verbose=False, # Suppress verbose output from YOLO
                    device=self.device,
                    classes=self.config["TARGET_CLASSES"] # Only detect specified obstacle classes
                )

                if results and results[0].boxes:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = self.model.names[cls] # Get class name from model's names mapping

                        # Scale YOLO detection coordinates back to original frame size (before padding)
                        scaled_x1 = x1 * (original_w / self.config["IMG_SIZE"])
                        scaled_y1 = y1 * (original_h / self.config["IMG_SIZE"])
                        scaled_x2 = x2 * (original_w / self.config["IMG_SIZE"])
                        scaled_y2 = y2 * (original_h / self.config["IMG_SIZE"])

                        # Adjust coordinates to the padded display frame (canvas)
                        final_x1 = int(scaled_x1 * (scaled_w / original_w) + x_offset)
                        final_y1 = int(scaled_y1 * (scaled_h / original_h) + y_offset)
                        final_x2 = int(scaled_x2 * (scaled_w / original_w) + x_offset)
                        final_y2 = int(scaled_y2 * (scaled_h / original_h) + y_offset)

                        current_detections.append(([final_x1, final_y1, final_x2, final_y2], conf, cls))
            elif not self.model:
                self._add_log_to_queue("WARNING: YOLO model not loaded. Skipping object detection.")
            elif not self.config["TARGET_CLASSES"]:
                self._add_log_to_queue("WARNING: No TARGET_CLASSES configured. Skipping object detection.")


            # --- Simple Object Tracking Update ---
            tracked_objects = self.tracker.update(current_detections, self.frame_count)

            objects_in_zone_tracked = 0
            
            # Iterate through confirmed tracks and draw them
            for track_data in tracked_objects:
                x1, y1, x2, y2 = map(int, track_data['bbox'])
                track_id = track_data['track_id']
                detection_class_id = track_data['class']
                
                # Safely get class name, in case model.names is somehow incomplete
                detection_class_name = self.model.names.get(detection_class_id, "Unknown")

                track_color = (255, 0, 0) # Default blue for tracked objects
                
                # Check if object is an obstacle and significantly overlaps with the zone
                if detection_class_name in OBSTACLE_CLASS_NAMES:
                    is_in_zone = check_significant_overlap(
                        (x1, y1, x2, y2),
                        zone_top_left_coords,
                        zone_bottom_right_coords
                    )
                    # Log only if an object is in the zone or if it's an important status change
                    if is_in_zone:
                        self._add_log_to_queue(f"Track ID {track_id} (Class: {detection_class_name}): Bbox ({x1},{y1},{x2},{y2}), IN ZONE: {is_in_zone}")
                        objects_in_zone_tracked += 1
                        track_color = (0, 0, 255) # Red if obstacle is in zone
                # else:
                    # self._add_log_to_queue(f"Track ID {track_id} (Class: {detection_class_name}): Not an obstacle, skipping zone check.")
                    # Keep default blue color for non-obstacle tracked objects

                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), track_color, 2)
                put_text(annotated_frame, f"ID: {track_id} ({detection_class_name})", (x1, y1 - 10), track_color, font_scale=0.6)


            # --- Runway Clearance Logic & Alerts ---
            runway_clear = (objects_in_zone_tracked == 0)
            zone_color = (0, 255, 0) if runway_clear else (0, 0, 255) # Green if clear, Red if not clear
            text_status = "RUNWAY CLEAR" if runway_clear else "RUNWAY NOT CLEAR"

            if not runway_clear:
                self._play_alert_sound() # Start continuous sound if not clear
                if self.config["VISUAL_ALERT_ENABLED"]:
                    set_margadarshi_alert_active(True) # Activate global flag for frontend visual alert
                self._trigger_alert_actions(annotated_frame, objects_in_zone_tracked) # Handle screenshots/voice alerts
            else:
                self._stop_alert_sound() # Stop continuous sound if clear
                set_margadarshi_alert_active(False) # Deactivate global flag for frontend visual alert

            # Log status changes
            if not runway_clear and self.was_runway_clear:
                self._add_log_to_queue("ALERT: Runway status changed to NOT CLEAR!")
            elif runway_clear and not self.was_runway_clear:
                self._add_log_to_queue("INFO: Runway is now CLEAR.")

            self.was_runway_clear = runway_clear # Update state for next frame

            # Draw the exclusion zone rectangle
            cv2.rectangle(annotated_frame, zone_top_left_coords, zone_bottom_right_coords, zone_color, 2)

            # Display status text (pulsing if not clear and visual alerts enabled)
            if not runway_clear:
                if self.config["VISUAL_ALERT_ENABLED"] and int(time.perf_counter() * 5) % 2 == 0: # Pulsing effect
                    put_text(annotated_frame, text_status, (20, 40), zone_color, font_scale=1.2, thickness=4)
                elif self.config["VISUAL_ALERT_ENABLED"]: # Always show if enabled, but without pulse
                    put_text(annotated_frame, text_status, (20, 40), (255, 255, 255), font_scale=1.2, thickness=4)
                else: # If visual alerts disabled, just show static text
                    put_text(annotated_frame, text_status, (20, 40), zone_color, font_scale=0.9, thickness=3)
            else:
                put_text(annotated_frame, text_status, (20, 40), zone_color, font_scale=0.9, thickness=3)

            # Display count of objects in zone
            put_text(annotated_frame, f"Objects in Zone: {objects_in_zone_tracked}", (20, 70), zone_color)

            # Calculate and display FPS
            current_time = time.perf_counter()
            fps = self.frame_count / (current_time - fps_start_time) if (current_time - fps_start_time) > 0 else 0
            # Reset FPS counter every second for a rolling average
            if current_time - fps_start_time >= 1.0:
                fps_start_time = current_time
                self.frame_count = 0
            put_text(annotated_frame, f"FPS: {fps:.1f}", (annotated_frame.shape[1] - 150, 30), (255, 255, 255))

            # --- Convert to JPEG and yield ---
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                self._add_log_to_queue("ERROR: Could not encode frame to JPEG.")
                continue # Skip this frame if encoding fails
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        self.cleanup() # Ensure cleanup is called when the generator loop exits

    def cleanup(self):
        """Clean up resources (release camera, stop threads)."""
        print("Shutting down Margadarshi IDS...")
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        self._stop_alert_sound() # Ensure sound thread is stopped
        self.running = False # Ensure the main loop is signaled to stop
        print("Margadarshi System shutdown complete.")

    def stop(self):
        """Signals the main loop of the IDS to stop."""
        print("Margadarshi IDS stop requested.")
        self.running = False

    def update_config(self, new_config: dict):
        """Updates the configuration of the Margadarshi system at runtime."""
        # Update internal config dictionary
        self.config.update(new_config)
        print(f"Margadarshi config updated: {self.config}")

        # Apply changes that require re-initialization or state update
        if "IP_STREAM_URL" in new_config and new_config["IP_STREAM_URL"] != self.config["IP_STREAM_URL"]:
             self._add_log_to_queue(f"Video source change detected to {new_config['IP_STREAM_URL']}. Restarting video stream...")
             self._init_video_stream(new_config["IP_STREAM_URL"]) # Re-initialize video stream immediately

        # Update specific internal states if their corresponding config values changed
        if "CONFIDENCE_THRESHOLD" in new_config:
            self.config["CONFIDENCE_THRESHOLD"] = new_config["CONFIDENCE_THRESHOLD"]
        if "NMS_IOU_THRESHOLD" in new_config:
            self.config["NMS_IOU_THRESHOLD"] = new_config["NMS_IOU_THRESHOLD"]
        if "IR_MODE" in new_config:
            self.ir_mode = new_config["IR_MODE"]
        if "ROTATION_STATE" in new_config:
            self.rotate_state = new_config["ROTATION_STATE"]
        if "ZONE_PROPORTIONAL_TOP_LEFT" in new_config:
            self.zone_top_left_prop = list(new_config["ZONE_PROPORTIONAL_TOP_LEFT"])
        if "ZONE_PROPORTIONAL_BOTTOM_RIGHT" in new_config:
            self.zone_bottom_right_prop = list(new_config["ZONE_PROPORTIONAL_BOTTOM_RIGHT"])
        if "VOICE_ALERTS_ENABLED" in new_config:
            self.config["VOICE_ALERTS_ENABLED"] = new_config["VOICE_ALERTS_ENABLED"]
        if "VISUAL_ALERT_ENABLED" in new_config:
            self.config["VISUAL_ALERT_ENABLED"] = new_config["VISUAL_ALERT_ENABLED"]
        if "AUTO_SCREENSHOT_ENABLED" in new_config:
            self.config["AUTO_SCREENSHOT_ENABLED"] = new_config["AUTO_SCREENSHOT_ENABLED"]
        if "ALERT_SOUND_ENABLED" in new_config:
            self.config["ALERT_SOUND_ENABLED"] = new_config["ALERT_SOUND_ENABLED"]
            if not self.config["ALERT_SOUND_ENABLED"]:
                self._stop_alert_sound() # Immediately stop sound if disabled

        self._add_log_to_queue("Margadarshi configuration updated.")

    def get_current_zone_pixels(self):
        """
        Returns the current zone coordinates in pixel values based on the current
        frame resolution and display scaling. This is used by the frontend to draw the overlay.
        """
        if self.cap is None or not self.cap.isOpened():
            # If camera not open, return default/zero coordinates
            return {'x': 0, 'y': 0, 'width': 0, 'height': 0}

        # It's best to get the actual frame dimensions from the cap object after a successful read
        # to ensure accuracy, rather than relying solely on configured display dimensions.
        # We temporarily read a frame to get its dimensions.
        ret, frame = self.cap.read()
        if not ret:
            self._add_log_to_queue("Warning: Could not read frame to get current dimensions for zone calculation. Using configured display dimensions.")
            original_w, original_h = self.config["DISPLAY_WIDTH"], self.config["DISPLAY_HEIGHT"]
        else:
            original_h, original_w = frame.shape[:2]
            # Rewind the frame if we read it just for dimensions and it's a file
            if not (isinstance(self.config["IP_STREAM_URL"], str) and self.config["IP_STREAM_URL"].isdigit()):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)

        display_w, display_h = self.config["DISPLAY_WIDTH"], self.config["DISPLAY_HEIGHT"]

        # Calculate scaling factor and offsets as done in generate_frames
        scale = min(display_w / original_w, display_h / original_h)
        scaled_w = int(original_w * scale)
        scaled_h = int(original_h * scale)

        x_offset = (display_w - scaled_w) // 2
        y_offset = (display_h - scaled_h) // 2

        # Calculate zone pixel coordinates based on proportional values and current display scaling
        zone_x1 = int(self.zone_top_left_prop[0] * scaled_w) + x_offset
        zone_y1 = int(self.zone_top_left_prop[1] * scaled_h) + y_offset
        zone_x2 = int(self.zone_bottom_right_prop[0] * scaled_w) + x_offset
        zone_y2 = int(self.zone_bottom_right_prop[1] * scaled_h) + y_offset

        return {
            'x': zone_x1,
            'y': zone_y1,
            'width': zone_x2 - zone_x1,
            'height': zone_y2 - zone_y1
        }

    def set_zone_proportional(self, tl_x_prop, tl_y_prop, br_x_prop, br_y_prop):
        """Sets the zone using proportional coordinates (0.0 to 1.0)."""
        # Clamp values to be within 0.0 and 1.0
        self.zone_top_left_prop[0] = max(0.0, min(1.0, tl_x_prop))
        self.zone_top_left_prop[1] = max(0.0, min(1.0, tl_y_prop))
        self.zone_bottom_right_prop[0] = max(0.0, min(1.0, br_x_prop))
        self.zone_bottom_right_prop[1] = max(0.0, min(1.0, br_y_prop))
        self._add_log_to_queue(f"Zone set proportionally: TL({self.zone_top_left_prop[0]:.2f},{self.zone_top_left_prop[1]:.2f}), BR({self.zone_bottom_right_prop[0]:.2f},{self.zone_bottom_right_prop[1]:.2f})")

    def reset_zone_to_default(self):
        """Resets the zone to its initial default proportional configuration."""
        self.zone_top_left_prop = list(DEFAULT_MARGADARSHI_CONFIG["ZONE_PROPORTIONAL_TOP_LEFT"])
        self.zone_bottom_right_prop = list(DEFAULT_MARGADARSHI_CONFIG["ZONE_PROPORTIONAL_BOTTOM_RIGHT"])
        self._add_log_to_queue("Zone reset to default.")
