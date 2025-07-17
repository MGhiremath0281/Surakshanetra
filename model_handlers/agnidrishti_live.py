import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO
import threading
import numpy as np
import pyttsx3 # For text-to-speech
from queue import Queue # Import Queue for thread-safe logging
import logging # Import logging module

# Configure logging for AgnidrishtiCamera module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Paths ---
# Adjust paths relative to the project root where app.py is run
MODEL_PATH = os.path.join('yolov8_models', 'yolofirenew.pt') # Assuming this is the primary model for Agnidrishti
ALERT_SOUND_PATH = 'alert.wav' # Assuming alert.wav is in the root directory
ALERT_SCREENSHOT_DIR = 'static/agnidrishti_alerts' # Directory for screenshots

# --- CONFIGURATION (Default values, can be overridden by Flask config) ---
DEFAULT_AGNIDRISHTI_CONFIG = {
    'camera_source': 0,
    'rotation_angle': 0, # 0, 90, 180, 270
    'ir_mode': False,
    'voice_alerts_enabled': True,
    'voice_gender': 'male', # 'male' or 'female'
    'detection_cooldown_seconds': 10, # Cooldown period between voice alerts for the same detection
    'DISPLAY_WIDTH': 1280, # Desired display width for the stream
    'DISPLAY_HEIGHT': 720, # Desired display height for the stream
    'DESIRED_CAMERA_FPS': 30, # Desired frames per second from the camera
    'PLAYBACK_SPEED': 1.0, # Multiplier for playback speed (1.0 = real-time)
    'CONFIDENCE_THRESHOLD': 0.5, # Confidence threshold for object detection
    'NMS_IOU_THRESHOLD': 0.7, # Non-Maximum Suppression IOU threshold
}

LOGS_DIRECTORY = 'detection_logs_agnidrishti'
LOG_FILE_NAME_PREFIX = 'agnidrishti_detection_log_'

# Global variables for this module, managed by the AgnidrishtiCamera class
_log_queue = Queue() # Queue for sending logs to Flask SSE/SocketIO
_tts_lock = threading.Lock() # Lock for TTS engine to prevent concurrent access issues
_last_alert_time = 0 # Last detection time for cooldown
_alert_active = False # Flag to indicate if an alert is currently active

def get_agnidrishti_log_queue():
    """Provides access to the log queue for Flask's SocketIO."""
    return _log_queue

def get_agnidrishti_alert_active():
    """Provides the current alert status for Agnidrishti."""
    return _alert_active

def set_agnidrishti_alert_active(status):
    """Sets the alert status for Agnidrishti (used for external control if needed)."""
    global _alert_active
    _alert_active = status

class AgnidrishtiCamera:
    def __init__(self, config=None):
        self.config = config if config is not None else DEFAULT_AGNIDRISHTI_CONFIG.copy()
        logger.info(f"AgnidrishtiCamera initialized with config: {self.config}")

        self.running = True
        self.cap = None
        self.log_file_handle = None
        self._is_ready = False # Flag to indicate if the instance is fully initialized and ready to stream
        
        self.model = None # Single model for Agnidrishti

        self._load_model()
        self._open_camera() 

        # Set _is_ready only if model is loaded and video stream is opened
        if self.model and self.cap and self.cap.isOpened():
            self._is_ready = True
            logger.info("AgnidrishtiCamera is now fully ready.")
            self._log_event("AgnidrishtiCamera is now fully ready.", to_console=False) # Log to queue
        else:
            self._is_ready = False
            logger.error("AgnidrishtiCamera failed to become ready. Check previous logs for details.")
            self._log_event("ERROR: AgnidrishtiCamera failed to initialize. Check logs.", to_console=False) # Log to queue

    @property
    def is_ready(self):
        """Property to check if the AgnidrishtiCamera instance is fully ready for streaming."""
        return self._is_ready

    def _load_model(self):
        try:
            model_abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", MODEL_PATH)

            if os.path.exists(model_abs_path):
                self.model = YOLO(model_abs_path)
                logger.info(f"Agnidrishti model '{MODEL_PATH}' loaded successfully.")
                self._log_event(f"✅ Agnidrishti model '{MODEL_PATH}' loaded successfully.", to_console=False)
            else:
                logger.error(f"Agnidrishti model not found at {model_abs_path}. Detection will be skipped.")
                self._log_event(f"❌ Agnidrishti model not found at {model_abs_path}. Detection will be skipped.", to_console=False)
                self.running = False # Cannot run without model

        except Exception as e:
            logger.exception(f"Agnidrishti model load error: {e}. Please check model path and integrity.")
            self._log_event(f"❌ Agnidrishti model load error: {e}. Please check model path and integrity.", to_console=False)
            self.running = False

    def _open_camera(self):
        """Initializes or re-initializes the video stream using self.config['camera_source']."""
        if self.cap:
            self.cap.release() # Release existing stream if any
            self.cap = None

        camera_source = self.config['camera_source']
        try:
            is_webcam = isinstance(camera_source, str) and camera_source.isdigit()

            if is_webcam:
                self.cap = cv2.VideoCapture(int(camera_source))
            else:
                self.cap = cv2.VideoCapture(camera_source)

            # --- IP Camera Retry Logic ---
            if isinstance(camera_source, str) and (camera_source.startswith("rtsp://") or camera_source.startswith("http://")):
                logger.info(f"Attempting to open IP camera stream: {camera_source}")
                self._log_event(f"Attempting to open IP camera stream: {camera_source}", to_console=True)
                time.sleep(2) # Initial wait for IP camera to establish connection
                for i in range(3): # Retry mechanism for IP cameras
                    if self.cap.isOpened():
                        logger.info(f"Successfully opened IP camera connection on attempt {i+1}.")
                        self._log_event(f"Successfully opened IP camera connection on attempt {i+1}.", to_console=True)
                        break
                    logger.warning(f"Retrying IP camera connection (attempt {i+1}/3)...")
                    self._log_event(f"Retrying IP camera connection (attempt {i+1}/3)...", to_console=True)
                    self.cap.release() # Release before retrying
                    time.sleep(3) # Wait before next retry
                    if is_webcam: # Re-evaluate for retry
                        self.cap = cv2.VideoCapture(int(camera_source))
                    else:
                        self.cap = cv2.VideoCapture(camera_source)
                if not self.cap.isOpened():
                    logger.error(f"Failed to open IP camera source '{camera_source}' after multiple retries.")
                    self._log_event(f"Failed to open IP camera source '{camera_source}' after multiple retries.", to_console=True)
            # --- End IP Camera Retry Logic ---

            if not self.cap or not self.cap.isOpened(): # Check if cap is None or not opened
                error_message = f"CRITICAL ERROR: Failed to open camera/video source '{camera_source}'. "
                if is_webcam:
                    error_message += "Check if webcam is connected and not in use by another application."
                elif isinstance(camera_source, str) and os.path.exists(camera_source):
                    error_message += "Check if video file path is correct and accessible."
                elif isinstance(camera_source, str) and (camera_source.startswith("rtsp://") or camera_source.startswith("http://")):
                    error_message += "Check IP camera URL, credentials, network connectivity, or firewall settings."
                else:
                    error_message += "Invalid camera source or access denied."
                logger.critical(error_message)
                self._log_event(error_message, to_console=True)
                self.running = False
            else:
                # Only set camera properties if it's a live webcam
                if is_webcam:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["DISPLAY_WIDTH"])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["DISPLAY_HEIGHT"])
                    self.cap.set(cv2.CAP_PROP_FPS, self.config["DESIRED_CAMERA_FPS"])
                    logger.info(f"Set webcam properties: Width={self.config['DISPLAY_WIDTH']}, Height={self.config['DISPLAY_HEIGHT']}, FPS={self.config['DESIRED_CAMERA_FPS']}")
                    self._log_event(f"Set webcam properties: Width={self.config['DISPLAY_WIDTH']}, Height={self.config['DISPLAY_HEIGHT']}, FPS={self.config['DESIRED_CAMERA_FPS']}", to_console=False)

                # Read first frame to get actual dimensions and ensure stream is alive
                ret, frame = self.cap.read()
                if not ret:
                    raise IOError("Could not read the first frame from the video source. Stream might be invalid or disconnected.")
                
                # If it's a file, reset to the beginning after reading the first frame
                if not is_webcam: # If it's a file, not a webcam
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                logger.info(f"Camera/video source '{camera_source}' opened successfully. Actual Resolution: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
                self._log_event(f"Camera/video source '{camera_source}' opened successfully. Actual Resolution: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}", to_console=False)

        except Exception as e:
            logger.error(f"Error opening camera source {camera_source}: {e}")
            self._log_event(f"CRITICAL ERROR: Failed to open camera/video source '{camera_source}': {e}", to_console=True)
            self.cap = None # Indicate failure
            self.running = False
            
    def _log_event(self, msg, to_file=True, to_console=True, to_queue=True):
        """Logs an event, writes to file, prints to console, and sends to SSE queue."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_msg = f"[{ts}] {msg}"

        if to_file:
            if self.log_file_handle is None or self.log_file_handle.closed:
                try:
                    os.makedirs(LOGS_DIRECTORY, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_file_path = os.path.join(LOGS_DIRECTORY, f"{LOG_FILE_NAME_PREFIX}{timestamp}.txt")
                    self.log_file_handle = open(log_file_path, 'a', encoding='utf-8')
                    self.log_file_handle.write(f"--- Agnidrishti Live Detection Log Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                    self.log_file_handle.write(f"Camera Source: {self.config['camera_source']}\n")
                    self.log_file_handle.write(f"Rotation Angle: {self.config['rotation_angle']}\n")
                    self.log_file_handle.write(f"IR Mode: {self.config['ir_mode']}\n")
                    self.log_file_handle.write(f"Model: {os.path.basename(MODEL_PATH)}\n")
                    self.log_file_handle.write("-" * 60 + "\n")
                    self._log_event(f"Logging to: {log_file_path}", to_file=False, to_console=to_console, to_queue=to_queue) # Avoid infinite recursion
                except Exception as e:
                    logger.error(f"Error opening log file: {e}. File logging disabled for this session.")
                    self.log_file_handle = None
            if self.log_file_handle and not self.log_file_handle.closed:
                try:
                    self.log_file_handle.write(log_msg + "\n")
                    self.log_file_handle.flush()
                except Exception as e:
                    logger.error(f"Error writing to log file from AgnidrishtiCamera: {e}")
        
        if to_console:
            print(log_msg)

        if to_queue:
            try:
                _log_queue.put(log_msg)
            except Exception as e:
                logger.error(f"Error putting message into _log_queue for AgnidrishtiCamera: {e}")

    def _play_voice_alert(self, text, voice_gender):
        """Converts text to speech and plays it directly on the device."""
        with _tts_lock: # Acquire lock before using TTS engine
            if not self.config['voice_alerts_enabled']:
                return
            try:
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                
                selected_voice_id = None
                for voice in voices:
                    if voice_gender.lower() == 'male':
                        # Prioritize a common male voice, e.g., David on Windows
                        if 'male' in voice.name.lower() or 'david' in voice.name.lower() or voice.id.endswith('Microsoft SAPI5 English (United States) - David'):
                            selected_voice_id = voice.id
                            break
                    elif voice_gender.lower() == 'female':
                        # Prioritize a common female voice, e.g., Zira on Windows
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower() or voice.id.endswith('Microsoft SAPI5 English (United States) - Zira'):
                            selected_voice_id = voice.id
                            break
                
                if selected_voice_id:
                    engine.setProperty('voice', selected_voice_id)
                else:
                    logger.warning(f"No suitable '{voice_gender}' voice found. Using default voice.")
                    self._log_event(f"Warning: No suitable '{voice_gender}' voice found. Using default voice.", to_file=False, to_console=False)

                engine.say(text)
                engine.runAndWait()
                logger.info(f"Voice alert played on device: '{text}'")
                self._log_event(f"Voice alert played on device: '{text}'", to_file=False, to_console=False)
            except Exception as e:
                logger.error(f"Failed to play voice alert on device: {e}")
                self._log_event(f"Failed to play voice alert on device: {e}", to_file=False, to_console=False)

    def gen_frames(self):
        """Generator function to capture frames, process them, and yield JPEG-encoded frames."""
        global _last_alert_time, _alert_active # Access global variables for cooldown and alert status

        if not self.is_ready:
            self._log_event("Waiting for AgnidrishtiCamera instance to be ready before generating frames...", to_console=False)
            start_wait = time.perf_counter()
            while not self.is_ready:
                time.sleep(0.1)
                if time.perf_counter() - start_wait > 60:
                    self._log_event("AgnidrishtiCamera instance not ready after 60 seconds, stopping frame generation attempt.", to_console=False)
                    return
            self._log_event("AgnidrishtiCamera instance is ready, starting frame generation.", to_console=False)

        logger.info(f"Starting Agnidrishti live detection stream for source: {self.config['camera_source']} with rotation={self.config['rotation_angle']} and IR_Mode={self.config['ir_mode']}")
        self._log_event(f"Starting Agnidrishti live detection stream for source: {self.config['camera_source']} with rotation={self.config['rotation_angle']} and IR_Mode={self.config['ir_mode']}", to_console=False)

        actual_camera_fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap and self.cap.isOpened() else self.config["DESIRED_CAMERA_FPS"]
        if actual_camera_fps <= 0: # Fallback if CAP_PROP_FPS returns 0 or negative
            actual_camera_fps = self.config["DESIRED_CAMERA_FPS"]

        base_frame_delay = 1.0 / actual_camera_fps if actual_camera_fps > 0 else 0.01 # Default to 100ms if FPS is zero

        while self.running:
            frame_processing_start_time = time.perf_counter() # Measure start time for current frame

            if self.cap is None or not self.cap.isOpened():
                self._log_event("Camera/video source not available or stream ended. Attempting to re-open...", to_console=False)
                time.sleep(1) # Wait before retrying
                self._open_camera() # Re-initialize camera
                if self.cap is None or not self.cap.isOpened():
                    self._log_event(f"Failed to re-open video source. Stopping stream.", to_console=False)
                    self.running = False
                continue

            ret, frame = self.cap.read()
            if not ret:
                self._log_event("Failed to grab frame or end of video stream. Resetting video if it's a file.", to_console=False)
                # If it's a file, reset to beginning; otherwise, stop.
                if not (isinstance(self.config["camera_source"], str) and self.config["camera_source"].isdigit()):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.1) # Small pause to prevent tight loop on file end
                    continue
                else:
                    self.running = False # Stop if it's a live camera and stream fails
                    continue

            if frame is None:
                self._log_event("Frame is None. Skipping frame.", to_console=False)
                continue

            # Apply rotation
            if self.config['rotation_angle'] == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.config['rotation_angle'] == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.config['rotation_angle'] == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Apply IR mode
            if self.config['ir_mode']:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)

            detection_messages = []
            current_frame_has_detection = False

            # Run model detection
            if self.model:
                try:
                    # Agnidrishti is typically for fire/smoke, so we might not need TARGET_CLASSES here
                    # If you need to specify classes for Agnidrishti, you would add a 'TARGET_CLASSES' key to its config
                    results = self.model(frame, verbose=False, conf=self.config['CONFIDENCE_THRESHOLD'], iou=self.config['NMS_IOU_THRESHOLD'])
                    for r in results:
                        frame = r.plot() # YOLO's plot function draws boxes on the frame
                        if len(r.boxes) > 0:
                            current_frame_has_detection = True
                            for box in r.boxes:
                                class_id = int(box.cls[0])
                                class_name = self.model.names[class_id]
                                detection_messages.append(f"Detected: {class_name}")
                except Exception as e:
                    logger.error(f"Error during Agnidrishti model inference: {e}")
                    self._log_event(f"Error during Agnidrishti model inference: {e}", to_console=False)

            # Handle alerts if detections occurred and cooldown allows
            if current_frame_has_detection:
                current_time = time.time()
                if current_time - _last_alert_time > self.config['detection_cooldown_seconds']:
                    full_alert_message = f"AGNIDRISHTI ALERT: {', '.join(detection_messages)}"
                    logger.warning(full_alert_message) # Log the combined alert
                    self._log_event(full_alert_message, to_console=True) # Log to queue and console

                    # Trigger voice alert on the device (in a separate thread)
                    if self.config['voice_alerts_enabled']:
                        voice_thread = threading.Thread(target=self._play_voice_alert, args=(full_alert_message, self.config['voice_gender']))
                        voice_thread.daemon = True
                        voice_thread.start()
                    
                    _last_alert_time = current_time # Reset cooldown timer
                    _alert_active = True # Set alert active flag
            else:
                _alert_active = False # No detection, so alert is not active

            # Encode the frame to JPEG bytes and yield it
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logger.error("Failed to encode frame to JPEG. Skipping frame.")
                self._log_event("Failed to encode frame to JPEG. Skipping frame.", to_console=False)
                continue
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Calculate time taken for this frame's processing and adjust sleep
            frame_processing_end_time = time.perf_counter()
            time_taken = frame_processing_end_time - frame_processing_start_time
            
            # Target delay for display, considering playback speed
            target_display_delay = base_frame_delay / self.config["PLAYBACK_SPEED"]

            sleep_time = max(0, target_display_delay - time_taken)
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.cleanup() # Ensure cleanup when the generator stops

    def cleanup(self):
        """Releases resources."""
        logger.info("AgnidrishtiCamera cleanup initiated.")
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        if self.log_file_handle:
            if not self.log_file_handle.closed:
                self.log_file_handle.write("-" * 60 + "\n")
                self.log_file_handle.write(f"--- Agnidrishti Live Detection Stream Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                self.log_file_handle.close()
            self.log_file_handle = None
        self._log_event("✅ Agnidrishti live detection stream shut down.", to_console=True, to_file=False)

    def stop(self):
        """Stops the main loop of the camera."""
        logger.info("AgnidrishtiCamera stop requested.")
        self.running = False

    def update_config(self, new_config):
        """Updates the internal configuration and triggers a restart if necessary."""
        # Only update relevant config items
        for key in ['camera_source', 'rotation_angle', 'ir_mode', 'voice_alerts_enabled', 'voice_gender', 'detection_cooldown_seconds',
                    'DISPLAY_WIDTH', 'DISPLAY_HEIGHT', 'DESIRED_CAMERA_FPS', 'PLAYBACK_SPEED',
                    'CONFIDENCE_THRESHOLD', 'NMS_IOU_THRESHOLD']: 
            if key in new_config:
                self.config[key] = new_config[key]
        
        # If camera source, display settings or FPS changed, re-open the camera
        if any(key in new_config for key in ["camera_source", "DISPLAY_WIDTH", "DISPLAY_HEIGHT", "DESIRED_CAMERA_FPS"]):
            self._log_event(f"Camera source or display settings changed. Restarting video stream...", to_console=False)
            self._open_camera() 
        
        # Agnidrishti doesn't have a MODEL_PATH in its config, so no model reload logic needed here.
        # If you were to add a configurable model path, you'd add similar logic to Shankasoochi's.

        logger.info(f"AgnidrishtiCamera config updated to: {self.config}")
        self._log_event(f"AgnidrishtiCamera config updated.", to_console=False)
