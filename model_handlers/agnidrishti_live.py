import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO
import threading
import numpy as np
import pyttsx3 # For text-to-speech
import json
from queue import Queue # Import Queue for thread-safe logging

# --- Paths ---
# Adjust paths relative to the project root where app.py is run
BASE_YOLO_MODEL_PATH = os.path.join('yolov8_models', 'yolov8x.pt')
FIRE_YOLO_MODEL_PATH = os.path.join('yolov8_models', 'yolofirenew.pt')

# --- CONFIGURATION (Default values, can be overridden by Flask config) ---
DEFAULT_CONFIG = {
    'camera_source': 0,
    'rotation_angle': 0,
    'ir_mode': False,
    'voice_alerts_enabled': True,
    'voice_gender': 'male',
    'detection_cooldown_seconds': 10
}

HUMAN_CLASS_NAME = 'person'
VEHICLE_CLASS_NAMES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
FIRE_CLASS_NAME = 'Fire'

CONFIDENCE_THRESHOLD_BASE = 0.5
CONFIDENCE_THRESHOLD_FIRE = 0.3

LOGS_DIRECTORY = 'detection_logs_live'
LOG_FILE_NAME_PREFIX = 'live_detection_log_'

# Global variables for this module, managed by the AgnidrishtiCamera class
_log_queue = Queue() # Queue for sending logs to Flask SSE/SocketIO
_tts_lock = threading.Lock() # Lock for TTS engine to prevent concurrent access issues
_last_alert_time = 0 # Last detection time for cooldown

def get_agnidrishti_log_queue():
    """Provides access to the log queue for Flask's SocketIO."""
    return _log_queue

class AgnidrishtiCamera:
    def __init__(self, config=None):
        self.config = config if config is not None else DEFAULT_CONFIG.copy()
        print(f"AgnidrishtiCamera initialized with config: {self.config}")

        self.running = True
        self.cap = None
        self.log_file_handle = None
        
        self.base_model = None
        self.fire_model = None

        self.human_id = None
        self.vehicle_ids = []
        self.fire_id = None

        self._load_models()
        self._open_camera()

    def _load_models(self):
        try:
            # Ensure model paths are absolute or relative to the script's location
            base_model_abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", BASE_YOLO_MODEL_PATH)
            fire_model_abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", FIRE_YOLO_MODEL_PATH)

            if os.path.exists(base_model_abs_path):
                self.base_model = YOLO(base_model_abs_path)
                self._log_event("✅ Base model loaded successfully.", to_console=True)
            else:
                self._log_event(f"❌ Base model not found at {base_model_abs_path}. Human/Vehicle detection will be skipped.", to_console=True)

            if os.path.exists(fire_model_abs_path):
                self.fire_model = YOLO(fire_model_abs_path)
                self._log_event("✅ Fire model loaded successfully.", to_console=True)
            else:
                self._log_event(f"❌ Fire model not found at {fire_model_abs_path}. Fire detection will be skipped.", to_console=True)

            if self.base_model is None and self.fire_model is None:
                self._log_event("❌ No YOLO models could be loaded. Detection will not function.", to_console=True)
                self.running = False # Cannot run without models

            self.human_id = self._get_class_id(self.base_model, HUMAN_CLASS_NAME) if self.base_model else None
            self.vehicle_ids = [self._get_class_id(self.base_model, vn) for vn in VEHICLE_CLASS_NAMES if self.base_model and self._get_class_id(self.base_model, vn) is not None] if self.base_model else []
            self.fire_id = self._get_class_id(self.fire_model, FIRE_CLASS_NAME) if self.fire_model else None

            if self.fire_id is None and self.fire_model:
                self._log_event(f"CRITICAL ERROR: Fire class '{FIRE_CLASS_NAME}' not found in fire model. Fire detection will not function.", to_console=True)

        except Exception as e:
            self._log_event(f"❌ Model load error: {e}. Please check model paths and integrity.", to_console=True)
            self.running = False

    def _open_camera(self):
        camera_source = self.config['camera_source']
        try:
            if isinstance(camera_source, str) and camera_source.isdigit():
                self.cap = cv2.VideoCapture(int(camera_source))
            else:
                self.cap = cv2.VideoCapture(camera_source)

            if isinstance(camera_source, str) and (camera_source.startswith("rtsp://") or camera_source.startswith("http://")):
                self._log_event(f"Attempting to open IP camera stream: {camera_source}", to_console=True)
                time.sleep(2)
                for i in range(3): # Retry mechanism for IP cameras
                    if self.cap.isOpened():
                        self._log_event(f"Successfully opened IP camera connection on attempt {i+1}.", to_console=True)
                        break
                    self._log_event(f"Retrying IP camera connection (attempt {i+1}/3)...", to_console=True)
                    self.cap.release() # Release before retrying
                    time.sleep(3)
                    if isinstance(camera_source, str) and camera_source.isdigit():
                        self.cap = cv2.VideoCapture(int(camera_source))
                    else:
                        self.cap = cv2.VideoCapture(camera_source)
                if not self.cap.isOpened():
                    self._log_event(f"Failed to open IP camera source '{camera_source}' after multiple retries.", to_console=True)

            if not self.cap.isOpened():
                error_message = f"CRITICAL ERROR: Failed to open camera/video source '{camera_source}'. "
                if isinstance(camera_source, int):
                    error_message += "Check if webcam is connected and not in use by another application."
                elif isinstance(camera_source, str) and os.path.exists(camera_source):
                    error_message += "Check if video file path is correct and accessible."
                elif isinstance(camera_source, str) and (camera_source.startswith("rtsp://") or camera_source.startswith("http://")):
                    error_message += "Check IP camera URL, credentials, network connectivity, or firewall settings."
                else:
                    error_message += "Invalid camera source or access denied."
                self._log_event(error_message, to_console=True)
                self.running = False
            else:
                self._log_event(f"Camera/video source '{camera_source}' opened successfully.", to_console=True)

        except Exception as e:
            self._log_event(f"Error opening camera source {camera_source}: {e}", to_console=True)
            self.running = False

    def _get_class_id(self, model, class_name):
        """Safely gets class ID from model names, logging if not found."""
        if not hasattr(model, 'names') or not isinstance(model.names, dict):
            self._log_event(f"Warning: Model does not have a valid 'names' attribute. Cannot lookup class '{class_name}'.")
            return None
        
        for class_id, name in model.names.items():
            if name.strip().lower() == class_name.strip().lower():
                return class_id
        self._log_event(f"Warning: Class '{class_name}' not found in model names.")
        return None

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
                    self.log_file_handle.write(f"--- Live Detection Log Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                    self.log_file_handle.write(f"Camera Source: {self.config['camera_source']}\n")
                    self.log_file_handle.write(f"Rotation Angle: {self.config['rotation_angle']}\n")
                    self.log_file_handle.write(f"IR Mode: {self.config['ir_mode']}\n")
                    self.log_file_handle.write(f"Base Model: {os.path.basename(BASE_YOLO_MODEL_PATH)}\n")
                    self.log_file_handle.write(f"Fire Model: {os.path.basename(FIRE_YOLO_MODEL_PATH)}\n")
                    self.log_file_handle.write("-" * 60 + "\n")
                    self._log_event(f"Logging to: {log_file_path}", to_file=False, to_console=to_console, to_queue=to_queue) # Avoid infinite recursion
                except Exception as e:
                    print(f"Error opening log file: {e}. File logging disabled for this session.")
                    self.log_file_handle = None
            if self.log_file_handle and not self.log_file_handle.closed:
                try:
                    self.log_file_handle.write(log_msg + "\n")
                    self.log_file_handle.flush()
                except Exception as e:
                    print(f"Error writing to log file from AgnidrishtiCamera: {e}")
        
        if to_console:
            print(log_msg)

        if to_queue:
            try:
                _log_queue.put(log_msg)
            except Exception as e:
                print(f"Error putting message into _log_queue for AgnidrishtiCamera: {e}")

    def _play_voice_alert(self, text, voice_gender):
        """Converts text to speech and plays it directly on the device using pyttsx3."""
        with _tts_lock: # Acquire lock before using TTS engine
            try:
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                
                selected_voice_id = None
                for voice in voices:
                    # Heuristic to find male/female voices across different OS
                    if voice_gender.lower() == 'male':
                        if 'male' in voice.name.lower() or 'david' in voice.name.lower() or 'zira' in voice.name.lower(): # Zira can be male too sometimes
                            selected_voice_id = voice.id
                            break
                    elif voice_gender.lower() == 'female':
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower() or 'helen' in voice.name.lower():
                            selected_voice_id = voice.id
                            break
                
                if selected_voice_id:
                    engine.setProperty('voice', selected_voice_id)
                else:
                    self._log_event(f"Warning: No suitable '{voice_gender}' voice found. Using default voice.", to_file=False)

                engine.say(text)
                engine.runAndWait()
                self._log_event(f"Voice alert played on device: '{text}'", to_file=False)
            except Exception as e:
                self._log_event(f"Failed to play voice alert on device: {e}", to_file=False)

    def gen_frames(self):
        """Generator function to capture frames, process them, and yield JPEG-encoded frames."""
        global _last_alert_time # Access the global last_alert_time for cooldown

        self._log_event(f"Starting Agnidrishti live detection stream for source: {self.config['camera_source']} with rotation={self.config['rotation_angle']} and IR_Mode={self.config['ir_mode']}", to_file=True)

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self._log_event("Camera/video source not available or stream ended. Attempting to re-open...", to_file=False)
                time.sleep(1) # Wait before retrying
                try:
                    # Attempt to re-open the original video source
                    camera_source = self.config['camera_source']
                    if isinstance(camera_source, str) and camera_source.isdigit():
                        self.cap = cv2.VideoCapture(int(camera_source))
                    else:
                        self.cap = cv2.VideoCapture(camera_source)
                    
                    if not self.cap.isOpened():
                        raise IOError("Failed to re-open video source.")
                    self._log_event("Video source re-opened successfully.", to_file=False)
                except Exception as e:
                    self._log_event(f"Failed to re-open video source: {e}. Stopping stream.", to_file=False)
                    self.running = False
                continue

            ret, frame = self.cap.read()
            if not ret:
                self._log_event("Failed to grab frame or end of video stream. Stopping stream.", to_file=False)
                break

            if frame is None:
                self._log_event("Frame is None. Skipping frame.", to_file=False)
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

            # Run base model detection
            if self.base_model and (self.human_id is not None or self.vehicle_ids):
                base_classes_to_detect = []
                if self.human_id is not None:
                    base_classes_to_detect.append(self.human_id)
                base_classes_to_detect.extend(self.vehicle_ids)

                if base_classes_to_detect:
                    try:
                        results_base = self.base_model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD_BASE, classes=base_classes_to_detect)
                        for r in results_base:
                            frame = r.plot() # YOLO's plot function draws boxes on the frame
                            for box in r.boxes:
                                class_id = int(box.cls[0])
                                class_name = self.base_model.names[class_id]
                                if class_name.lower() == HUMAN_CLASS_NAME.lower():
                                    msg = f"Person detected"
                                    detection_messages.append(msg)
                                elif class_name.lower() in [c.lower() for c in VEHICLE_CLASS_NAMES]:
                                    msg = f"Vehicle ({class_name}) detected"
                                    detection_messages.append(msg)
                    except Exception as e:
                        self._log_event(f"Error during base model inference: {e}", to_file=False)

            # Run fire model detection
            if self.fire_model and self.fire_id is not None:
                try:
                    results_fire = self.fire_model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD_FIRE, classes=[self.fire_id])
                    for r in results_fire:
                        frame = r.plot()
                        if len(r.boxes) > 0:
                            msg = f"🔥🔥🔥 Fire detected 🔥🔥🔥"
                            detection_messages.append(msg)
                except Exception as e:
                    self._log_event(f"Error during fire model inference: {e}", to_file=False)

            # Handle alerts if detections occurred and cooldown allows
            if detection_messages:
                current_time = time.time()
                if current_time - _last_alert_time > self.config['detection_cooldown_seconds']:
                    full_alert_message = f"ALERT: {', '.join(detection_messages)}"
                    self._log_event(full_alert_message, to_file=True) # Log the combined alert

                    # Trigger voice alert on the device (in a separate thread)
                    if self.config['voice_alerts_enabled']:
                        voice_thread = threading.Thread(target=self._play_voice_alert, args=(full_alert_message, self.config['voice_gender']))
                        voice_thread.daemon = True
                        voice_thread.start()
                    
                    _last_alert_time = current_time # Reset cooldown timer

            # Encode the frame to JPEG bytes and yield it
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                self._log_event("Failed to encode frame to JPEG. Skipping frame.", to_file=False)
                continue
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        self.cleanup() # Ensure cleanup when the generator stops

    def cleanup(self):
        """Releases resources."""
        print("AgnidrishtiCamera cleanup initiated.")
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        if self.log_file_handle:
            if not self.log_file_handle.closed:
                self.log_file_handle.write("-" * 60 + "\n")
                self.log_file_handle.write(f"--- Live Detection Stream Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                self.log_file_handle.close()
            self.log_file_handle = None
        self._log_event("✅ Agnidrishti live detection stream shut down.", to_console=True, to_file=False)

    def stop(self):
        """Stops the main loop of the camera."""
        print("AgnidrishtiCamera stop requested.")
        self.running = False

    def update_config(self, new_config):
        """Updates the internal configuration and triggers a restart if necessary."""
        # Only update relevant config items
        for key in ['camera_source', 'rotation_angle', 'ir_mode', 'voice_alerts_enabled', 'voice_gender', 'detection_cooldown_seconds']:
            if key in new_config:
                self.config[key] = new_config[key]
        
        # If camera source or IR mode changed, a restart of the stream is typically needed
        # For simplicity, we'll let the Flask route handle restarting the generator
        # by re-initializing the camera when the page is reloaded or navigated to.
        print(f"AgnidrishtiCamera config updated to: {self.config}")