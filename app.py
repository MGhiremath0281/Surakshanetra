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
BASE_YOLO_MODEL_PATH = os.path.join('yolov8_models', 'yolov8x.pt')
FIRE_YOLO_MODEL_PATH = os.path.join('yolov8_models', 'yolofirenew.pt')

# --- CONFIGURATION ---
DEFAULT_CONFIG = {
    'camera_source': 0,
    'rotation_angle': 0,
    'ir_mode': False,
    'voice_alerts_enabled': True,  # Default to True for on-device voice alerts
    'voice_gender': 'male', # Male or female voice
    'detection_cooldown_seconds': 10 # Cooldown to prevent spamming alerts
}

HUMAN_CLASS_NAME = 'person'
VEHICLE_CLASS_NAMES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
FIRE_CLASS_NAME = 'Fire'

CONFIDENCE_THRESHOLD_BASE = 0.5
CONFIDENCE_THRESHOLD_FIRE = 0.3

LOGS_DIRECTORY = 'detection_logs_live'
LOG_FILE_NAME_PREFIX = 'live_detection_log_'

LOG_FILE_HANDLE = None

# --- Flask App Setup ---
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify

app = Flask(__name__)
current_config = {}
active_cap = None
cap_lock = threading.Lock()
# Lock for TTS engine to prevent concurrent access issues
tts_lock = threading.Lock()
# Last detection time for cooldown
last_alert_time = 0

# --- SSE Log Queue ---
# This queue will hold log messages to be sent to the dashboard via SSE
log_queue = Queue()


# --- Configuration Management Functions ---
CONFIG_FILE = 'stream_config.json'

def load_config():
    """Loads configuration from a JSON file, or returns default if not found/invalid."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Ensure all default keys are present in loaded config
                for key, default_val in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = default_val
                # Remove any old email keys if they exist in the loaded config
                config.pop('email_alerts_enabled', None)
                config.pop('recipient_email', None)
                config.pop('sender_email', None)
                config.pop('sender_password', None)
                config.pop('smtp_server', None)
                config.pop('smtp_port', None)
                return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load config file ({e}). Using default configuration.")
    return DEFAULT_CONFIG.copy()

def save_config(config):
    """Saves the current configuration to a JSON file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    except IOError as e:
        print(f"Error saving config file: {e}")

# Load initial configuration when the app starts
current_config = load_config()
print(f"Initial Loaded Config: {current_config}")


# --- Helper Functions ---
def get_class_id(model, class_name, captured_logs):
    """Safely gets class ID from model names, logging if not found."""
    if not hasattr(model, 'names') or not isinstance(model.names, dict):
        print(f"Warning: Model does not have a valid 'names' attribute. Cannot lookup class '{class_name}'.")
        return None

    for class_id, name in model.names.items():
        if name.strip().lower() == class_name.strip().lower():
            return class_id
    print(f"Warning: Class '{class_name}' not found in model names.")
    return None

def log_event_wrapper(msg, captured_logs=None):
    """Logs an event, appends it to a list, writes to file, prints to console, and sends to SSE queue."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_msg = f"[{ts}] {msg}"

    # For internal debugging/display (captured_logs argument is typically for generator scope)
    if captured_logs is not None:
        captured_logs.append(log_msg)

    # Write to global log file
    if LOG_FILE_HANDLE and not LOG_FILE_HANDLE.closed:
        try:
            LOG_FILE_HANDLE.write(log_msg + "\n")
            LOG_FILE_HANDLE.flush()
        except Exception as e:
            print(f"Error writing to log file from app.py context: {e}")
    
    # Print to console
    print(log_msg)

    # Send to SSE queue for real-time display on dashboard
    try:
        log_queue.put(log_msg)
    except Exception as e:
        print(f"Error putting message into log_queue: {e}")


def get_blank_frame(width=640, height=480, text="ERROR: NO FEED"):
    """Generates a blank black frame with an error message."""
    blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(blank_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return blank_frame

# --- Notification Functions ---
def play_voice_alert(text, voice_gender):
    """Converts text to speech and plays it directly on the device."""
    with tts_lock: # Acquire lock before using TTS engine
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            
            selected_voice_id = None
            for voice in voices:
                # Attempt to find a male or female voice based on preference and common indicators
                if voice_gender.lower() == 'male':
                    if 'male' in voice.name.lower() or 'david' in voice.name.lower() or voice.id.endswith('Microsoft SAPI5 English (United States) - David'): # Example for Windows David
                        selected_voice_id = voice.id
                        break
                elif voice_gender.lower() == 'female':
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower() or voice.id.endswith('Microsoft SAPI5 English (United States) - Zira'): # Example for Windows Zira
                        selected_voice_id = voice.id
                        break
            
            if selected_voice_id:
                engine.setProperty('voice', selected_voice_id)
            else:
                log_event_wrapper(f"Warning: No suitable '{voice_gender}' voice found. Using default voice.", None)

            engine.say(text)
            engine.runAndWait()
            log_event_wrapper(f"Voice alert played on device: '{text}'", None)
        except Exception as e:
            log_event_wrapper(f"Failed to play voice alert on device: {e}", None)


# --- MAIN STREAMING GENERATOR FUNCTION ---
def generate_frames_with_detection(camera_source, rotation_angle, ir_mode):
    global LOG_FILE_HANDLE
    global active_cap
    global last_alert_time # Access the global last_alert_time for cooldown
    captured_logs = []

    # Setup logging to file
    try:
        if not os.path.exists(LOGS_DIRECTORY):
            os.makedirs(LOGS_DIRECTORY)
            log_event_wrapper(f"Created log directory: {LOGS_DIRECTORY}", captured_logs)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(LOGS_DIRECTORY, f"{LOG_FILE_NAME_PREFIX}{timestamp}.txt")
        LOG_FILE_HANDLE = open(log_file_path, 'a', encoding='utf-8')
        
        LOG_FILE_HANDLE.write(f"--- Live Detection Log Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        LOG_FILE_HANDLE.write(f"Camera Source: {camera_source}\n")
        LOG_FILE_HANDLE.write(f"Rotation Angle: {rotation_angle}\n")
        LOG_FILE_HANDLE.write(f"IR Mode: {ir_mode}\n")
        LOG_FILE_HANDLE.write(f"Base Model: {os.path.basename(BASE_YOLO_MODEL_PATH)}\n")
        LOG_FILE_HANDLE.write(f"Fire Model: {os.path.basename(FIRE_YOLO_MODEL_PATH)}\n")
        LOG_FILE_HANDLE.write("-" * 60 + "\n")
        log_event_wrapper(f"Logging to: {log_file_path}", captured_logs)
    except Exception as e:
        log_event_wrapper(f"Failed to open log file: {e}. File logging disabled for this session.", captured_logs)
        LOG_FILE_HANDLE = None

    # Load YOLO models
    base_model = None
    fire_model = None
    try:
        if os.path.exists(BASE_YOLO_MODEL_PATH):
            base_model = YOLO(BASE_YOLO_MODEL_PATH)
            log_event_wrapper("✅ Base model loaded successfully.", captured_logs)
        else:
            log_event_wrapper(f"❌ Base model not found at {BASE_YOLO_MODEL_PATH}. Human/Vehicle detection will be skipped.", captured_logs)

        if os.path.exists(FIRE_YOLO_MODEL_PATH):
            fire_model = YOLO(FIRE_YOLO_MODEL_PATH)
            log_event_wrapper("✅ Fire model loaded successfully.", captured_logs)
        else:
            log_event_wrapper(f"❌ Fire model not found at {FIRE_YOLO_MODEL_PATH}. Fire detection will be skipped.", captured_logs)

        if base_model is None and fire_model is None:
            log_event_wrapper("❌ No YOLO models could be loaded. Detection will not function.", captured_logs)
            if LOG_FILE_HANDLE: LOG_FILE_HANDLE.close()
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', get_blank_frame(text=f"ERROR: NO MODELS LOADED"))[1].tobytes() + b'\r\n'
            return

    except Exception as e:
        log_event_wrapper(f"❌ Model load error: {e}. Please check model paths and integrity.", captured_logs)
        if LOG_FILE_HANDLE: LOG_FILE_HANDLE.close()
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', get_blank_frame(text=f"ERROR: MODEL LOAD FAILED - {e}"))[1].tobytes() + b'\r\n'
        return

    HUMAN_ID = get_class_id(base_model, HUMAN_CLASS_NAME, captured_logs) if base_model else None
    VEHICLE_IDS = [get_class_id(base_model, vn, captured_logs) for vn in VEHICLE_CLASS_NAMES if base_model and get_class_id(base_model, vn, captured_logs) is not None] if base_model else []
    FIRE_ID = get_class_id(fire_model, FIRE_CLASS_NAME, captured_logs) if fire_model else None

    if FIRE_ID is None and fire_model:
        log_event_wrapper(f"CRITICAL ERROR: Fire class '{FIRE_CLASS_NAME}' not found in fire model. Fire detection will not function.", captured_logs)

    # Acquire lock for camera access
    with cap_lock:
        if active_cap and active_cap.isOpened():
            log_event_wrapper("Releasing previous camera source.", captured_logs)
            active_cap.release()

        cap = cv2.VideoCapture(camera_source)
        active_cap = cap # Set the global active_cap

        if isinstance(camera_source, str) and (camera_source.startswith("rtsp://") or camera_source.startswith("http://")):
            log_event_wrapper(f"Attempting to open IP camera stream: {camera_source}", captured_logs)
            time.sleep(2)
            for i in range(3): # Retry mechanism for IP cameras
                if cap.isOpened():
                    log_event_wrapper(f"Successfully opened IP camera connection on attempt {i+1}.", captured_logs)
                    break
                log_event_wrapper(f"Retrying IP camera connection (attempt {i+1}/3)...", captured_logs)
                cap.release() # Release before retrying
                time.sleep(3)
                cap = cv2.VideoCapture(camera_source)
            if not cap.isOpened():
                log_event_wrapper(f"Failed to open IP camera source '{camera_source}' after multiple retries.", captured_logs)

        if not cap.isOpened():
            error_message = f"CRITICAL ERROR: Failed to open camera/video source '{camera_source}'. "
            if isinstance(camera_source, int):
                error_message += "Check if webcam is connected and not in use by another application."
            elif isinstance(camera_source, str) and os.path.exists(camera_source):
                error_message += "Check if video file path is correct and accessible."
            elif isinstance(camera_source, str) and (camera_source.startswith("rtsp://") or camera_source.startswith("http://")):
                error_message += "Check IP camera URL, credentials, network connectivity, or firewall settings."
            else:
                error_message += "Invalid camera source or access denied."
                
            log_event_wrapper(error_message, captured_logs)
            if LOG_FILE_HANDLE: LOG_FILE_HANDLE.close()
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', get_blank_frame(text=f"ERROR: CAMERA/VIDEO SOURCE FAILED"))[1].tobytes() + b'\r\n'
            return

    log_event_wrapper(f"Starting live detection stream for source: {camera_source} with rotation={rotation_angle} and IR_Mode={ir_mode}", captured_logs)

    while True:
        # Dynamically load current config to apply changes without restarting
        current_stream_config = load_config()
        current_rotation_angle = current_stream_config['rotation_angle']
        current_ir_mode = current_stream_config['ir_mode']
        voice_alerts_enabled = current_stream_config['voice_alerts_enabled']
        voice_gender = current_stream_config['voice_gender']
        detection_cooldown_seconds = current_stream_config['detection_cooldown_seconds']


        ret, frame = cap.read()
        if not ret:
            log_event_wrapper("Failed to grab frame or end of video stream. Stopping stream.", captured_logs)
            if isinstance(camera_source, str) and not (camera_source.startswith("rtsp://") or camera_source.startswith("http://")):
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', get_blank_frame(text="VIDEO ENDED / DISCONNECTED"))[1].tobytes() + b'\r\n'
            break

        if frame is None:
            log_event_wrapper("Frame is None. Skipping frame.", captured_logs)
            continue

        # Apply rotation
        if current_rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif current_rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif current_rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Apply IR mode
        if current_ir_mode:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)

        detection_messages = []

        # Run base model detection
        if base_model and (HUMAN_ID is not None or VEHICLE_IDS):
            base_classes_to_detect = []
            if HUMAN_ID is not None:
                base_classes_to_detect.append(HUMAN_ID)
            base_classes_to_detect.extend(VEHICLE_IDS)

            if base_classes_to_detect:
                try:
                    results_base = base_model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD_BASE, classes=base_classes_to_detect)
                    for r in results_base:
                        # Visualize results on the frame
                        frame = r.plot()
                        for box in r.boxes:
                            class_id = int(box.cls[0])
                            # conf = float(box.conf[0]) # No longer used in message
                            class_name = base_model.names[class_id]
                            if class_name.lower() == HUMAN_CLASS_NAME.lower():
                                msg = f"Person detected"
                                detection_messages.append(msg)
                            elif class_name.lower() in [c.lower() for c in VEHICLE_CLASS_NAMES]:
                                msg = f"Vehicle ({class_name}) detected"
                                detection_messages.append(msg)
                except Exception as e:
                    log_event_wrapper(f"Error during base model inference: {e}", captured_logs)

        # Run fire model detection
        if fire_model and FIRE_ID is not None:
            try:
                results_fire = fire_model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD_FIRE, classes=[FIRE_ID])
                for r in results_fire:
                    frame = r.plot()
                    if len(r.boxes) > 0: # If any fire detection occurred
                        # conf = float(r.boxes[0].conf[0]) # No longer used in message
                        msg = f"🔥🔥🔥 Fire detected 🔥🔥🔥"
                        detection_messages.append(msg)
            except Exception as e:
                log_event_wrapper(f"Error during fire model inference: {e}", captured_logs)

        # Handle alerts if detections occurred and cooldown allows
        if detection_messages:
            current_time = time.time()
            if current_time - last_alert_time > detection_cooldown_seconds:
                full_alert_message = f"ALERT: {', '.join(detection_messages)}"
                log_event_wrapper(full_alert_message, captured_logs) # Log the combined alert

                # Trigger voice alert on the device (in a separate thread)
                if voice_alerts_enabled:
                    voice_thread = threading.Thread(target=play_voice_alert, args=(full_alert_message, voice_gender))
                    voice_thread.daemon = True
                    voice_thread.start()
                
                last_alert_time = current_time # Reset cooldown timer

        # Encode the frame to JPEG bytes and yield it
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            log_event_wrapper("Failed to encode frame to JPEG. Skipping frame.", captured_logs)
            continue
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release resources when the streaming loop breaks
    with cap_lock:
        if active_cap and active_cap.isOpened():
            active_cap.release()
            active_cap = None

    if LOG_FILE_HANDLE:
        LOG_FILE_HANDLE.write("-" * 60 + "\n")
        LOG_FILE_HANDLE.write(f"--- Live Detection Stream Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        LOG_FILE_HANDLE.close()
    log_event_wrapper("✅ Live detection stream shut down.", captured_logs)


# --- NEW SSE Log Stream Route ---
@app.route('/log_stream')
def log_stream():
    def generate_logs():
        while True:
            # Get log messages from the queue. Wait for a short time.
            # If no message, the loop continues, keeping the connection alive.
            try:
                log_message = log_queue.get(timeout=1) # Get with a timeout to avoid blocking indefinitely
                yield f"data: {log_message}\n\n"
            except Exception:
                # No new log message, keep connection alive with a comment or simply continue
                # yield ": heartbeat\n\n" # Optional: send a heartbeat to prevent timeouts
                pass # Just continue loop if no new message, connection stays open
            time.sleep(0.1) # Small delay to prevent busy-waiting

    return Response(generate_logs(), mimetype='text/event-stream')


# --- Flask Routes ---
@app.route('/')
def index():
    global current_config
    current_config = load_config()
    now_ist = datetime.now()
    return render_template('index.html', config=current_config, now=now_ist)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames_with_detection(
        camera_source=current_config['camera_source'],
        rotation_angle=current_config['rotation_angle'],
        ir_mode=current_config['ir_mode']
    ), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/configure_stream', methods=['POST'])
def configure_stream():
    global current_config

    new_source = request.form.get('camera_source_input')
    if new_source:
        try:
            current_config['camera_source'] = int(new_source) if new_source.isdigit() else new_source
        except ValueError:
            print(f"Warning: Invalid camera source input '{new_source}'. Keeping current.")

    rotation_str = request.form.get('rotation_angle', '0')
    try:
        current_config['rotation_angle'] = int(rotation_str)
    except ValueError:
        print(f"Warning: Invalid rotation angle input '{rotation_str}'. Keeping current.")
    
    current_config['voice_alerts_enabled'] = 'voice_alerts_enabled' in request.form
    current_config['voice_gender'] = request.form.get('voice_gender', current_config['voice_gender'])
    current_config['detection_cooldown_seconds'] = int(request.form.get('detection_cooldown_seconds', current_config['detection_cooldown_seconds']))

    save_config(current_config)
    print(f"Configuration updated: {current_config}")
    return redirect(url_for('index'))

@app.route('/toggle_ir_mode', methods=['POST'])
def toggle_ir_mode():
    global current_config

    current_config['ir_mode'] = not current_config['ir_mode']
    save_config(current_config)
    print(f"IR Mode toggled to: {current_config['ir_mode']}")
    return jsonify({'success': True, 'new_ir_state': current_config['ir_mode']})


# --- Standalone Test / Initial Setup ---
if __name__ == "__main__":
    print("--- Running SurakshaNethra (Flask App) ---")
    print("Ensure yolov8_models and templates/index.html exist.")
    print("Access the web interface at http://127.0.0.1:5000")

    os.makedirs('yolov8_models', exist_ok=True)
    os.makedirs('detection_logs_live', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    # Create dummy model files if they don't exist
    for model_path in [BASE_YOLO_MODEL_PATH, FIRE_YOLO_MODEL_PATH]:
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'w') as f:
                f.write("DUMMY_YOLO_MODEL_FILE_PLACEHOLDER")
            print(f"Created dummy model file: {model_path}. Please replace with your actual YOLOv8 models.")

    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
        print(f"Created initial config file: {CONFIG_FILE}")

    app.run(host='0.0.0.0', debug=True)