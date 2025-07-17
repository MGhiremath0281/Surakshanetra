from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from flask_socketio import SocketIO, emit
import threading
import time
import os
import queue
from datetime import datetime
import json
import cv2
import numpy as np
import traceback # Import traceback for detailed error logging
from playsound import playsound # For playing alert sound

# --- Import your feature handlers ---
# Ensure these modules and their classes exist and are correctly implemented
import model_handlers.agnidrishti_live as agnidrishti_live_module
from model_handlers.agnidrishti_live import AgnidrishtiCamera
import model_handlers.simharekha_live as simharekha_live_module
from model_handlers.simharekha_live import ForbiddenZoneIDS, intrusion_log_queue, get_alert_active as get_fzids_alert_active, set_alert_active as set_fzids_alert_active
import model_handlers.vihangvetri_live as vihangvetri_live_module
from model_handlers.vihangvetri_live import VihangVetriCamera
import model_handlers.margadarshi_live as margadarshi_live_module
from model_handlers.margadarshi_live import MargadarshiCamera, DEFAULT_MARGADARSHI_CONFIG
# --- NEW: Import Shankasoochi module directly and its camera class ---
import model_handlers.shankasoochi_live as shankasoochi_live_module
from model_handlers.shankasoochi_live import ShankasoochiCamera # Import the refactored class

app = Flask(__name__)
# It's good practice to load the secret key from environment variables for production
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_default_secret_key_change_this_in_production_12345')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Dictionary to hold active feature instances
active_feature_instance = {
    "agnidrishti": None,
    "forbidden_zone": None,
    "vihangvetri": None,
    "margadarshi": None,
    "shankasoochi": None # NEW: Add Shankasoochi
}

# Locks to ensure thread-safe access to feature instances
feature_locks = {
    "agnidrishti": threading.Lock(),
    "forbidden_zone": threading.Lock(),
    "vihangvetri": threading.Lock(),
    "margadarshi": threading.Lock(),
    "shankasoochi": threading.Lock() # NEW: Add lock for Shankasoochi
}

CONFIG_FILE = 'stream_config.json'

# --- Default Configurations for Features ---
DEFAULT_AGNIDRISHTI_CONFIG = {
    'camera_source': 0, # Default to webcam 0
    'rotation_angle': 0,
    'ir_mode': False,
    'voice_alerts_enabled': True,
    'voice_gender': 'male',
    'detection_cooldown_seconds': 10
}

# Default sources for features whose configs aren't stored in stream_config.json here.
DEFAULT_FORBIDDEN_ZONE_SOURCE = "0"
DEFAULT_VIHANGVETRI_SOURCE = r"C:\Users\adity\Desktop\OBJDETECT\data\dronenew.mp4" # Default source for VihangVetri

# Margadarshi config uses DEFAULT_MARGADARSHI_CONFIG from its own module

# --- Shankasoochi Default Config ---
# Note: model_path, screenshot_folder, alert_sound_path will be set to absolute paths on startup
DEFAULT_SHANKASOOCHI_CONFIG = {
    'camera_source': 0,
    'model_path': 'yolov8x.pt', # Relative path within project, will be converted to absolute
    'screenshot_folder': 'screenshots', # Relative path within static/, will be converted to absolute
    'alert_sound_path': 'audio/alert.wav', # Relative path within static/, will be converted to absolute
    'tracker': 'bytetrack.yaml',
    'conf': 0.25, # Confidence threshold
    'iou': 0.7, # IoU threshold for NMS
    'agnostic_nms': False,
    'max_assoc_dist': 150, # Max association distance for tracker
    'holding_margin': 30, # Frames to hold track after losing detection
    'sticky_frames': 15, # Frames to stick to a detected anomaly before alerting
    'alert_frames': 10 # Number of consecutive frames an anomaly must be present to trigger an alert
}

def load_config():
    """Loads configurations from file, merging with defaults and handling legacy keys."""
    # Start with a full default config structure
    config = {
        'agnidrishti': DEFAULT_AGNIDRISHTI_CONFIG.copy(),
        'shankasoochi': DEFAULT_SHANKASOOCHI_CONFIG.copy(),
        # Add other feature configs here if they need persistent storage
    }
    loaded_config = {}
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)

        # Merge loaded config with defaults, preferring loaded values
        for feature_name, defaults in config.items():
            if feature_name in loaded_config:
                # Update existing feature config with loaded values
                defaults.update(loaded_config[feature_name])
                # Also, remove any keys in the loaded_config that are NOT in the current defaults
                # This handles cases where old config keys are deprecated.
                keys_to_remove = [k for k in loaded_config[feature_name] if k not in config[feature_name]]
                for k in keys_to_remove:
                    defaults.pop(k, None)
            config[feature_name] = defaults # Assign the merged config back

        # Special handling for Agnidrishti legacy keys removal (if they still exist in file)
        if 'agnidrishti' in config:
            for legacy_key in ['email_alerts_enabled', 'recipient_email', 'sender_email',
                               'sender_password', 'smtp_server', 'smtp_port']:
                config['agnidrishti'].pop(legacy_key, None)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load config file ({e}). Using default configurations.")
        # If there's an error, just return the default structure
        return {
            'agnidrishti': DEFAULT_AGNIDRISHTI_CONFIG.copy(),
            'shankasoochi': DEFAULT_SHANKASOOCHI_CONFIG.copy(),
        }
    return config

def save_config(config):
    """Saves the current application configurations to a JSON file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    except IOError as e:
        print(f"Error saving config file: {e}")

# Load all configurations at startup
current_app_config = load_config()
print(f"Initial App Config: {current_app_config}")

# Initialize non-JSON-managed config globals with their defaults
current_forbidden_zone_source = DEFAULT_FORBIDDEN_ZONE_SOURCE
current_vihangvetri_source = DEFAULT_VIHANGVETRI_SOURCE
if not os.path.exists(current_vihangvetri_source) and current_vihangvetri_source != "0": # "0" is for webcam, no file check needed
    print(f"WARNING: VihangVetri default video source not found: {current_vihangvetri_source}. Defaulting to webcam (0).")
    current_vihangvetri_source = "0"

current_margadarshi_config = DEFAULT_MARGADARSHI_CONFIG.copy() # Margadarshi config is managed within its module for now.

# Determine the absolute paths for Shankasoochi's assets
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths for Shankasoochi assets, ensuring they are placed in 'static' for web access
# The model path is typically relative to the model_handlers, so include that in the absolute path
SHANKASOOCHI_MODEL_ABSOLUTE_PATH = os.path.join(script_dir, 'model_handlers', current_app_config['shankasoochi']['model_path'])
SHANKASOOCHI_SCREENSHOT_ABSOLUTE_FOLDER = os.path.join(script_dir, 'static', current_app_config['shankasoochi']['screenshot_folder'])
SHANKASOOCHI_ALERT_SOUND_ABSOLUTE_PATH = os.path.join(script_dir, 'static', current_app_config['shankasoochi']['alert_sound_path'])

# Update Shankasoochi config with absolute paths (these are used by ShankasoochiCamera)
current_app_config['shankasoochi']['model_path'] = SHANKASOOCHI_MODEL_ABSOLUTE_PATH
current_app_config['shankasoochi']['screenshot_folder'] = SHANKASOOCHI_SCREENSHOT_ABSOLUTE_FOLDER
current_app_config['shankasoochi']['alert_sound_path'] = SHANKASOOCHI_ALERT_SOUND_ABSOLUTE_PATH

# Create a queue for Shankasoochi alerts to be sent to the frontend
shankasoochi_alert_queue = queue.Queue()

# Save the updated configuration (especially useful if paths changed or new defaults were added)
save_config(current_app_config)

def stop_all_active_features():
    """Stops all currently active feature instances gracefully."""
    print("Stopping all active feature instances...")
    for feature_name, instance in active_feature_instance.items():
        if instance is not None:
            with feature_locks[feature_name]:
                try:
                    instance.stop()
                    # A small delay to allow threads to shut down
                    time.sleep(0.5)
                    active_feature_instance[feature_name] = None
                    print(f"Stopped {feature_name}.")
                except Exception as e:
                    print(f"Error stopping {feature_name}: {e}")
                    traceback.print_exc() # Log the full traceback

def get_placeholder_frame():
    """Generates a placeholder image (either from file or dynamically) when no video feed is available."""
    placeholder_path = os.path.join(app.root_path, 'static', 'uploads', 'error_placeholder.jpg')
    try:
        with open(placeholder_path, 'rb') as f:
            return b'--frame\r\n' \
                   b'Content-Type: image/jpeg\r\n\r\n' + f.read() + b'\r\n'
    except FileNotFoundError:
        print(f"Error: Placeholder image not found at {placeholder_path}! Generating dynamic placeholder.")
        # Create a blank image with error text
        blank_image = np.zeros(shape=[480, 640, 3], dtype=np.uint8)
        text = "NO FEED / ERROR"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (blank_image.shape[1] - text_size[0]) // 2
        text_y = (blank_image.shape[0] + text_size[1]) // 2
        cv2.putText(blank_image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
        ret, buffer = cv2.imencode('.jpg', blank_image)
        return b'--frame\r\n' \
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'

# --- Video Feed Generators ---
# These functions continuously fetch frames from active camera instances
# and yield them as multipart JPEG responses.
def generate_frames_agnidrishti():
    while True:
        with feature_locks["agnidrishti"]:
            camera = active_feature_instance["agnidrishti"]
        if camera:
            frame_bytes = camera.get_frame()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                yield get_placeholder_frame()
            time.sleep(0.03) # ~30 FPS
        else:
            yield get_placeholder_frame()
            time.sleep(1) # Sleep longer if no camera active

def generate_frames_forbidden_zone():
    while True:
        with feature_locks["forbidden_zone"]:
            camera = active_feature_instance["forbidden_zone"]
        if camera:
            frame_bytes = camera.get_frame()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                yield get_placeholder_frame()
            time.sleep(0.03)
        else:
            yield get_placeholder_frame()
            time.sleep(1)

def generate_frames_vihangvetri():
    while True:
        with feature_locks["vihangvetri"]:
            camera = active_feature_instance["vihangvetri"]
        if camera:
            frame_bytes = camera.get_frame()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                yield get_placeholder_frame()
            time.sleep(0.03)
        else:
            yield get_placeholder_frame()
            time.sleep(1)

def generate_frames_margadarshi():
    while True:
        with feature_locks["margadarshi"]:
            camera = active_feature_instance["margadarshi"]
        if camera:
            frame_bytes = camera.get_frame()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                yield get_placeholder_frame()
            time.sleep(0.03)
        else:
            yield get_placeholder_frame()
            time.sleep(1)

# NEW: Generator for Shankasoochi frames
def generate_frames_shankasoochi():
    while True:
        with feature_locks["shankasoochi"]:
            camera = active_feature_instance["shankasoochi"]
        if camera:
            frame_bytes = camera.get_frame()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                yield get_placeholder_frame()
            time.sleep(0.03)
        else:
            yield get_placeholder_frame()
            time.sleep(1)

# --- Flask Routes ---
@app.route('/')
def index():
    # Pass current configurations and active status to the template for display
    agnidrishti_active = active_feature_instance["agnidrishti"] is not None
    forbidden_zone_active = active_feature_instance["forbidden_zone"] is not None
    vihangvetri_active = active_feature_instance["vihangvetri"] is not None
    margadarshi_active = active_feature_instance["margadarshi"] is not None
    shankasoochi_active = active_feature_instance["shankasoochi"] is not None # NEW

    # Create display-friendly config dictionaries (excluding paths/sensitive info)
    agnidrishti_config_display = {k: v for k, v in current_app_config['agnidrishti'].items()
                                  if k not in ['voice_gender', 'detection_cooldown_seconds']}
    shankasoochi_config_display = {k: v for k, v in current_app_config['shankasoochi'].items()
                                   if k not in ['model_path', 'screenshot_folder', 'alert_sound_path']} # NEW

    return render_template('index.html',
                           now=datetime.now(), # Pass current time for "Last Update"
                           agnidrishti_config=agnidrishti_config_display,
                           agnidrishti_active=agnidrishti_active,
                           forbidden_zone_source=current_forbidden_zone_source,
                           forbidden_zone_alert_active=get_fzids_alert_active(),
                           forbidden_zone_active=forbidden_zone_active,
                           vihangvetri_source=current_vihangvetri_source,
                           vihangvetri_active=vihangvetri_active,
                           margadarshi_config=current_margadarshi_config, # Passing full config for Margadarshi
                           margadarshi_active=margadarshi_active,
                           shankasoochi_config=shankasoochi_config_display, # NEW: Pass this to index.html
                           shankasoochi_active=shankasoochi_active # NEW
                          )

@app.route('/features')
def features():
    # Route for the Features page, passes current time for "Last Update"
    return render_template('features.html', now=datetime.now())

# Route for Agnidrishti Dashboard (assuming a template exists for it)
@app.route('/dashboard')
def agnidrishti_dashboard():
    return render_template('agnidrishti_dashboard.html', now=datetime.now())

# Route for Simharekha Dashboard (assuming a template exists for it)
@app.route('/forbidden_zone_ids')
def simharekha_dashboard():
    return render_template('simharekha_dashboard.html', now=datetime.now())

# Route for VihangVetri Dashboard (assuming a template exists for it)
@app.route('/vihangvetri_dashboard')
def vihangvetri_dashboard():
    return render_template('vihangvetri_dashboard.html', now=datetime.now())

# Route for Margadarshi Dashboard
@app.route('/margadarshi_dashboard')
def margadarshi_dashboard():
    return render_template('margadarshi_dashboard.html', now=datetime.now())

# NEW: Route for Shankasoochi Dashboard
@app.route('/shankasoochi_dashboard')
def shankasoochi_dashboard():
    # Pass current configurations and active status to the template for display
    shankasoochi_active = active_feature_instance["shankasoochi"] is not None
    shankasoochi_config_display = {k: v for k, v in current_app_config['shankasoochi'].items()
                                   if k not in ['model_path', 'screenshot_folder', 'alert_sound_path']}

    return render_template('shankasoochi_dashboard.html',
                           now=datetime.now(),
                           shankasoochi_config=shankasoochi_config_display, # <--- THIS IS THE FIX YOU NEEDED
                           shankasoochi_active=shankasoochi_active
                          )

# NEW: Endpoint to get Shankasoochi config for frontend form population
@app.route('/get_shankasoochi_config', methods=['GET'])
def get_shankasoochi_config():
    # Return a copy of the config, excluding absolute paths for security/simplicity
    config_for_frontend = {k: v for k, v in current_app_config['shankasoochi'].items()
                           if k not in ['model_path', 'screenshot_folder', 'alert_sound_path']}
    return jsonify(config_for_frontend)


# --- Video Feed Endpoints ---
@app.route('/video_feed_agnidrishti')
def video_feed_agnidrishti():
    return Response(generate_frames_agnidrishti(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_forbidden_zone')
def video_feed_forbidden_zone():
    return Response(generate_frames_forbidden_zone(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_vihangvetri')
def video_feed_vihangvetri():
    return Response(generate_frames_vihangvetri(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_margadarshi')
def video_feed_margadarshi():
    return Response(generate_frames_margadarshi(), mimetype='multipart/x-mixed-replace; boundary=frame')

# NEW: Shankasoochi video feed endpoint
@app.route('/video_feed_shankasoochi')
def video_feed_shankasoochi():
    return Response(generate_frames_shankasoochi(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Start/Stop Endpoints for all features ---
@app.route('/start_feature/<feature_name>', methods=['POST'])
def start_feature(feature_name):
    global current_forbidden_zone_source, current_vihangvetri_source, current_margadarshi_config

    # Stop all other features to ensure only one video stream is active at a time
    # This prevents resource contention and multiple AI models running simultaneously
    stop_all_active_features()

    with feature_locks[feature_name]:
        try:
            if feature_name == "agnidrishti":
                source = current_app_config['agnidrishti']['camera_source']
                active_feature_instance[feature_name] = AgnidrishtiCamera(
                    source=source,
                    rotation_angle=current_app_config['agnidrishti']['rotation_angle'],
                    ir_mode=current_app_config['agnidrishti']['ir_mode'],
                    voice_alerts_enabled=current_app_config['agnidrishti']['voice_alerts_enabled'],
                    voice_gender=current_app_config['agnidrishti']['voice_gender'],
                    detection_cooldown_seconds=current_app_config['agnidrishti']['detection_cooldown_seconds']
                )
            elif feature_name == "forbidden_zone":
                source = current_forbidden_zone_source
                active_feature_instance[feature_name] = ForbiddenZoneIDS(source=source)
            elif feature_name == "vihangvetri":
                source = current_vihangvetri_source
                active_feature_instance[feature_name] = VihangVetriCamera(source=source)
            elif feature_name == "margadarshi":
                source = current_margadarshi_config['camera_source'] # Use config from global
                active_feature_instance[feature_name] = MargadarshiCamera(source=source, config=current_margadarshi_config)
            # NEW: Shankasoochi start logic
            elif feature_name == "shankasoochi":
                # Use the complete, absolute paths from the loaded config
                source = current_app_config['shankasoochi']['camera_source']
                active_feature_instance[feature_name] = ShankasoochiCamera(
                    source=source,
                    config=current_app_config['shankasoochi'],
                    alert_queue=shankasoochi_alert_queue # Pass the queue for alerts
                )
            else:
                return jsonify({"status": "error", "message": "Unknown feature"}), 400

            active_feature_instance[feature_name].start()
            print(f"Started {feature_name} with source {source}")
            return jsonify({"status": "success", "message": f"{feature_name} started."})
        except Exception as e:
            traceback.print_exc() # Print full traceback to console for debugging
            print(f"Error starting {feature_name}: {e}")
            return jsonify({"status": "error", "message": f"Failed to start {feature_name}: {str(e)}"}), 500

@app.route('/stop_feature/<feature_name>', methods=['POST'])
def stop_feature(feature_name):
    with feature_locks[feature_name]:
        if active_feature_instance[feature_name]:
            try:
                active_feature_instance[feature_name].stop()
                active_feature_instance[feature_name] = None
                print(f"Stopped {feature_name}.")
                return jsonify({"status": "success", "message": f"{feature_name} stopped."})
            except Exception as e:
                traceback.print_exc()
                print(f"Error stopping {feature_name}: {e}")
                return jsonify({"status": "error", "message": f"Failed to stop {feature_name}: {str(e)}"}), 500
        return jsonify({"status": "info", "message": f"{feature_name} not running."})

# --- Configuration Endpoints ---
@app.route('/configure_agnidrishti', methods=['POST'])
def configure_agnidrishti():
    global current_app_config
    data = request.json
    try:
        # Update Agnidrishti configuration based on received data
        current_app_config['agnidrishti']['camera_source'] = int(data.get('camera_source', current_app_config['agnidrishti']['camera_source']))
        current_app_config['agnidrishti']['rotation_angle'] = int(data.get('rotation_angle', current_app_config['agnidrishti']['rotation_angle']))
        current_app_config['agnidrishti']['ir_mode'] = bool(data.get('ir_mode', current_app_config['agnidrishti']['ir_mode']))
        current_app_config['agnidrishti']['voice_alerts_enabled'] = bool(data.get('voice_alerts_enabled', current_app_config['agnidrishti']['voice_alerts_enabled']))
        current_app_config['agnidrishti']['voice_gender'] = data.get('voice_gender', current_app_config['agnidrishti']['voice_gender'])
        current_app_config['agnidrishti']['detection_cooldown_seconds'] = int(data.get('detection_cooldown_seconds', current_app_config['agnidrishti']['detection_cooldown_seconds']))

        save_config(current_app_config)
        print(f"Agnidrishti Config Updated: {current_app_config['agnidrishti']}")
        # If Agnidrishti is active, restart it with the new configuration
        if active_feature_instance["agnidrishti"]:
            stop_feature("agnidrishti")
            # Call start_feature which will re-instantiate with the updated current_app_config
            return start_feature("agnidrishti")
        return jsonify({"status": "success", "message": "Agnidrishti configuration updated."})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Failed to update Agnidrishti configuration: {str(e)}"}), 400

@app.route('/configure_forbidden_zone', methods=['POST'])
def configure_forbidden_zone():
    global current_forbidden_zone_source
    data = request.json
    try:
        new_source = data.get('camera_source')
        if new_source is not None:
            current_forbidden_zone_source = new_source
            print(f"Forbidden Zone Source Updated: {current_forbidden_zone_source}")
            # If Forbidden Zone is active, restart it with the new source
            if active_feature_instance["forbidden_zone"]:
                stop_feature("forbidden_zone")
                return start_feature("forbidden_zone")
            return jsonify({"status": "success", "message": "Forbidden Zone configuration updated."})
        return jsonify({"status": "error", "message": "No camera source provided."}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Failed to update Forbidden Zone configuration: {str(e)}"}), 400

@app.route('/toggle_forbidden_zone_alert', methods=['POST'])
def toggle_forbidden_zone_alert():
    data = request.json
    new_state = data.get('active')
    if new_state is not None:
        set_fzids_alert_active(new_state)
        print(f"Forbidden Zone Alert Active: {get_fzids_alert_active()}")
        return jsonify({"status": "success", "active": get_fzids_alert_active()})
    return jsonify({"status": "error", "message": "Invalid state"}), 400

@app.route('/get_forbidden_zone_logs')
def get_forbidden_zone_logs():
    logs = []
    # Drain the queue, but don't block if empty
    while not intrusion_log_queue.empty():
        try:
            logs.append(intrusion_log_queue.get_nowait())
        except queue.Empty:
            break
    return jsonify({"logs": logs})

@app.route('/configure_vihangvetri', methods=['POST'])
def configure_vihangvetri():
    global current_vihangvetri_source
    data = request.json
    try:
        new_source = data.get('camera_source')
        if new_source is not None:
            current_vihangvetri_source = new_source
            print(f"VihangVetri Source Updated: {current_vihangvetri_source}")
            # If VihangVetri is active, restart it with the new source
            if active_feature_instance["vihangvetri"]:
                stop_feature("vihangvetri")
                return start_feature("vihangvetri")
            return jsonify({"status": "success", "message": "VihangVetri configuration updated."})
        return jsonify({"status": "error", "message": "No camera source provided."}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Failed to update VihangVetri configuration: {str(e)}"}), 400

@app.route('/configure_margadarshi', methods=['POST'])
def configure_margadarshi():
    global current_margadarshi_config
    data = request.json
    try:
        # Update Margadarshi config parameters from request data
        # Ensure correct type casting for numeric and boolean values
        current_margadarshi_config['camera_source'] = data.get('camera_source', current_margadarshi_config['camera_source'])
        current_margadarshi_config['object_classes'] = data.get('object_classes', current_margadarshi_config['object_classes'])
        current_margadarshi_config['confidence_threshold'] = float(data.get('confidence_threshold', current_margadarshi_config['confidence_threshold']))
        # Line coordinates might come as strings from form, ensure they are converted to list of lists of ints
        if 'line_coords' in data and data['line_coords'] is not None:
            current_margadarshi_config['line_coords'] = [list(map(int, point)) for point in data['line_coords']]
        current_margadarshi_config['alert_cooldown'] = int(data.get('alert_cooldown', current_margadarshi_config['alert_cooldown']))
        current_margadarshi_config['direction'] = data.get('direction', current_margadarshi_config['direction'])

        print(f"Margadarshi Config Updated: {current_margadarshi_config}")
        # If Margadarshi is active, restart it with new config
        if active_feature_instance["margadarshi"]:
            stop_feature("margadarshi")
            return start_feature("margadarshi")
        return jsonify({"status": "success", "message": "Margadarshi configuration updated."})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Failed to update Margadarshi configuration: {str(e)}"}), 400

# NEW: Shankasoochi configuration endpoint
@app.route('/configure_shankasoochi', methods=['POST'])
def configure_shankasoochi():
    global current_app_config
    data = request.json
    try:
        # Update Shankasoochi configuration based on received data
        current_app_config['shankasoochi']['camera_source'] = int(data.get('camera_source', current_app_config['shankasoochi']['camera_source']))
        current_app_config['shankasoochi']['tracker'] = data.get('tracker', current_app_config['shankasoochi']['tracker'])
        current_app_config['shankasoochi']['conf'] = float(data.get('conf', current_app_config['shankasoochi']['conf']))
        current_app_config['shankasoochi']['iou'] = float(data.get('iou', current_app_config['shankasoochi']['iou']))
        current_app_config['shankasoochi']['agnostic_nms'] = bool(data.get('agnostic_nms', current_app_config['shankasoochi']['agnostic_nms']))
        current_app_config['shankasoochi']['max_assoc_dist'] = int(data.get('max_assoc_dist', current_app_config['shankasoochi']['max_assoc_dist']))
        current_app_config['shankasoochi']['holding_margin'] = int(data.get('holding_margin', current_app_config['shankasoochi']['holding_margin']))
        current_app_config['shankasoochi']['sticky_frames'] = int(data.get('sticky_frames', current_app_config['shankasoochi']['sticky_frames']))
        current_app_config['shankasoochi']['alert_frames'] = int(data.get('alert_frames', current_app_config['shankasoochi']['alert_frames']))

        # model_path, screenshot_folder, and alert_sound_path are managed on app startup
        # and typically not updated via this config endpoint.
        save_config(current_app_config)
        print(f"Shankasoochi Config Updated: {current_app_config['shankasoochi']}")
        # If Shankasoochi is active, restart it with new config
        if active_feature_instance["shankasoochi"]:
            stop_feature("shankasoochi")
            # Call start_feature which will re-instantiate with the updated current_app_config
            return start_feature("shankasoochi")
        return jsonify({"status": "success", "message": "Shankasoochi configuration updated."})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Failed to update Shankasoochi configuration: {str(e)}"}), 400


# --- Shutdown Handler ---
@app.route('/shutdown', methods=['POST'])
def shutdown():
    print("Shutting down server...")
    stop_all_active_features()
    # Use a threading.Event or similar for more graceful shutdown in complex apps.
    # For a simple development server, os._exit(0) can force an exit.
    # In production, rely on the WSGI server's shutdown mechanism.
    threading.Thread(target=lambda: os._exit(0)).start() # Force exit after a small delay
    return 'Server shutting down...'


# --- SocketIO for alerts ---
@socketio.on('connect')
def test_connect():
    print('Client connected to SocketIO')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected from SocketIO')

def send_intrusion_logs():
    """Background task to send intrusion logs via SocketIO."""
    while True:
        if not intrusion_log_queue.empty():
            log_entry = intrusion_log_queue.get()
            socketio.emit('intrusion_log', log_entry)
        socketio.sleep(0.5) # Check for logs every 0.5 seconds

def send_shankasoochi_alerts():
    """NEW: Background task to send Shankasoochi anomaly alerts via SocketIO."""
    while True:
        if not shankasoochi_alert_queue.empty():
            alert_data = shankasoochi_alert_queue.get()
            print(f"Emitting Shankasoochi alert: {alert_data.get('description')}")
            socketio.emit('anomaly_alert', alert_data)
            # Play sound on the server side when an alert is emitted
            try:
                # Correctly refer to the alert sound path from the current_app_config
                sound_path_from_config = current_app_config['shankasoochi']['alert_sound_path']
                if os.path.exists(sound_path_from_config):
                    playsound(sound_path_from_config, block=False)
                    print(f"Played alert sound: {sound_path_from_config}")
                else:
                    print(f"Alert sound file not found for playback: {sound_path_from_config}")
            except Exception as e:
                print(f"Error playing alert sound: {e}")
                traceback.print_exc()
        socketio.sleep(0.1) # Check for alerts more frequently

# Start background threads to send logs/alerts
socketio.start_background_task(send_intrusion_logs)
socketio.start_background_task(send_shankasoochi_alerts) # NEW background task

if __name__ == '__main__':
    # --- Ensure necessary directories exist on startup ---
    # Create the Shankasoochi screenshots folder if it doesn't exist
    if not os.path.exists(SHANKASOOCHI_SCREENSHOT_ABSOLUTE_FOLDER):
        os.makedirs(SHANKASOOCHI_SCREENSHOT_ABSOLUTE_FOLDER, exist_ok=True)
        print(f"Created Shankasoochi screenshot folder: {SHANKASOOCHI_SCREENSHOT_ABSOLUTE_FOLDER}")

    # Create static/audio folder if it doesn't exist
    audio_dir = os.path.dirname(SHANKASOOCHI_ALERT_SOUND_ABSOLUTE_PATH)
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir, exist_ok=True)
        print(f"Created audio folder: {audio_dir}")

    # Ensure placeholder image exists
    uploads_dir = os.path.join(app.root_path, 'static', 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir, exist_ok=True)
        print(f"Created uploads folder for placeholder: {uploads_dir}")

    placeholder_path = os.path.join(uploads_dir, 'error_placeholder.jpg')
    if not os.path.exists(placeholder_path):
        print(f"Placeholder image '{placeholder_path}' not found. Creating a blank one.")
        blank_image = np.zeros(shape=[480, 640, 3], dtype=np.uint8)
        text = "NO FEED / ERROR"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (blank_image.shape[1] - text_size[0]) // 2
        text_y = (blank_image.shape[0] + text_size[1]) // 2
        cv2.putText(blank_image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
        cv2.imwrite(placeholder_path, blank_image)

    # Ensure default alert.wav exists for Shankasoochi
    # This prevents errors if playsound is called for a non-existent file.
    # You should place your actual alert.wav file here: `static/audio/alert.wav`
    # The default path is 'audio/alert.wav' inside 'static'
    alert_wav_path_in_static_audio = os.path.join(script_dir, 'static', 'audio', 'alert.wav')
    if not os.path.exists(alert_wav_path_in_static_audio):
        print(f"Alert sound file '{alert_wav_path_in_static_audio}' not found. "
              "Please provide a valid .wav file for alerts. "
              "A silent dummy file can be created if no sound is desired, "
              "otherwise, audio alerts will not function for Shankasoochi.")
        # Optional: Create a dummy silent WAV file if you want to ensure the path exists
        # from scipy.io.wavfile import write
        # write(alert_wav_path_in_static_audio, 44100, np.zeros(44100, dtype=np.int16)) # 1 second of silence

    print(f"Flask app starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)

    # Ensure all features are stopped on application shutdown
    stop_all_active_features()
    print("Flask app terminated.")