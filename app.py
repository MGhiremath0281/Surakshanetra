from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from flask_socketio import SocketIO, emit
import threading
import time
import os
import queue
from datetime import datetime
import json # For loading/saving config
import cv2 # For placeholder image generation
import numpy as np # For placeholder image generation

# --- Import your feature handlers ---
# Make sure the import paths are correct based on your 'model_handlers' folder
from model_handlers.agnidrishti_live import AgnidrishtiCamera, get_agnidrishti_log_queue
from model_handlers.simharekha_live import ForbiddenZoneIDS, intrusion_log_queue, get_alert_active as get_fzids_alert_active, set_alert_active as set_fzids_alert_active

app = Flask(__name__)
# Configure a secret key for Flask sessions and SocketIO
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_default_secret_key_change_this_in_production')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- Global instances for managing cameras/features ---
active_feature_instance = {
    "agnidrishti": None,
    "forbidden_zone": None
}

# --- Threading locks for camera management ---
feature_locks = {
    "agnidrishti": threading.Lock(),
    "forbidden_zone": threading.Lock()
}

# --- Configuration Management for Agnidrishti (from original app.py) ---
CONFIG_FILE = 'stream_config.json'
DEFAULT_AGNIDRISHTI_CONFIG = {
    'camera_source': 0,
    'rotation_angle': 0,
    'ir_mode': False,
    'voice_alerts_enabled': True,
    'voice_gender': 'male',
    'detection_cooldown_seconds': 10
}

def load_agnidrishti_config():
    """Loads configuration for Agnidrishti from a JSON file, or returns default if not found/invalid."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Ensure all default keys are present in loaded config
                for key, default_val in DEFAULT_AGNIDRISHTI_CONFIG.items():
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
        print(f"Warning: Could not load config file ({e}). Using default configuration for Agnidrishti.")
    return DEFAULT_AGNIDRISHTI_CONFIG.copy()

def save_agnidrishti_config(config):
    """Saves the current configuration for Agnidrishti to a JSON file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    except IOError as e:
        print(f"Error saving config file: {e}")

# Load initial configuration for Agnidrishti when the app starts
current_agnidrishti_config = load_agnidrishti_config()
print(f"Initial Agnidrishti Config: {current_agnidrishti_config}")

# --- Source Configuration for Forbidden Zone IDS (can be made persistent later) ---
current_forbidden_zone_source = "0" # Default webcam for Forbidden Zone IDS

# --- Utility Functions ---
def stop_all_active_features():
    """Stops all currently running feature instances."""
    print("Stopping all active feature instances...")
    for feature_name, instance in active_feature_instance.items():
        if instance is not None:
            with feature_locks[feature_name]:
                try:
                    instance.stop()
                    # Give it a moment to stop its internal loop
                    time.sleep(0.5) 
                    active_feature_instance[feature_name] = None
                    print(f"Stopped {feature_name}.")
                except Exception as e:
                    print(f"Error stopping {feature_name}: {e}")

def get_placeholder_frame():
    """Returns a placeholder image when no camera is active or an error occurs."""
    # Ensure this path is correct relative to app.py
    placeholder_path = os.path.join(app.root_path, 'static', 'uploads', 'error_placeholder.jpg')
    try:
        with open(placeholder_path, 'rb') as f:
            return b'--frame\r\n' \
                   b'Content-Type: image/jpeg\r\n\r\n' + f.read() + b'\r\n'
    except FileNotFoundError:
        print(f"Error: Placeholder image not found at {placeholder_path}! Please ensure it exists.")
        # Fallback to a hardcoded black image if placeholder is missing
        blank_image = np.zeros(shape=[480, 640, 3], dtype=np.uint8)
        text = "NO IMAGE"
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


# --- Flask Routes ---

@app.route('/')
def index():
    """Landing page."""
    stop_all_active_features() # Ensure nothing is running when on landing page
    return render_template('index.html')

@app.route('/features')
def features():
    """Features selection page."""
    stop_all_active_features() # Ensure nothing is running when on features page
    return render_template('features.html')

@app.route('/dashboard')
def dashboard():
    """Your original AI Monitoring Dashboard."""
    # Ensure other features are stopped and Agnidrishti is initialized/running
    stop_all_active_features()
    with feature_locks["agnidrishti"]:
        if active_feature_instance["agnidrishti"] is None:
            # Pass the current config to the AgnidrishtiCamera
            active_feature_instance["agnidrishti"] = AgnidrishtiCamera(config=current_agnidrishti_config)
            print("AgnidrishtiCamera (Dashboard) started.")
    return render_template('dashboard.html', config=current_agnidrishti_config, now=datetime.now())

@app.route('/forbidden_zone_ids')
def forbidden_zone_ids():
    """New Forbidden Zone IDS feature page."""
    # Ensure other features are stopped and ForbiddenZoneIDS is initialized/running
    stop_all_active_features()
    global current_forbidden_zone_source # Access global variable
    with feature_locks["forbidden_zone"]:
        if active_feature_instance["forbidden_zone"] is None:
            active_feature_instance["forbidden_zone"] = ForbiddenZoneIDS(video_source=current_forbidden_zone_source)
            print("ForbiddenZoneIDS started.")
    return render_template('forbidden_zone.html', current_source=current_forbidden_zone_source)

# --- Video Streaming Endpoints ---

@app.route('/video_feed/<feature_name>')
def video_feed(feature_name):
    """
    Video streaming route for either feature.
    Expected feature_name: 'agnidrishti' or 'forbidden_zone'
    """
    if feature_name not in active_feature_instance or active_feature_instance[feature_name] is None:
        print(f"Video feed requested for inactive or unknown feature: {feature_name}")
        return Response(get_placeholder_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    # Use the appropriate generator for the feature
    if feature_name == "agnidrishti":
        return Response(active_feature_instance[feature_name].gen_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif feature_name == "forbidden_zone":
        return Response(active_feature_instance[feature_name].generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(get_placeholder_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


# --- Agnidrishti Specific Routes (from original app.py) ---
@app.route('/configure_stream', methods=['POST'])
def configure_stream():
    global current_agnidrishti_config

    new_source = request.form.get('camera_source_input')
    if new_source:
        try:
            current_agnidrishti_config['camera_source'] = int(new_source) if new_source.isdigit() else new_source
        except ValueError:
            print(f"Warning: Invalid camera source input '{new_source}'. Keeping current.")

    rotation_str = request.form.get('rotation_angle', '0')
    try:
        current_agnidrishti_config['rotation_angle'] = int(rotation_str)
    except ValueError:
        print(f"Warning: Invalid rotation angle input '{rotation_str}'. Keeping current.")
    
    current_agnidrishti_config['voice_alerts_enabled'] = 'voice_alerts_enabled' in request.form
    current_agnidrishti_config['voice_gender'] = request.form.get('voice_gender', current_agnidrishti_config['voice_gender'])
    current_agnidrishti_config['detection_cooldown_seconds'] = int(request.form.get('detection_cooldown_seconds', current_agnidrishti_config['detection_cooldown_seconds']))

    save_agnidrishti_config(current_agnidrishti_config)
    print(f"Agnidrishti Configuration updated: {current_agnidrishti_config}")
    
    # Update the running instance if it exists
    with feature_locks["agnidrishti"]:
        if active_feature_instance["agnidrishti"]:
            active_feature_instance["agnidrishti"].update_config(current_agnidrishti_config)

    return redirect(url_for('dashboard'))

@app.route('/toggle_ir_mode', methods=['POST'])
def toggle_ir_mode():
    global current_agnidrishti_config

    current_agnidrishti_config['ir_mode'] = not current_agnidrishti_config['ir_mode']
    save_agnidrishti_config(current_agnidrishti_config)
    print(f"Agnidrishti IR Mode toggled to: {current_agnidrishti_config['ir_mode']}")
    
    # Update the running instance if it exists
    with feature_locks["agnidrishti"]:
        if active_feature_instance["agnidrishti"]:
            active_feature_instance["agnidrishti"].update_config(current_agnidrishti_config)

    return jsonify({'success': True, 'new_ir_state': current_agnidrishti_config['ir_mode']})


# --- Forbidden Zone IDS Control Endpoints ---
@app.route('/set_forbidden_zone_source', methods=['POST'])
def set_forbidden_zone_source():
    global current_forbidden_zone_source
    data = request.get_json()
    new_source = data.get('source')
    
    if new_source is None:
        return jsonify({"status": "error", "message": "No source provided"}), 400

    print(f"Attempting to change Forbidden Zone IDS source to: {new_source}")
    current_forbidden_zone_source = new_source
    
    # Stop existing instance and restart with new source
    stop_all_active_features() # This will stop FZIDS if it's running
    
    # Re-initialize the FZIDS instance with the new source if it's the active view
    # This assumes the user will be redirected back to the FZIDS page
    # A more robust solution might involve re-initializing on the fly in the video_feed generator
    # but for simplicity, we'll let the next page load handle it.
    
    return jsonify({"status": "success", "message": f"Forbidden Zone IDS source set to {new_source}. Please refresh the page or navigate to the Forbidden Zone IDS feature."})


@app.route('/fzids_toggle_zone/<zone_name>', methods=['POST'])
def fzids_toggle_zone(zone_name):
    with feature_locks["forbidden_zone"]:
        if active_feature_instance["forbidden_zone"]:
            is_active = active_feature_instance["forbidden_zone"].toggle_zone_active(zone_name)
            if is_active is not None:
                return jsonify({"status": "success", "zone": zone_name, "active": is_active})
            return jsonify({"status": "error", "message": f"Zone {zone_name} not found"}), 404
        return jsonify({"status": "error", "message": "Forbidden Zone IDS not running"}), 400

@app.route('/fzids_scale_zone/<zone_name>/<action>', methods=['POST'])
def fzids_scale_zone(zone_name, action):
    increase = (action == 'increase')
    with feature_locks["forbidden_zone"]:
        if active_feature_instance["forbidden_zone"]:
            scale_factor = active_feature_instance["forbidden_zone"].scale_zone(zone_name, increase)
            if scale_factor is not None:
                return jsonify({"status": "success", "zone": zone_name, "scale_factor": scale_factor})
            return jsonify({"status": "error", "message": f"Zone {zone_name} not found"}), 404
        return jsonify({"status": "error", "message": "Forbidden Zone IDS not running"}), 400

@app.route('/fzids_reset_zones', methods=['POST'])
def fzids_reset_zones():
    with feature_locks["forbidden_zone"]:
        if active_feature_instance["forbidden_zone"]:
            active_feature_instance["forbidden_zone"].reset_zones()
            return jsonify({"status": "success", "message": "Zones reset to default"})
        return jsonify({"status": "error", "message": "Forbidden Zone IDS not running"}), 400


# --- SocketIO for Real-time Logs and Alerts ---
@socketio.on('connect')
def handle_connect():
    print('Client connected to SocketIO')
    # Optionally send initial state or welcome message
    # These initial messages are now handled by the JS on DOMContentLoaded
    pass


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected from SocketIO')

def log_producer():
    """Continuously fetches logs from queues and emits them via SocketIO."""
    while True:
        try:
            # Agnidrishti Logs
            agnidrishti_q = get_agnidrishti_log_queue()
            while not agnidrishti_q.empty():
                log_entry = agnidrishti_q.get()
                socketio.emit('agnidrishti_log', {'log': log_entry})
            
            # Forbidden Zone IDS Logs
            while not intrusion_log_queue.empty():
                log_entry = intrusion_log_queue.get()
                socketio.emit('forbidden_zone_log', {'log': log_entry})
            
            # Forbidden Zone IDS Alert Status (for pulsing indicator)
            socketio.emit('forbidden_zone_alert_status', {'active': get_fzids_alert_active()})

            time.sleep(0.1) # Poll more frequently for smoother log updates
        except Exception as e:
            print(f"Error in log_producer: {e}")
            time.sleep(1) # Wait longer on error

# Start the log producer thread
log_thread = threading.Thread(target=log_producer, daemon=True)
log_thread.start()

# --- Application Startup ---
if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('yolov8_models', exist_ok=True)
    os.makedirs('detection_logs_live', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True) # For FZIDS snapshots
    os.makedirs('model_handlers', exist_ok=True) # Ensure model_handlers exists

    # Create dummy model files if they don't exist
    for model_path in [
        os.path.join('yolov8_models', 'yolov8x.pt'),
        os.path.join('yolov8_models', 'yolofirenew.pt')
    ]:
        if not os.path.exists(model_path):
            with open(model_path, 'w') as f:
                f.write("DUMMY_YOLO_MODEL_FILE_PLACEHOLDER")
            print(f"Created dummy model file: {model_path}. Please replace with your actual YOLOv8 models.")

    # Create initial config file for Agnidrishti if it doesn't exist
    if not os.path.exists(CONFIG_FILE):
        save_agnidrishti_config(DEFAULT_AGNIDRISHTI_CONFIG)
        print(f"Created initial config file: {CONFIG_FILE}")

    # Create the placeholder image if it doesn't exist
    placeholder_path = os.path.join(app.root_path, 'static', 'uploads', 'error_placeholder.jpg')
    if not os.path.exists(placeholder_path):
        try:
            # Create a simple black image with text
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
            print(f"Created placeholder image: {placeholder_path}")
        except Exception as e:
            print(f"Could not create placeholder image: {e}")

    # Run Flask-SocketIO app
    print("Starting Flask application...")
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)