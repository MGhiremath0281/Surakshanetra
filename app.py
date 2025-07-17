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

# --- Import your feature handlers ---
import model_handlers.agnidrishti_live as agnidrishti_live_module # Changed to module import
from model_handlers.agnidrishti_live import AgnidrishtiCamera # Keep for direct class access
from model_handlers.simharekha_live import ForbiddenZoneIDS, intrusion_log_queue, get_alert_active as get_fzids_alert_active, set_alert_active as set_fzids_alert_active
import model_handlers.vihangvetri_live as vihangvetri_live_module # Changed to module import
from model_handlers.vihangvetri_live import VihangVetriCamera # Keep for direct class access
# --- NEW: Import Margadarshi module directly ---
import model_handlers.margadarshi_live as margadarshi_live_module
from model_handlers.margadarshi_live import MargadarshiCamera, DEFAULT_MARGADARSHI_CONFIG # Keep these for direct class/config access

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_default_secret_key_change_this_in_production')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

active_feature_instance = {
    "agnidrishti": None,
    "forbidden_zone": None,
    "vihangvetri": None,
    "margadarshi": None # NEW: Add Margadarshi
}

feature_locks = {
    "agnidrishti": threading.Lock(),
    "forbidden_zone": threading.Lock(),
    "vihangvetri": threading.Lock(),
    "margadarshi": threading.Lock() # NEW: Add lock for Margadarshi
}

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
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                for key, default_val in DEFAULT_AGNIDRISHTI_CONFIG.items():
                    if key not in config:
                        config[key] = default_val
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
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    except IOError as e:
        print(f"Error saving config file: {e}")

current_agnidrishti_config = load_agnidrishti_config()
print(f"Initial Agnidrishti Config: {current_agnidrishti_config}")

current_forbidden_zone_source = "0"

current_vihangvetri_source = r"C:\Users\adity\Desktop\OBJDETECT\data\dronenew.mp4" 
if not os.path.exists(current_vihangvetri_source):
    print(f"WARNING: VihangVetri default video source not found: {current_vihangvetri_source}. Defaulting to webcam (0).")
    current_vihangvetri_source = "0"

# NEW: Margadarshi configuration and source
current_margadarshi_config = DEFAULT_MARGADARSHI_CONFIG.copy()
# You can load/save this config from a file if you want it persistent across restarts
# For now, it will reset to defaults each time app.py starts.
print(f"Initial Margadarshi Config: {current_margadarshi_config}")


def stop_all_active_features():
    print("Stopping all active feature instances...")
    for feature_name, instance in active_feature_instance.items():
        if instance is not None:
            with feature_locks[feature_name]:
                try:
                    instance.stop()
                    time.sleep(0.5) 
                    active_feature_instance[feature_name] = None
                    print(f"Stopped {feature_name}.")
                except Exception as e:
                    print(f"Error stopping {feature_name}: {e}")

def get_placeholder_frame():
    placeholder_path = os.path.join(app.root_path, 'static', 'uploads', 'error_placeholder.jpg')
    try:
        with open(placeholder_path, 'rb') as f:
            return b'--frame\r\n' \
                   b'Content-Type: image/jpeg\r\n\r\n' + f.read() + b'\r\n'
    except FileNotFoundError:
        print(f"Error: Placeholder image not found at {placeholder_path}! Please ensure it exists.")
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
               b'Content-Type: image/jpeg\r\r\n' + buffer.tobytes() + b'\r\n'


@app.route('/')
def index():
    stop_all_active_features()
    return render_template('index.html')

@app.route('/features')
def features():
    stop_all_active_features()
    return render_template('features.html')

@app.route('/dashboard')
def dashboard():
    stop_all_active_features()
    with feature_locks["agnidrishti"]:
        if active_feature_instance["agnidrishti"] is None:
            active_feature_instance["agnidrishti"] = AgnidrishtiCamera(config=current_agnidrishti_config)
            print("AgnidrishtiCamera (Dashboard) started.")
    return render_template('dashboard.html', config=current_agnidrishti_config, now=datetime.now())

@app.route('/forbidden_zone_ids')
def forbidden_zone_ids():
    stop_all_active_features()
    global current_forbidden_zone_source
    with feature_locks["forbidden_zone"]:
        if active_feature_instance["forbidden_zone"] is None:
            active_feature_instance["forbidden_zone"] = ForbiddenZoneIDS(video_source=current_forbidden_zone_source)
            print("ForbiddenZoneIDS started.")
    return render_template('forbidden_zone.html', current_source=current_forbidden_zone_source)

@app.route('/vihangvetri_dashboard')
def vihangvetri_dashboard():
    stop_all_active_features()
    global current_vihangvetri_source
    with feature_locks["vihangvetri"]:
        if active_feature_instance["vihangvetri"] is None:
            active_feature_instance["vihangvetri"] = VihangVetriCamera(video_source=current_vihangvetri_source)
            print("VihangVetriCamera started.")
    initial_zone = active_feature_instance["vihangvetri"].zone if active_feature_instance["vihangvetri"] else {'x': 0, 'y': 0, 'width': 0, 'height': 0}
    return render_template('vihangvetri_dashboard.html', 
                           current_source=current_vihangvetri_source,
                           initial_zone_x=initial_zone['x'],
                           initial_zone_y=initial_zone['y'],
                           initial_zone_width=initial_zone['width'],
                           initial_zone_height=initial_zone['height'])

# NEW: Margadarshi Dashboard Route
@app.route('/margadarshi_dashboard')
def margadarshi_dashboard():
    stop_all_active_features()
    global current_margadarshi_config
    with feature_locks["margadarshi"]:
        if active_feature_instance["margadarshi"] is None:
            # Pass the current config to the MargadarshiCamera instance
            active_feature_instance["margadarshi"] = MargadarshiCamera(config=current_margadarshi_config)
            print("MargadarshiCamera started.")
    
    # Get initial zone from the instance for frontend display
    initial_zone_pixels = active_feature_instance["margadarshi"].get_current_zone_pixels() if active_feature_instance["margadarshi"] else {'x': 0, 'y': 0, 'width': 0, 'height': 0}

    return render_template('margadarshi_dashboard.html', 
                           config=current_margadarshi_config,
                           initial_zone_x=initial_zone_pixels['x'],
                           initial_zone_y=initial_zone_pixels['y'],
                           initial_zone_width=initial_zone_pixels['width'],
                           initial_zone_height=initial_zone_pixels['height'],
                           now=datetime.now())


@app.route('/video_feed/<feature_name>')
def video_feed(feature_name):
    if feature_name not in active_feature_instance:
        print(f"Video feed requested for unknown feature: {feature_name}")
        return Response(get_placeholder_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    # Add a small wait for Margadarshi to initialize if it's not ready
    if feature_name == "margadarshi":
        timeout = 5 # seconds
        start_time = time.time()
        while active_feature_instance[feature_name] is None and (time.time() - start_time) < timeout:
            time.sleep(0.1) # Wait for 100ms
        
        if active_feature_instance[feature_name] is None:
            print(f"Timeout: Margadarshi instance not ready after {timeout} seconds. Serving placeholder.")
            return Response(get_placeholder_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

    # If the instance is still None after waiting (or for other features), serve placeholder
    if active_feature_instance[feature_name] is None:
        print(f"Video feed requested for inactive feature: {feature_name}. Serving placeholder.")
        return Response(get_placeholder_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    if feature_name == "agnidrishti":
        return Response(active_feature_instance[feature_name].gen_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif feature_name == "forbidden_zone":
        return Response(active_feature_instance[feature_name].generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif feature_name == "vihangvetri":
        return Response(active_feature_instance[feature_name].generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    # NEW: Margadarshi video feed
    elif feature_name == "margadarshi":
        return Response(active_feature_instance[feature_name].generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(get_placeholder_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


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
    
    with feature_locks["agnidrishti"]:
        if active_feature_instance["agnidrishti"]:
            active_feature_instance["agnidrishti"].update_config(current_agnidrishti_config)

    return jsonify({'success': True, 'new_ir_state': current_agnidrishti_config['ir_mode']})


@app.route('/set_forbidden_zone_source', methods=['POST'])
def set_forbidden_zone_source():
    global current_forbidden_zone_source
    data = request.get_json()
    new_source = data.get('source')
    
    if new_source is None:
        return jsonify({"status": "error", "message": "No source provided"}), 400

    print(f"Attempting to change Forbidden Zone IDS source to: {new_source}")
    current_forbidden_zone_source = new_source
    
    stop_all_active_features()
    
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


@app.route('/set_vihangvetri_source', methods=['POST'])
def set_vihangvetri_source():
    global current_vihangvetri_source
    data = request.get_json()
    new_source = data.get('source')
    
    if new_source is None:
        return jsonify({"status": "error", "message": "No source provided"}), 400

    print(f"Attempting to change VihangVetri source to: {new_source}")
    current_vihangvetri_source = new_source
    
    stop_all_active_features()
    
    return jsonify({"status": "success", "message": f"VihangVetri source set to {new_source}. Please refresh the page or navigate to the VihangVetri feature."})

@app.route('/set_vihangvetri_zone', methods=['POST'])
def set_vihangvetri_zone():
    data = request.get_json()
    try:
        x = int(data.get('x'))
        y = int(data.get('y'))
        width = int(data.get('width'))
        height = int(data.get('height'))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Invalid zone coordinates or dimensions"}), 400

    with feature_locks["vihangvetri"]:
        if active_feature_instance["vihangvetri"]:
            active_feature_instance["vihangvetri"].set_zone(x, y, width, height)
            return jsonify({"status": "success", "message": "VihangVetri zone updated"})
        return jsonify({"status": "error", "message": "VihangVetri not running"}), 400

@app.route('/reset_vihangvetri_zone', methods=['POST'])
def reset_vihangvetri_zone():
    with feature_locks["vihangvetri"]:
        if active_feature_instance["vihangvetri"]:
            active_feature_instance["vihangvetri"].reset_zone()
            return jsonify({"status": "success", "message": "VihangVetri zone reset to default"})
        return jsonify({"status": "error", "message": "VihangVetri not running"}), 400

# NEW: Margadarshi Configuration Routes
@app.route('/set_margadarshi_source', methods=['POST'])
def set_margadarshi_source():
    global current_margadarshi_config
    data = request.get_json()
    new_source = data.get('source')
    
    if new_source is None:
        return jsonify({"status": "error", "message": "No source provided"}), 400

    print(f"Attempting to change Margadarshi source to: {new_source}")
    current_margadarshi_config['IP_STREAM_URL'] = new_source
    
    stop_all_active_features() # Restart to apply new source
    
    return jsonify({"status": "success", "message": f"Margadarshi source set to {new_source}. Please refresh the page or navigate to the Margadarshi feature."})

@app.route('/update_margadarshi_config', methods=['POST'])
def update_margadarshi_config():
    global current_margadarshi_config
    
    # Ensure request body is JSON and not empty
    data = request.get_json(silent=True) # silent=True returns None if parsing fails
    if not data or not isinstance(data, dict):
        print(f"ERROR: Invalid or empty JSON received for Margadarshi config update. Data: {data}")
        return jsonify({"status": "error", "message": "Invalid or empty JSON data received"}), 400

    updated_values = {}
    
    # Handle numerical inputs
    for key in ["CONFIDENCE_THRESHOLD", "NMS_IOU_THRESHOLD", "LOG_INTERVAL_NOT_CLEAR_SEC", "MAX_PIXEL_DISTANCE_FOR_TRACK", "TRACK_EXPIRY_FRAMES"]:
        if key in data and data[key] is not None:
            try:
                updated_values[key] = float(data[key]) if "THRESHOLD" in key else int(data[key])
            except (TypeError, ValueError):
                print(f"Warning: Invalid value for {key}: {data[key]}")
                # Optionally, return an error to the frontend, but continue processing other valid fields
    
    # Handle boolean toggles
    for key in ["IR_MODE", "VOICE_ALERTS_ENABLED", "VISUAL_ALERT_ENABLED", "AUTO_SCREENSHOT_ENABLED", "ALERT_SOUND_ENABLED"]:
        if key in data:
            updated_values[key] = bool(data[key])

    # Handle rotation (integer 0-3)
    if "ROTATION_STATE" in data and data["ROTATION_STATE"] is not None:
        try:
            updated_values["ROTATION_STATE"] = int(data["ROTATION_STATE"]) % 4
        except (TypeError, ValueError):
            print(f"Warning: Invalid value for ROTATION_STATE: {data['ROTATION_STATE']}")

    # Update global config
    current_margadarshi_config.update(updated_values)

    # Update live instance if running
    with feature_locks["margadarshi"]:
        if active_feature_instance["margadarshi"]:
            active_feature_instance["margadarshi"].update_config(updated_values)
            return jsonify({"status": "success", "message": "Margadarshi configuration updated", "config": current_margadarshi_config})
        return jsonify({"status": "error", "message": "Margadarshi not running"}), 400

@app.route('/set_margadarshi_zone_proportional', methods=['POST'])
def set_margadarshi_zone_proportional():
    data = request.get_json()
    try:
        tl_x = float(data.get('tl_x_prop'))
        tl_y = float(data.get('tl_y_prop'))
        br_x = float(data.get('br_x_prop'))
        br_y = float(data.get('br_y_prop'))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Invalid zone proportional coordinates"}), 400

    with feature_locks["margadarshi"]:
        if active_feature_instance["margadarshi"]:
            active_feature_instance["margadarshi"].set_zone_proportional(tl_x, tl_y, br_x, br_y)
            return jsonify({"status": "success", "message": "Margadarshi zone updated proportionally"})
        return jsonify({"status": "error", "message": "Margadarshi not running"}), 400

@app.route('/reset_margadarshi_zone', methods=['POST'])
def reset_margadarshi_zone():
    with feature_locks["margadarshi"]:
        if active_feature_instance["margadarshi"]:
            active_feature_instance["margadarshi"].reset_zone_to_default()
            return jsonify({"status": "success", "message": "Margadarshi zone reset to default"})
        return jsonify({"status": "error", "message": "Margadarshi not running"}), 400


@socketio.on('connect')
def handle_connect():
    print('Client connected to SocketIO')
    pass


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected from SocketIO')

# Modified log_producer to accept arguments for queue and alert status getters
def log_producer(agnidrishti_q_getter, fzids_q_getter, fzids_alert_getter, 
                 vihangvetri_q_getter, vihangvetri_alert_getter,
                 margadarshi_q_getter, margadarshi_alert_getter):
    while True:
        try:
            agnidrishti_q = agnidrishti_q_getter()
            while not agnidrishti_q.empty():
                log_entry = agnidrishti_q.get()
                socketio.emit('agnidrishti_log', {'log': log_entry})
            
            fzids_q = fzids_q_getter()
            while not fzids_q.empty():
                log_entry = fzids_q.get()
                socketio.emit('forbidden_zone_log', {'log': log_entry})
            
            socketio.emit('forbidden_zone_alert_status', {'active': fzids_alert_getter()})

            vihangvetri_q = vihangvetri_q_getter()
            while not vihangvetri_q.empty():
                log_entry = vihangvetri_q.get()
                socketio.emit('vihangvetri_log', {'log': log_entry})
            
            socketio.emit('vihangvetri_alert_status', {'active': vihangvetri_alert_getter()})

            # NEW: Margadarshi logs and alert status
            margadarshi_q = margadarshi_q_getter()
            while not margadarshi_q.empty():
                log_entry = margadarshi_q.get()
                socketio.emit('margadarshi_log', {'log': log_entry})
            
            socketio.emit('margadarshi_alert_status', {'active': margadarshi_alert_getter()})


            time.sleep(0.1)
        except Exception as e:
            print(f"Error in log_producer: {e}")
            time.sleep(1)

if __name__ == '__main__':
    os.makedirs('yolov8_models', exist_ok=True)
    os.makedirs('detection_logs_live', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('model_handlers', exist_ok=True)
    os.makedirs('static/runway_alerts', exist_ok=True) # NEW: Create directory for Margadarshi screenshots

    for model_path in [
        os.path.join('yolov8_models', 'yolov8x.pt'),
        os.path.join('yolov8_models', 'yolofirenew.pt'),
        os.path.join('yolov8_models', 'best.pt'),
        os.path.join('yolov8_models', 'yolov8n.pt') # NEW: Ensure yolov8n.pt exists or is a placeholder
    ]:
        if not os.path.exists(model_path):
            with open(model_path, 'w') as f:
                f.write("DUMMY_YOLO_MODEL_FILE_PLACEHOLDER")
            print(f"Created dummy model file: {model_path}. Please replace with your actual YOLO models.")

    if not os.path.exists(CONFIG_FILE):
        save_agnidrishti_config(DEFAULT_AGNIDRISHTI_CONFIG)
        print(f"Created initial config file: {CONFIG_FILE}")

    placeholder_path = os.path.join(app.root_path, 'static', 'uploads', 'error_placeholder.jpg')
    if not os.path.exists(placeholder_path):
        try:
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

    # Pass the getter functions to the log_producer thread
    log_thread = threading.Thread(target=log_producer, daemon=True, args=(
        agnidrishti_live_module.get_agnidrishti_log_queue, # Access via module
        lambda: intrusion_log_queue, # Changed to a lambda to make it a getter function
        get_fzids_alert_active,
        vihangvetri_live_module.get_vihangvetri_log_queue, # Access via module
        vihangvetri_live_module.get_vihangvetri_alert_active, # Access via module
        margadarshi_live_module.get_margadarshi_log_queue, # Access via module
        margadarshi_live_module.get_margadarshi_alert_active # Access via module
    ))
    log_thread.start()

    # Explicitly list the files to be watched by the reloader
    watched_files = [
        os.path.abspath(__file__), # app.py
        os.path.abspath(os.path.join('model_handlers', 'agnidrishti_live.py')),
        os.path.abspath(os.path.join('model_handlers', 'simharekha_live.py')),
        os.path.abspath(os.path.join('model_handlers', 'vihangvetri_live.py')),
        os.path.abspath(os.path.join('model_handlers', 'margadarshi_live.py')), # NEW: Add Margadarshi
        os.path.abspath(os.path.join('templates', 'index.html')),
        os.path.abspath(os.path.join('templates', 'features.html')),
        os.path.abspath(os.path.join('templates', 'dashboard.html')),
        os.path.abspath(os.path.join('templates', 'forbidden_zone.html')),
        os.path.abspath(os.path.join('templates', 'vihangvetri_dashboard.html')),
        os.path.abspath(os.path.join('templates', 'margadarshi_dashboard.html')), # NEW: Add Margadarshi template
        os.path.abspath(CONFIG_FILE)
    ]
    
    watched_files = [f for f in watched_files if os.path.exists(f)]

    print("Starting Flask application...")
    try:
        socketio.run(app, debug=True, allow_unsafe_werkzeug=True, extra_files=watched_files)
    except Exception as e:
        print(f"FATAL ERROR during Flask application startup: {e}")
        traceback.print_exc() # Print full traceback for more details

