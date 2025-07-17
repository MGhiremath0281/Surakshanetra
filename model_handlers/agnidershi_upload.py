import cv2
import numpy as np
from ultralytics import YOLO # Keep this import if you want to use YOLO for testing in __main__
import time
from datetime import datetime
import os
import threading
from playsound import playsound
import io
import sys

# --- Configuration for this specific file's processing ---
# Set the path to your input video file directly here.
INPUT_VIDEO_PATH = r'C:\Users\adity\Desktop\SurakshaNethra\data\firedata2.mp4' # <--- SET YOUR DESIRED FIXED VIDEO PATH HERE

# Alert Sound Paths (relative to the project root, assuming sounds are in 'static' folder)
FIRE_SOUND_PATH = os.path.join('static', 'alert.mp3')
HUMAN_SOUND_PATH = os.path.join('static', 'alert.mp3')
VEHICLE_SOUND_PATH = os.path.join('static', 'alert.mp3')

CONFIDENCE_THRESHOLD_BASE = 0.5
CONFIDENCE_THRESHOLD_FIRE = 0.3

ALERT_COOLDOWN_SECONDS = 3

# --- Helper for capturing print output and logging ---
class Capturing(list):
    """Context manager to capture stdout."""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # ensure re-enable of stdout
        sys.stdout = self._stdout

def log_event_wrapper(message, captured_logs):
    """Logs an event and appends it to a list for web display."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_msg = f"[{now}] {message}"
    captured_logs.append(log_msg)
    # If you also want to see these logs in the server's console, uncomment the next line:
    # print(log_msg)

def play_sound_async_wrapper(path, captured_logs):
    """Plays a sound file asynchronously on the server."""
    if os.path.exists(path):
        try:
            threading.Thread(target=playsound, args=(path,), kwargs={'block': False}).start()
        except Exception as e:
            log_event_wrapper(f"Error playing sound {path}: {e}", captured_logs)
    else:
        log_event_wrapper(f"Warning: Sound file not found at {path}", captured_logs)

def trigger_sound_alert_wrapper(alert_type, current_alert_state, captured_logs):
    """
    Triggers a specific sound alert on the server, respecting cooldown.
    current_alert_state is a dictionary { 'last_alert_time': float, 'last_alert_type': str }
    """
    path = None
    if alert_type == "FIRE":
        path = FIRE_SOUND_PATH
    elif alert_type == "HUMAN":
        path = HUMAN_SOUND_PATH
    elif alert_type == "VEHICLE":
        path = VEHICLE_SOUND_PATH
    elif alert_type == "HUMAN_VEHICLE":
        path = HUMAN_SOUND_PATH # Using human sound for combined
    
    if path:
        current_time = time.time()
        if alert_type != current_alert_state['last_alert_type'] or \
           (current_time - current_alert_state['last_alert_time']) >= ALERT_COOLDOWN_SECONDS:
            
            play_sound_async_wrapper(path, captured_logs)
            log_event_wrapper(f"Sound alert triggered: {alert_type}", captured_logs)
            current_alert_state['last_alert_type'] = alert_type
            current_alert_state['last_alert_time'] = current_time

# --- Main processing function (renamed for clarity) ---
# Added base_model and fire_model as arguments
def process_fixed_video_with_detection(output_path, rotation_angle=0, ir_mode=False, base_model=None, fire_model=None):
    """
    Processes the hardcoded INPUT_VIDEO_PATH, applies detection models,
    and saves an annotated output video.
    
    Args:
        output_path (str): Path where the processed output video will be saved.
        rotation_angle (int): Angle for video rotation (0, 90, 180, 270).
        ir_mode (bool): If True, applies an IR-like colormap to the video frames.
        base_model (YOLO): Pre-loaded YOLO model for human/vehicle detection.
        fire_model (YOLO): Pre-loaded YOLO model for fire detection.
        
    Returns:
        tuple: (success (bool), output_file_path (str or None), list_of_logs (list of str))
    """
    captured_logs = []
    
    # Check if models were passed (they should be from app.py global loading)
    if base_model is None or fire_model is None:
        log_event_wrapper("❌ Error: YOLO models were not provided to process_fixed_video_with_detection. Ensure they are pre-loaded in app.py.", captured_logs)
        return False, None, captured_logs

    # Check if the hardcoded input video path exists
    if not os.path.exists(INPUT_VIDEO_PATH):
        log_event_wrapper(f"❌ Error: Hardcoded input video file not found at {INPUT_VIDEO_PATH}. Please check the path.", captured_logs)
        return False, None, captured_logs

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        log_event_wrapper(f"❌ Could not open video: {INPUT_VIDEO_PATH}. Check file integrity or permissions.", captured_logs)
        return False, None, captured_logs

    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Using mp4v for broader compatibility
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Adjust width/height if rotation is applied
    if rotation_angle in [90, 270]:
        w, h = h, w

    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not out.isOpened():
        log_event_wrapper(f"❌ Could not create output video: {output_path}. Check codec or path permissions.", captured_logs)
        cap.release()
        return False, None, captured_logs

    # ID mappings (ensure these class names match your model's actual names)
    HUMAN_CLASS_NAME = 'person'
    VEHICLE_CLASS_NAMES = ['car', 'truck', 'bus', 'motorcycle'] # Add other vehicle types if needed
    FIRE_CLASS_NAME = 'fire' # Ensure this matches your fire model's class name

    # Dynamic class ID lookup for robustness
    # Check if base_model.names exists before using it
    HUMAN_CLASS_ID = next((k for k, v in base_model.names.items() if v.lower() == HUMAN_CLASS_NAME.lower()), None) if hasattr(base_model, 'names') else None
    VEHICLE_CLASS_IDS = [k for k, v in base_model.names.items() if v.lower() in VEHICLE_CLASS_NAMES] if hasattr(base_model, 'names') else []
    FIRE_CLASS_ID = next((k for k, v in fire_model.names.items() if v.lower() == FIRE_CLASS_NAME.lower()), None) if hasattr(fire_model, 'names') else None

    if HUMAN_CLASS_ID is None: log_event_wrapper(f"Warning: Class '{HUMAN_CLASS_NAME}' not found in base model names. Human detection might not work.", captured_logs)
    if not VEHICLE_CLASS_IDS: log_event_wrapper(f"Warning: No specified vehicle classes found in base model names from {VEHICLE_CLASS_NAMES}. Vehicle detection might not work.", captured_logs)
    if FIRE_CLASS_ID is None: log_event_wrapper(f"Warning: Class '{FIRE_CLASS_NAME}' not found in fire model names. Fire detection might not work.", captured_logs)

    # Alert state for this specific processing run
    current_alert_state = {'last_alert_time': 0, 'last_alert_type': ""}

    while True:
        ret, frame = cap.read()
        if not ret:
            log_event_wrapper("End of video stream or failed to read frame.", captured_logs)
            break

        # Apply rotation based on the provided argument
        if rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Apply IR mode based on the provided argument
        if ir_mode:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)

        detections = []

        # Run base model detection only if relevant class IDs were found
        base_classes_to_detect = []
        if HUMAN_CLASS_ID is not None:
            base_classes_to_detect.append(HUMAN_CLASS_ID)
        base_classes_to_detect.extend(VEHICLE_CLASS_IDS)

        if base_classes_to_detect:
            # Pass classes as a list of integers
            base_results = base_model(frame, conf=CONFIDENCE_THRESHOLD_BASE, classes=base_classes_to_detect, verbose=False)
            for r in base_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    detections.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'cls': cls, 'model': 'base'})

        # Run fire model detection only if fire class ID was found
        if FIRE_CLASS_ID is not None:
            # Pass classes as a single integer for fire model if only one class
            fire_results = fire_model(frame, conf=CONFIDENCE_THRESHOLD_FIRE, classes=[FIRE_CLASS_ID], verbose=False)
            for r in fire_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    detections.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'cls': cls, 'model': 'fire'})

        fire_detected = human_detected = vehicle_detected = False

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf, cls_id, model = det['conf'], det['cls'], det['model']
            label = ""
            color = (0, 255, 0) # Default green for safety

            if model == 'fire' and cls_id == FIRE_CLASS_ID:
                label = f'FIRE {conf:.2f}'
                color = (0, 0, 255) # Red for fire
                fire_detected = True
            elif model == 'base':
                if cls_id == HUMAN_CLASS_ID:
                    label = f'Person {conf:.2f}'
                    color = (255, 0, 0) # Blue for human
                    human_detected = True
                elif cls_id in VEHICLE_CLASS_IDS:
                    label = f'{base_model.names.get(cls_id, "Vehicle").capitalize()} {conf:.2f}'
                    color = (0, 255, 255) # Yellow for vehicle
                    vehicle_detected = True
                else:
                    continue # Skip if class ID not in our target lists

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Determine the highest priority alert type for sound and logging
        alert_type = None
        if fire_detected:
            alert_type = "FIRE"
        elif human_detected and vehicle_detected:
            alert_type = "HUMAN_VEHICLE"
        elif human_detected:
            alert_type = "HUMAN"
        elif vehicle_detected:
            alert_type = "VEHICLE"

        # Trigger sound alert if conditions are met (cooldown and type change)
        trigger_sound_alert_wrapper(alert_type, current_alert_state, captured_logs)

        out.write(frame) # Write the processed frame to the output video

    cap.release()
    out.release()
    log_event_wrapper(f"✅ Video processing complete. Output saved to: {output_path}", captured_logs)
    return True, output_path, captured_logs

# This `if __name__ == "__main__":` block is for local testing of this script
# and will NOT be executed when imported by Flask. It demonstrates how the
# function can be called and its output captured.
if __name__ == "__main__":
    print("--- Running agnidershi_upload.py in standalone test mode ---")
    
    # IMPORTANT: For local testing, ensure these paths point to valid files relative to this script's execution.
    # Adjust as needed if you're running this from a different directory.
    
    # Create dummy directories if they don't exist for testing
    os.makedirs('temp_test_processed', exist_ok=True)
    os.makedirs('static', exist_ok=True) # Ensure sound path exists for testing
    os.makedirs('yolov8_models', exist_ok=True) # Ensure model path exists for testing
    os.makedirs('pre_defined_videos', exist_ok=True) # Ensure video input path exists for testing

    # Create dummy sound files if they don't exist for testing playsound
    dummy_sound_content = b'RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88\x13\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00' # Minimal WAV header
    for sound_file in [FIRE_SOUND_PATH, HUMAN_SOUND_PATH, VEHICLE_SOUND_PATH]:
        if not os.path.exists(sound_file):
            try:
                with open(sound_file, 'wb') as f:
                    f.write(dummy_sound_content)
                print(f"Created dummy sound file: {sound_file}")
            except Exception as e:
                print(f"Could not create dummy sound file {sound_file}: {e}")

    # For standalone testing, you'd need to mock the models or ensure they exist
    # For a quick test, you might skip model loading or use dummy models if they don't exist.
    # In a real scenario, you'd make sure your .pt files are present.
    test_base_model = None
    test_fire_model = None
    try:
        # These paths must be correct relative to where you run this script if testing standalone
        base_model_path = os.path.join('yolov8_models', 'yolov8x.pt')
        fire_model_path = os.path.join('yolov8_models', 'yolofirenew.pt')

        if os.path.exists(base_model_path):
            test_base_model = YOLO(base_model_path)
            print("Test base model loaded for standalone run.")
        else:
            print(f"Warning: Base model not found at {base_model_path}. Detection will be skipped for human/vehicle.")

        if os.path.exists(fire_model_path):
            test_fire_model = YOLO(fire_model_path)
            print("Test fire model loaded for standalone run.")
        else:
            print(f"Warning: Fire model not found at {fire_model_path}. Detection will be skipped for fire.")

    except Exception as e:
        print(f"Could not load test models for standalone run: {e}. Skipping detection.")

    # Generate a unique output filename for standalone test
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"processed_standalone_{timestamp}.mp4"
    output_video_path = os.path.join('temp_test_processed', output_filename)

    # Check if the hardcoded INPUT_VIDEO_PATH exists for the standalone test
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"Error: INPUT_VIDEO_PATH for standalone test not found: {INPUT_VIDEO_PATH}. Please provide a video file for testing.")
        # Create a dummy video file for testing if it doesn't exist
        print("Creating a dummy video file for testing...")
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            dummy_out = cv2.VideoWriter(INPUT_VIDEO_PATH, fourcc, 10, (640, 480))
            if dummy_out.isOpened():
                for _ in range(30): # Write 30 frames (3 seconds at 10fps)
                    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8) + np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
                    cv2.putText(dummy_frame, "Dummy Video Frame", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    dummy_out.write(dummy_frame)
                dummy_out.release()
                print(f"Dummy video created at {INPUT_VIDEO_PATH}")
            else:
                print(f"Failed to create dummy video at {INPUT_VIDEO_PATH}. Please check permissions.")
        except Exception as e:
            print(f"Error creating dummy video: {e}")

    if os.path.exists(INPUT_VIDEO_PATH) and test_base_model and test_fire_model:
        print(f"Starting video processing for: {INPUT_VIDEO_PATH}")
        
        # Call the main processing function
        success, result_path, function_logs = process_fixed_video_with_detection(
            output_video_path,
            rotation_angle=0, # Change for testing rotation
            ir_mode=False,    # Change for testing IR mode
            base_model=test_base_model, # Pass the loaded models
            fire_model=test_fire_model  # Pass the loaded models
        )

        print("\n--- Standalone Processing Logs ---")
        for log_line in function_logs:
            print(log_line)
        print("----------------------------------")

        if success:
            print(f"\n✅ Processing successful. Output video: {result_path}")
        else:
            print(f"\n❌ Processing failed. Check logs for details.")
    else:
        print(f"\nSkipping video processing test. Ensure '{INPUT_VIDEO_PATH}', 'yolov8x.pt', and 'yolofirenew.pt' are available.")
    
    print("--- agnidershi_upload.py standalone test finished ---")