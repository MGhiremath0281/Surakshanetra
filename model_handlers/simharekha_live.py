import cv2
import numpy as np
from ultralytics import YOLO
import time
import pyttsx3 # For text-to-speech
import os
from datetime import datetime
import json
import threading
import queue # For thread-safe logging

# --- Global Configuration ---
# Adjust paths as per your project structure relative to app.py
YOLO_MODEL_PATH = "yolov8_models/yolov8x.pt"
SNAPSHOTS_DIR = "static/uploads" # Where to save intrusion snapshots
LOG_DIR = "detection_logs_live" # Where to save intrusion logs

# Ensure directories exist
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Thread-safe queue for logs to be consumed by Flask
intrusion_log_queue = queue.Queue()
# Thread-safe flag for active alerts (to communicate to frontend)
alert_active_flag = False
alert_flag_lock = threading.Lock()
_tts_lock_fzids = threading.Lock() # Dedicated TTS lock for FZIDS

def set_alert_active(state: bool):
    global alert_active_flag
    with alert_flag_lock:
        alert_active_flag = state

def get_alert_active() -> bool:
    with alert_flag_lock:
        return alert_active_flag

class ForbiddenZoneIDS:
    def __init__(self, video_source: str = "0"): # Changed to string for flexibility
        """
        Initialize the Forbidden Zone Intrusion Detection System
        Args:
            video_source: Video source (0 for webcam, "http://ip_cam_url" for IP cam, or path to video file)
        """
        self.video_source_init = video_source # Store original source for re-opening
        print(f"Initializing Forbidden Zone IDS with source: {video_source}")
        # Initialize YOLO model
        print(f"Loading YOLOv8x model from {YOLO_MODEL_PATH}...")
        try:
            # Ensure the model path is correct relative to where app.py is run
            self.model = YOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", YOLO_MODEL_PATH))
            print("YOLOv8x model loaded.")
        except Exception as e:
            print(f"Error loading YOLO model from {YOLO_MODEL_PATH}: {e}")
            self.model = None # Handle case where model fails to load

        # Video capture
        try:
            # Convert source to int if it's a digit string for webcam, else use as is
            if isinstance(video_source, str) and video_source.isdigit():
                self.cap = cv2.VideoCapture(int(video_source))
            else:
                self.cap = cv2.VideoCapture(video_source)
            
            if not self.cap.isOpened():
                raise IOError(f"Cannot open video source: {video_source}")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Video source opened successfully. Resolution: {self.frame_width}x{self.frame_height}")

        except Exception as e:
            print(f"Error opening video source {video_source}: {e}")
            self.cap = None # Indicate that video capture failed

        # Define target classes for detection (including drones)
        self.target_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',
            14: 'bird', 15: 'cat', 16: 'dog',
        }
        
        # Initialize forbidden zones (polygons) - lighter colors
        self.zones = {
            'zone1': {
                'points': np.array([[200, 200], [500, 200], [500, 400], [200, 400]], dtype=np.int32),
                'color': (0, 100, 255),  # Light Red/Orange
                'scale_factor': 1.0, 'center': None, 'active': True
            },
            'zone2': {
                'points': np.array([[700, 300], [1000, 300], [1000, 500], [700, 500]], dtype=np.int32),
                'color': (255, 100, 0),  # Light Blue
                'scale_factor': 1.0, 'center': None, 'active': True
            }
        }
        self._update_zone_centers() # Update zone centers

        # Alert system
        self.alert_active = False
        self.alert_start_time = 0
        self.alert_duration = 3.0  # seconds (visual alert overlay)
        self.last_audio_alert_time = 0
        self.audio_alert_cooldown = 3.0  # seconds (for voice alerts)
        
        # Logging
        self.log_file = os.path.join(LOG_DIR, f"intrusion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.intrusion_logs = []
        
        # Control flags
        self.running = True
        
        print("Forbidden Zone IDS initialized successfully!")
    
    def _update_zone_centers(self):
        """Update the center points of all zones"""
        for zone_name, zone in self.zones.items():
            points = zone['points']
            center_x = int(np.mean(points[:, 0]))
            center_y = int(np.mean(points[:, 1]))
            zone['center'] = (center_x, center_y)
    
    def _scale_zone(self, zone_name: str, scale_factor: float):
        """Scale a zone around its center point"""
        zone = self.zones[zone_name]
        center = zone['center']
        points = zone['points']
        
        scaled_points = []
        for point in points:
            translated = point - center
            scaled = translated * scale_factor
            final_point = scaled + center
            scaled_points.append(final_point)
        
        zone['points'] = np.array(scaled_points, dtype=np.int32)
        zone['scale_factor'] *= scale_factor
        
        # Ensure points stay within frame bounds
        zone['points'][:, 0] = np.clip(zone['points'][:, 0], 0, self.frame_width - 1)
        zone['points'][:, 1] = np.clip(zone['points'][:, 1], 0, self.frame_height - 1)
    
    def _point_in_polygon(self, point: tuple, polygon: np.ndarray) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y # Corrected this line to update p1x, p1y for next iteration
        return inside
    
    def _play_voice_alert(self, text):
        """Converts text to speech and plays it directly on the device using pyttsx3."""
        with _tts_lock_fzids: # Acquire lock before using TTS engine
            try:
                engine = pyttsx3.init()
                # You can set properties like rate or volume here if needed
                # engine.setProperty('rate', 150) # Speed of speech
                # engine.setProperty('volume', 0.9) # Volume (0.0 to 1.0)
                
                # You can try to select a specific voice if available
                # voices = engine.getProperty('voices')
                # for voice in voices:
                #     if "english" in voice.name.lower() and "male" in voice.name.lower():
                #         engine.setProperty('voice', voice.id)
                #         break

                engine.say(text)
                engine.runAndWait()
                print(f"Voice alert played on device: '{text}'")
            except Exception as e:
                print(f"Failed to play voice alert on device: {e}")
                self._add_log_to_queue(f"Voice alert failed: {e}")

    def _trigger_alert(self, zone_name: str, object_class: str, confidence: float, current_frame_for_snapshot: np.ndarray):
        """Trigger intrusion alert with enhanced visibility, voice alert, and snapshot."""
        current_time = time.time()
        
        # Set overall alert active state for visual overlay
        self.alert_active = True
        set_alert_active(True) # Update global flag for frontend
        self.alert_start_time = current_time
        
        # Generate and play voice alert using pyttsx3 (subject to cooldown)
        if current_time - self.last_audio_alert_time >= self.audio_alert_cooldown:
            alert_text = f"Warning! {object_class} detected in {zone_name}. Intrusion alert!"
            threading.Thread(target=self._play_voice_alert, args=(alert_text,)).start()
            self.last_audio_alert_time = current_time # Update audio alert time

        # Take a snapshot
        snapshot_filename = os.path.join(SNAPSHOTS_DIR, f"intrusion_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        try:
            cv2.imwrite(snapshot_filename, current_frame_for_snapshot)
            print(f"Intrusion snapshot saved: {snapshot_filename}")
            self._add_log_to_queue(f"Snapshot saved: {os.path.basename(snapshot_filename)}")
        except Exception as e:
            print(f"Error saving snapshot: {e}")
            self._add_log_to_queue(f"Snapshot failed: {object_class} in {zone_name}.")

        # Log intrusion
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'zone': zone_name,
            'object_class': object_class,
            'confidence': f"{confidence:.2f}", # Store as string for JSON
            'zone_scale': f"{self.zones[zone_name]['scale_factor']:.2f}",
            'snapshot_file': os.path.basename(snapshot_filename) # Add snapshot file name to log
        }
        self.intrusion_logs.append(log_entry)
        self._save_log() # Save log to file
        self._add_log_to_queue(f"Intrusion detected: {object_class} in {zone_name} (Confidence: {confidence:.2f})")

    def _add_log_to_queue(self, message: str):
        """Adds a log message to the thread-safe queue for frontend display."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        intrusion_log_queue.put(f"[{timestamp}] {message}")

    def _save_log(self):
        """Save intrusion logs to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.intrusion_logs, f, indent=2)
        except Exception as e:
            print(f"Error saving log: {e}")
    
    def _draw_zones(self, frame: np.ndarray):
        """Draw forbidden zones on the frame with light colors"""
        for zone_name, zone in self.zones.items():
            if not zone['active']:
                continue
                
            points = zone['points']
            color = zone['color']
            
            # Draw zone polygon with light semi-transparent fill
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)  # Light overlay
            
            # Draw bright border
            cv2.polylines(frame, [points], True, color, 4)
            
            # Draw zone label with better visibility
            center = zone['center']
            label = f"{zone_name.upper()} (Scale: {zone['scale_factor']:.2f})"
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (center[0] - text_size[0]//2 - 5, center[1] - 25),
                            (center[0] + text_size[0]//2 + 5, center[1] - 5), (0, 0, 0), -1)
            
            cv2.putText(frame, label, (center[0] - text_size[0]//2, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _draw_detections(self, frame: np.ndarray, results):
        """Draw detection boxes and check for intrusions with consistent drone sizes"""
        # A copy of the frame is needed to draw detections and send snapshot with drawn elements
        frame_with_detections_and_zones = frame.copy() 

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if class_id not in self.target_classes:
                        continue
                    
                    object_class = self.target_classes[class_id]
                    
                    if object_class in ['bird', 'cat', 'dog']: # Treating these as potential drones
                        drone_size = 60
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        x1 = center_x - drone_size // 2
                        x2 = center_x + drone_size // 2
                        y1 = center_y - drone_size // 2
                        y2 = center_y + drone_size // 2
                        
                        object_class = 'drone' # Rename for display
                    
                    centroid_x = int((x1 + x2) / 2)
                    centroid_y = int((y1 + y2) / 2)
                    centroid = (centroid_x, centroid_y)
                    
                    intrusion_detected = False
                    intruded_zone = None
                    
                    for zone_name, zone in self.zones.items():
                        if not zone['active']:
                            continue
                            
                        if self._point_in_polygon(centroid, zone['points']):
                            # Trigger alert, passing the current frame (with zones drawn) for the snapshot
                            self._trigger_alert(zone_name, object_class, confidence, frame_with_detections_and_zones)
                            intrusion_detected = True
                            intruded_zone = zone_name
                            break
                    
                    # Draw detection box with bright colors for visibility
                    if intrusion_detected:
                        color = (0, 0, 255)  # Bright red for intrusion
                        thickness = 4
                    else:
                        color = (0, 255, 0)  # Bright green for normal detection
                        thickness = 2
                    
                    cv2.rectangle(frame_with_detections_and_zones, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    cv2.circle(frame_with_detections_and_zones, centroid, 8, color, -1)
                    cv2.circle(frame_with_detections_and_zones, centroid, 10, (255, 255, 255), 2)
                    
                    label = f"{object_class}: {confidence:.2f}"
                    if intrusion_detected:
                        label += f" [INTRUSION - {intruded_zone.upper()}]"
                    
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame_with_detections_and_zones, (int(x1), int(y1) - 30), 
                                    (int(x1) + text_size[0] + 10, int(y1) - 5), (0, 0, 0), -1)
                    cv2.putText(frame_with_detections_and_zones, label, (int(x1) + 5, int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Copy the modified frame back to the original reference
        frame[:] = frame_with_detections_and_zones[:] 
    
    def _draw_alert_overlay(self, frame: np.ndarray):
        """Draw enhanced alert overlay when intrusion is detected"""
        current_time = time.time()
        if not self.alert_active or current_time - self.alert_start_time > self.alert_duration:
            self.alert_active = False
            set_alert_active(False) # Clear global flag
            return
        
        alpha = 0.4 * (1 + np.sin(current_time * 15))  # Faster pulsing
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), (0, 0, 255), -1)
        cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0, frame)
        
        border_thickness = 20
        cv2.rectangle(frame, (0, 0), (self.frame_width, self.frame_height), 
                        (0, 0, 255), border_thickness)
        
        alert_text = "🚨 INTRUSION DETECTED 🚨"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)[0]
        text_x = (self.frame_width - text_size[0]) // 2
        text_y = (self.frame_height + text_size[1]) // 2
        
        cv2.rectangle(frame, (text_x - 20, text_y - text_size[1] - 10),
                        (text_x + text_size[0] + 20, text_y + 10), (0, 0, 0), -1)
        
        cv2.putText(frame, alert_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
        
        warning_text = "UNAUTHORIZED ACCESS DETECTED"
        warning_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        warning_x = (self.frame_width - warning_size[0]) // 2
        warning_y = text_y + 80
        
        cv2.putText(frame, warning_text, (warning_x, warning_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    def generate_frames(self):
        """Generator function to yield processed frames for Flask streaming."""
        fps_start_time = time.time()
        frame_count = 0

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self._add_log_to_queue("Video capture not available or stream ended. Attempting to re-open...")
                time.sleep(1) # Wait before retrying
                try:
                    # Re-initialize the camera with its original source
                    if isinstance(self.video_source_init, str) and self.video_source_init.isdigit():
                        self.cap = cv2.VideoCapture(int(self.video_source_init))
                    else:
                        self.cap = cv2.VideoCapture(self.video_source_init)

                    if not self.cap.isOpened():
                        raise IOError("Failed to re-open video source.")
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self._add_log_to_queue("Video source re-opened successfully.")
                except Exception as e:
                    self._add_log_to_queue(f"Failed to re-open video source: {e}. Stopping stream.")
                    self.running = False
                continue

            ret, frame = self.cap.read()
            if not ret:
                self._add_log_to_queue("Error: Could not read frame from video source. Stream might have ended or disconnected.")
                self.running = False # End the stream if frames cannot be read
                continue
            
            # Draw zones first
            self._draw_zones(frame)
            
            # Run YOLO detection if model is loaded
            if self.model:
                results = self.model(frame, verbose=False)
                self._draw_detections(frame, results)
            
            # Draw alert overlay
            self._draw_alert_overlay(frame)
            
            # Calculate and display FPS (can be integrated into UI later if needed)
            frame_count += 1
            if time.time() - fps_start_time >= 1.0: # Update FPS every second
                fps = frame_count / (time.time() - fps_start_time)
                # print(f"FPS: {fps:.2f}") # For console logging
                fps_start_time = time.time()
                frame_count = 0
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        self.cleanup() # Ensure cleanup when the generator stops

    def cleanup(self):
        """Clean up resources"""
        print("Shutting down Forbidden Zone IDS...")
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows() # This might be problematic in Flask, but keeping as per original code.
        
        # Save final log
        self._save_log()
        
        print("System shutdown complete.")

    def stop(self):
        """Stops the main loop of the IDS."""
        print("Forbidden Zone IDS stop requested.")
        self.running = False

    def toggle_zone_active(self, zone_name: str):
        if zone_name in self.zones:
            self.zones[zone_name]['active'] = not self.zones[zone_name]['active']
            self._add_log_to_queue(f"Zone {zone_name} {'activated' if self.zones[zone_name]['active'] else 'deactivated'}.")
            print(f"Zone {zone_name} {'activated' if self.zones[zone_name]['active'] else 'deactivated'}")
            return self.zones[zone_name]['active']
        return None

    def scale_zone(self, zone_name: str, increase: bool):
        scale_factor = 1.1 if increase else 0.9
        if zone_name in self.zones:
            self._scale_zone(zone_name, scale_factor)
            self._add_log_to_queue(f"Zone {zone_name} scaled by {scale_factor:.2f} (new scale: {self.zones[zone_name]['scale_factor']:.2f}).")
            print(f"Zone {zone_name} scaled by {scale_factor:.2f}")
            return self.zones[zone_name]['scale_factor']
        return None
    
    def reset_zones(self):
        self._reset_zones()
        self._add_log_to_queue("All zones reset to default.")
        print("All zones reset to default.")

    def _reset_zones(self):
        """Reset zones to default configuration"""
        self.zones['zone1']['points'] = np.array([
            [200, 200], [500, 200], [500, 400], [200, 400]
        ], dtype=np.int32)
        self.zones['zone1']['scale_factor'] = 1.0
        
        self.zones['zone2']['points'] = np.array([
            [700, 300], [1000, 300], [1000, 500], [700, 500]
        ], dtype=np.int32)
        self.zones['zone2']['scale_factor'] = 1.0
        
        self._update_zone_centers()