import warnings
warnings.simplefilter(action='ignore', category=Warning)
import cv2
import torch # Re-added for torch.hub.load
import numpy as np
from PIL import Image # Re-added for Image.fromarray
import pyttsx3
import os
import threading
import queue
from datetime import datetime
import time
import smtplib
import ssl
from email.message import EmailMessage

# ==================== Email Configuration ====================
EMAIL_ADDRESS = "muttuh028@gmail.com"
EMAIL_PASSWORD = "kcxj atas xddm bbqr"
TO_EMAIL = "hmuktanandg@gmail.com"

# --- Global Configuration ---
# IMPORTANT: For torch.hub.load('ultralytics/yolov5', ...), this path should be to your 'best.pt'
# and the system needs to be able to access the 'ultralytics/yolov5' GitHub repository.
# If you face 'hubconf.py' errors, it means the repository couldn't be downloaded/found.
YOLO_MODEL_PATH = "yolov8_models/best.pt"
SNAPSHOTS_DIR = "static/uploads"
LOG_DIR = "detection_logs_live"

os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

vihangvetri_log_queue = queue.Queue()
vihangvetri_alert_active_flag = False
vihangvetri_alert_lock = threading.Lock()
_tts_lock_vihangvetri = threading.Lock()

def set_vihangvetri_alert_active(state: bool):
    global vihangvetri_alert_active_flag
    with vihangvetri_alert_lock:
        vihangvetri_alert_active_flag = state

def get_vihangvetri_alert_active() -> bool:
    with vihangvetri_alert_lock:
        return vihangvetri_alert_active_flag

def get_vihangvetri_log_queue():
    return vihangvetri_log_queue

class VihangVetriCamera:
    def __init__(self, video_source: str = "0"):
        self.video_source_init = video_source
        print(f"Initializing VihangVetri with source: {video_source}")

        # Initialize YOLOv5 model using torch.hub.load as in original code
        print(f"Loading YOLOv5 model from {YOLO_MODEL_PATH} using torch.hub.load...")
        try:
            model_abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", YOLO_MODEL_PATH)
            # Using source='github' as in your original code. This will attempt to clone/download
            # the ultralytics/yolov5 repo to find hubconf.py, then load your custom model.
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_abs_path, source='github', verbose=False)
            print("YOLOv5 model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLOv5 model from {YOLO_MODEL_PATH}: {e}")
            self.model = None
            self._add_log_to_queue(f"ERROR: Failed to load YOLOv5 model: {e}. Check network and GitHub access.")

        try:
            if isinstance(video_source, str) and video_source.isdigit():
                self.cap = cv2.VideoCapture(int(video_source))
            else:
                self.cap = cv2.VideoCapture(video_source)
            
            if not self.cap.isOpened():
                raise IOError(f"Cannot open video source: {video_source}")
            
            ret, frame = self.cap.read()
            if not ret:
                raise IOError("Could not read the first frame from the video source.")
            
            self.frame_height, self.frame_width, _ = frame.shape
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            print(f"Video source opened successfully. Resolution: {self.frame_width}x{self.frame_height}")

        except Exception as e:
            print(f"Error opening video source {video_source}: {e}")
            self.cap = None
            self._add_log_to_queue(f"ERROR: Failed to open video source: {e}")

        # Zone Configuration (initial values, can be set via Flask frontend)
        initial_zone_width = int(self.frame_width * 0.5) if self.cap else 600
        initial_zone_height = int(self.frame_height * 0.5) if self.cap else 400
        zone_x = (self.frame_width - initial_zone_width) // 2 if self.cap else 100
        zone_y = (self.frame_height - initial_zone_height) // 2 if self.cap else 100

        self.zone = {'x': zone_x, 'y': zone_y, 'width': initial_zone_width, 'height': initial_zone_height}
        self.ZONE_COLOR_NORMAL = (0, 255, 0)
        self.ZONE_COLOR_ACTIVE = (0, 255, 255)
        self.ZONE_THICKNESS = 2

        self.alert_active = False # For visual overlay
        self.alert_start_time = 0
        self.alert_duration = 3.0
        self.last_audio_alert_time = 0
        self.audio_alert_cooldown = 5.0
        self.last_email_alert_time = 0
        self.email_alert_cooldown = 30.0
        self.alert_triggered_in_current_event = False # To ensure alerts (email/voice) only trigger once per "event"

        self.log_file = os.path.join(LOG_DIR, f"vihangvetri_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.detection_logs = []
        
        self.running = True
        
        print("VihangVetri IDS initialized successfully!")

    def _play_voice_alert(self, text):
        with _tts_lock_vihangvetri:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say(text)
                engine.runAndWait()
                print(f"Voice alert played on device: '{text}'")
            except Exception as e:
                print(f"Failed to play voice alert on device: {e}")
                self._add_log_to_queue(f"Voice alert failed: {e}")

    def _send_alert_email(self, screenshot_path: str, drone_crop_path: str, object_class: str, confidence: float, cx: int, cy: int):
        current_time = time.time()
        if current_time - self.last_email_alert_time < self.email_alert_cooldown:
            return # Cooldown not over
            
        self.last_email_alert_time = current_time

        msg = EmailMessage()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = TO_EMAIL
        msg["Subject"] = f"VIHANGVETRI ALERT: {object_class.upper()} Detected in Monitored Zone"
        
        body = f"""
        Dear Security Monitor,

        An unauthorized aerial object has been detected in the monitored zone.

        Details:
        - Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - Object Detected: {object_class.upper()}
        - Confidence: {confidence:.2f}
        - Zone: X:{self.zone['x']}, Y:{self.zone['y']}, Width:{self.zone['width']}, Height:{self.zone['height']}
        - Drone Coordinates (Center): ({cx}, {cy})

        Please review the attached snapshot and cropped drone image for visual confirmation.

        This is an automated alert from VihangVetri.

        Sincerely,
        VihangVetri Drone Detection System
        """
        msg.set_content(body)

        try:
            if os.path.exists(screenshot_path):
                with open(screenshot_path, 'rb') as f:
                    msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename="screenshot.jpg")
            else:
                self._add_log_to_queue(f"Warning: Full screenshot file not found at {screenshot_path}. Email may be incomplete.")

            if os.path.exists(drone_crop_path):
                with open(drone_crop_path, 'rb') as f:
                    msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename="drone_crop.jpg")
            else:
                self._add_log_to_queue(f"Warning: Drone crop file not found at {drone_crop_path}. Email may be incomplete.")
            
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
                smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                smtp.send_message(msg)
            print(f"Alert email sent to {TO_EMAIL}")
            self._add_log_to_queue(f"Email alert sent for {object_class} at ({cx}, {cy}).")
        except FileNotFoundError:
            print(f"Error: One or more image files not found for email. Cannot send email.")
            self._add_log_to_queue(f"Email failed: Image file missing for {object_class}.")
        except smtplib.SMTPAuthenticationError as e:
            print(f"Error: SMTP Authentication failed. Check your Gmail app password and email address. Details: {e}")
            print("Ensure you are using a Gmail App Password if 2-Factor Authentication is enabled.")
            self._add_log_to_queue(f"Email failed: SMTP Auth error for {object_class}.")
        except smtplib.SMTPException as e:
            print(f"Error sending email: {e}")
            self._add_log_to_queue(f"Email failed: SMTP error for {object_class}.")
        except Exception as e:
            print(f"An unexpected error occurred during email sending: {e}")
            self._add_log_to_queue(f"Email failed: Unknown error for {object_class}.")


    def _trigger_alert_actions(self, object_class: str, confidence: float, current_frame_for_snapshot: np.ndarray, x1_obj: int, y1_obj: int, x2_obj: int, y2_obj: int, cx_obj: int, cy_obj: int):
        """
        Triggers voice alert, saves snapshots, and sends email.
        This function is called only when a drone *enters* the zone or after cooldown.
        """
        
        # Voice Alert
        current_time = time.time()
        if current_time - self.last_audio_alert_time >= self.audio_alert_cooldown:
            alert_text = f"Warning! {object_class} detected in the monitored zone."
            threading.Thread(target=self._play_voice_alert, args=(alert_text,)).start()
            self.last_audio_alert_time = current_time

        # Take a snapshot and drone crop
        screenshot_filename = os.path.join(SNAPSHOTS_DIR, f"vihangvetri_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        drone_crop_filename = os.path.join(SNAPSHOTS_DIR, f"vihangvetri_drone_crop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        
        screenshot_saved = False
        drone_crop_saved = False

        if current_frame_for_snapshot is not None and current_frame_for_snapshot.size > 0:
            try:
                cv2.imwrite(screenshot_filename, current_frame_for_snapshot)
                print(f"VihangVetri full screenshot saved: {screenshot_filename}")
                self._add_log_to_queue(f"Full Snapshot saved: {os.path.basename(screenshot_filename)}")
                screenshot_saved = True
            except Exception as e:
                print(f"Error saving full screenshot: {e}")
                self._add_log_to_queue(f"Full Snapshot failed: {object_class}.")

            y1_crop = max(0, int(y1_obj))
            y2_crop = min(self.frame_height, int(y2_obj))
            x1_crop = max(0, int(x1_obj))
            x2_crop = min(self.frame_width, int(x2_obj))

            if y2_crop > y1_crop and x2_crop > x1_crop:
                drone_crop = current_frame_for_snapshot[y1_crop:y2_crop, x1_crop:x2_crop]
                if drone_crop.size > 0:
                    try:
                        cv2.imwrite(drone_crop_filename, drone_crop)
                        print(f"VihangVetri drone crop saved: {drone_crop_filename}")
                        self._add_log_to_queue(f"Drone Crop saved: {os.path.basename(drone_crop_filename)}")
                        drone_crop_saved = True
                    except Exception as e:
                        print(f"Error saving drone crop: {e}")
                        self._add_log_to_queue(f"Drone Crop failed: {object_class}.")
                else:
                    print("Warning: Cropped drone image is empty.")
                    self._add_log_to_queue("Warning: Drone crop empty.")
            else:
                print("Warning: Invalid crop coordinates for drone image.")
                self._add_log_to_queue("Warning: Invalid drone crop coordinates.")
        else:
            print("Warning: Attempted to save an empty or invalid snapshot frame (original).")
            self._add_log_to_queue("Warning: Snapshot/Crop not saved (empty frame).")


        if screenshot_saved or drone_crop_saved:
            self._send_alert_email(screenshot_filename, drone_crop_filename, object_class, confidence, cx_obj, cy_obj)
        else:
            self._add_log_to_queue(f"Email not sent for {object_class} due to missing images.")


        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'object_class': object_class,
            'confidence': f"{confidence:.2f}",
            'zone': self.zone,
            'drone_coords': {'x1': int(x1_obj), 'y1': int(y1_obj), 'x2': int(x2_obj), 'y2': int(y2_obj), 'cx': cx_obj, 'cy': cy_obj},
            'full_snapshot_file': os.path.basename(screenshot_filename) if screenshot_saved else "N/A",
            'drone_crop_file': os.path.basename(drone_crop_filename) if drone_crop_saved else "N/A"
        }
        self.detection_logs.append(log_entry)
        self._save_log()
        self._add_log_to_queue(f"Drone detected: {object_class} (Confidence: {confidence:.2f}) at ({cx_obj}, {cy_obj})")


    def _add_log_to_queue(self, message: str):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        vihangvetri_log_queue.put(f"[{timestamp}] {message}")

    def _save_log(self):
        try:
            with open(self.log_file, 'w') as f:
                import json
                json.dump(self.detection_logs, f, indent=2)
        except Exception as e:
            print(f"Error saving log: {e}")
    
    def _draw_zone(self, frame: np.ndarray):
        x, y, w, h = self.zone['x'], self.zone['y'], self.zone['width'], self.zone['height']
        
        x = max(0, min(x, self.frame_width - 1))
        y = max(0, min(y, self.frame_height - 1))
        w = max(10, min(w, self.frame_width - x))
        h = max(10, min(h, self.frame_height - y))

        self.zone['x'], self.zone['y'], self.zone['width'], self.zone['height'] = x, y, w, h

        cv2.rectangle(frame, (x, y), (x + w, y + h), self.ZONE_COLOR_NORMAL, self.ZONE_THICKNESS)
        cv2.putText(frame, "Monitored Zone", (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.ZONE_COLOR_NORMAL, 2)

    def _draw_detections(self, frame: np.ndarray, results):
        original_frame_for_snapshot = frame.copy() 
        
        drone_in_zone_this_frame = False
        first_drone_in_zone_details = None

        # Define zone boundaries from self.zone
        rect_x_min = self.zone['x']
        rect_y_min = self.zone['y']
        rect_x_max = self.zone['x'] + self.zone['width']
        rect_y_max = self.zone['y'] + self.zone['height']

        # Ensure min/max are correctly ordered for intersection check
        zone_x1, zone_y1 = min(rect_x_min, rect_x_max), min(rect_y_min, rect_y_max)
        zone_x2, zone_y2 = max(rect_x_min, rect_x_max), max(rect_y_min, rect_y_max)


        for *xyxy, conf, cls in results.xyxy[0].tolist(): # Iterate directly over the list of detections
            x1, y1, x2, y2 = xyxy
            
            detected_class_name = self.model.names[int(cls)] if self.model and hasattr(self.model, 'names') else "unknown"

            # Filter for 'Drone' class if your model has it, or remove if it detects all objects
            # Based on your original code, it seems to expect 'Drone' as a class.
            # You might need to adjust 'Drone' to the actual class name in your 'best.pt'
            if detected_class_name.lower() != 'drone': # Assuming 'drone' is the class name in your model
                continue
            
            if conf > 0.5: # Confidence threshold
                # Draw the bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                # Display the confidence score above the box
                text_conf = "{:.2f}%".format(conf * 100)
                cv2.putText(frame, text_conf, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Display the bounding box coordinates below the box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                text_coords = "({}, {})".format(cx, cy) # Changed to center coords for consistency
                cv2.putText(frame, text_coords, (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Check for intersection with the zone (AABB intersection)
                # Check if the drone's bounding box overlaps with the zone
                # This is a more robust intersection check than just the centroid or corners
                overlap_x = max(0, min(x2, zone_x2) - max(x1, zone_x1))
                overlap_y = max(0, min(y2, zone_y2) - max(y1, zone_y1))

                if overlap_x > 0 and overlap_y > 0: # If there's any overlap
                    drone_in_zone_this_frame = True
                    cv2.putText(frame, "Warning: Drone Detected Under Restricted Area!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if first_drone_in_zone_details is None:
                        first_drone_in_zone_details = {
                            'object_class': detected_class_name,
                            'confidence': conf,
                            'x1_obj': x1, 'y1_obj': y1, 'x2_obj': x2, 'y2_obj': y2,
                            'cx_obj': cx, 'cy_obj': cy
                        }
        
        # Trigger alert actions ONLY if a drone is in the zone AND it's a new event or cooldown allows
        if drone_in_zone_this_frame:
            # The visual alert (pulsing indicator) is continuous as long as drone is in zone
            self.alert_active = True
            set_vihangvetri_alert_active(True)
            self.alert_start_time = time.time() # Keep resetting visual alert timer as long as drone is in zone

            # Trigger email/voice only once per event or after cooldown
            if not self.alert_triggered_in_current_event:
                self._trigger_alert_actions(
                    first_drone_in_zone_details['object_class'],
                    first_drone_in_zone_details['confidence'],
                    original_frame_for_snapshot,
                    first_drone_in_zone_details['x1_obj'],
                    first_drone_in_zone_details['y1_obj'],
                    first_drone_in_zone_details['x2_obj'],
                    first_drone_in_zone_details['y2_obj'],
                    first_drone_in_zone_details['cx_obj'],
                    first_drone_in_zone_details['cy_obj']
                )
                self.alert_triggered_in_current_event = True # Mark alert as triggered for this event
        else:
            # If no drone is in the zone in this frame, reset the alert trigger state
            self.alert_active = False
            set_vihangvetri_alert_active(False)
            self.alert_triggered_in_current_event = False # Reset for next intrusion event

    def _draw_alert_overlay(self, frame: np.ndarray):
        current_time = time.time()
        if not self.alert_active or current_time - self.alert_start_time > self.alert_duration:
            self.alert_active = False
            set_vihangvetri_alert_active(False)
            return
        
        alpha = 0.4 * (1 + np.sin(current_time * 15))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), (0, 0, 255), -1)
        cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0, frame)
        
        border_thickness = 20
        cv2.rectangle(frame, (0, 0), (self.frame_width, self.frame_height), 
                        (0, 0, 255), border_thickness)
        
        alert_text = "🚨 DRONE IN ZONE 🚨"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)[0]
        text_x = (self.frame_width - text_size[0]) // 2
        text_y = (self.frame_height + text_size[1]) // 2
        
        cv2.rectangle(frame, (text_x - 20, text_y - text_size[1] - 10),
                        (text_x + text_size[0] + 20, text_y + 10), (0, 0, 0), -1)
        
        cv2.putText(frame, alert_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
        
        warning_text = "UNAUTHORIZED AERIAL OBJECT"
        warning_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        warning_x = (self.frame_width - warning_size[0]) // 2
        warning_y = text_y + 80
        
        cv2.putText(frame, warning_text, (warning_x, warning_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    def generate_frames(self):
        fps_start_time = time.time()
        frame_count = 0

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self._add_log_to_queue("Video capture not available or stream ended. Attempting to re-open...")
                time.sleep(1)
                try:
                    if isinstance(self.video_source_init, str) and self.video_source_init.isdigit():
                        self.cap = cv2.VideoCapture(int(self.video_source_init))
                    else:
                        self.cap = cv2.VideoCapture(self.video_source_init)

                    if not self.cap.isOpened():
                        raise IOError("Failed to re-open video source.")
                    
                    if not (isinstance(self.video_source_init, str) and self.video_source_init.isdigit()):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self._add_log_to_queue("Video source re-opened successfully.")
                except Exception as e:
                    self._add_log_to_queue(f"Failed to re-open video source: {e}. Stopping stream.")
                    self.running = False
                continue

            ret, frame = self.cap.read()
            if not ret:
                self._add_log_to_queue("Error: Could not read frame from video source. Stream might have ended or disconnected. Resetting video if it's a file.")
                if not (isinstance(self.video_source_init, str) and self.video_source_init.isdigit()):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.running = False
                    continue
            
            self._draw_zone(frame)
            
            if self.model:
                # Convert frame to a format that torch.hub.load model expects (RGB PIL Image)
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results = self.model(img_pil, size=640) # Run inference
                self._draw_detections(frame, results)
            
            self._draw_alert_overlay(frame)
            
            frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                fps = frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                frame_count = 0
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        self.cleanup()

    def cleanup(self):
        print("Shutting down VihangVetri IDS...")
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        
        self._save_log()
        
        print("System shutdown complete.")

    def stop(self):
        print("VihangVetri IDS stop requested.")
        self.running = False

    def set_zone(self, x: int, y: int, width: int, height: int):
        self.zone['x'] = x
        self.zone['y'] = y
        self.zone['width'] = width
        self.zone['height'] = height
        self._add_log_to_queue(f"Zone set to X:{x}, Y:{y}, W:{width}, H:{height}.")
        print(f"VihangVetri Zone set to: {self.zone}")

    def reset_zone(self):
        initial_zone_width = int(self.frame_width * 0.5) if self.cap else 600
        initial_zone_height = int(self.frame_height * 0.5) if self.cap else 400
        zone_x = (self.frame_width - initial_zone_width) // 2 if self.cap else 100
        zone_y = (self.frame_height - initial_zone_height) // 2 if self.cap else 100
        self.zone = {'x': zone_x, 'y': zone_y, 'width': initial_zone_width, 'height': initial_zone_height}
        self._add_log_to_queue("Zone reset to default.")
        print("VihangVetri Zone reset to default.")