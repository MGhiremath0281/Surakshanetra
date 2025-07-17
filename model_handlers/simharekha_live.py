import cv2
import numpy as np
from ultralytics import YOLO
import time
from playsound import playsound
from gtts import gTTS
import os
from datetime import datetime
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from typing import List, Tuple, Dict, Any

class ForbiddenZoneIDS:
    def __init__(self, model_path: str = "yolov8x.pt", video_source: int = 0):
        """
        Initialize the Forbidden Zone Intrusion Detection System
        
        Args:
            model_path: Path to YOLOv8x model
            video_source: Video source (0 for webcam, or path to video file)
        """
        # Initialize YOLO model
        print("Loading YOLOv8x model...")
        self.model = YOLO(model_path)
        
        # Video capture
        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Define target classes for detection (including drones)
        self.target_classes = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            14: 'bird',  # Often detected as drones
            15: 'cat',   # Small objects
            16: 'dog',   # Small objects
            # Note: YOLOv8 doesn't have explicit drone class, but we can detect flying objects
        }
        
        # Initialize forbidden zones (polygons) - lighter colors
        self.zones = {
            'zone1': {
                'points': np.array([
                    [200, 200],
                    [500, 200],
                    [500, 400],
                    [200, 400]
                ], dtype=np.int32),
                'color': (0, 100, 255),  # Light Red/Orange
                'scale_factor': 1.0,
                'center': None,
                'active': True
            },
            'zone2': {
                'points': np.array([
                    [700, 300],
                    [1000, 300],
                    [1000, 500],
                    [700, 500]
                ], dtype=np.int32),
                'color': (255, 100, 0),  # Light Blue
                'scale_factor': 1.0,
                'center': None,
                'active': True
            }
        }
        
        # Update zone centers
        self._update_zone_centers()
        
        # Alert system
        self.alert_active = False
        self.alert_start_time = 0
        self.alert_duration = 3.0  # seconds (visual alert overlay)
        self.last_audio_alert_time = 0
        self.audio_alert_cooldown = 3.0  # seconds (for gTTS voice alerts)
        self.last_email_alert_time = 0
        self.email_alert_cooldown = 30.0 # seconds (for email alerts, e.g., 30 seconds)
        
        # Logging
        self.log_file = f"intrusion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.intrusion_logs = []
        
        # Control flags
        self.running = True
        self.show_help = False
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'zone1_intrusions': 0,
            'zone2_intrusions': 0,
            'fps': 0
        }
        
        # Temporary file for gTTS audio
        self.tts_audio_file = "temp_alert.mp3" 

        # Email Credentials
        self.EMAIL_ADDRESS = "muttuh028@gmail.com"
        self.EMAIL_PASSWORD = "kcxj atas xddm bbqr" # Replace with your Gmail app password
        self.TO_EMAIL = "hmuktanandg@gmail.com"
        
        print("Forbidden Zone IDS initialized successfully!")
        print("Press 'h' for help and controls")
    
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
        
        # Scale points around center
        scaled_points = []
        for point in points:
            # Translate to origin
            translated = point - center
            # Scale
            scaled = translated * scale_factor
            # Translate back
            final_point = scaled + center
            scaled_points.append(final_point)
        
        # Update zone points
        zone['points'] = np.array(scaled_points, dtype=np.int32)
        zone['scale_factor'] *= scale_factor
        
        # Ensure points stay within frame bounds
        zone['points'][:, 0] = np.clip(zone['points'][:, 0], 0, self.frame_width)
        zone['points'][:, 1] = np.clip(zone['points'][:, 1], 0, self.frame_height)
    
    def _point_in_polygon(self, point: Tuple[int, int], polygon: np.ndarray) -> bool:
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
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _send_alert_email(self, snapshot_path: str, zone_name: str, object_class: str, confidence: float):
        """Sends an email with the intrusion snapshot and details."""
        current_time = time.time()
        if current_time - self.last_email_alert_time < self.email_alert_cooldown:
            return # Cooldown not over
            
        self.last_email_alert_time = current_time # Update last email time

        msg = MIMEMultipart()
        msg['From'] = self.EMAIL_ADDRESS
        msg['To'] = self.TO_EMAIL
        msg['Subject'] = f"CRITICAL INTRUSION ALERT - {zone_name.upper()} - {object_class.upper()}"

        body = f"""
        Dear Security Monitor,

        An intrusion has been detected in a forbidden zone.

        Details:
        - Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - Zone: {zone_name.upper()}
        - Object Detected: {object_class.upper()}
        - Confidence: {confidence:.2f}

        Please review the attached snapshot for visual confirmation.

        This is an automated alert.

        Sincerely,
        Forbidden Zone IDS
        """
        msg.attach(MIMEText(body, 'plain'))

        try:
            with open(snapshot_path, 'rb') as f:
                img = MIMEImage(f.read(), name=os.path.basename(snapshot_path))
                msg.attach(img)
            
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(self.EMAIL_ADDRESS, self.EMAIL_PASSWORD)
                smtp.send_message(msg)
            print(f"Email alert sent successfully to {self.TO_EMAIL}!")
        except FileNotFoundError:
            print(f"Error: Snapshot file not found at {snapshot_path}. Cannot send email.")
        except smtplib.SMTPAuthenticationError as e:
            print(f"Error: SMTP Authentication failed. Check your Gmail app password and email address. Details: {e}")
            print("Ensure you are using a Gmail App Password if 2-Factor Authentication is enabled.")
        except smtplib.SMTPException as e:
            print(f"Error sending email: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during email sending: {e}")


    def _trigger_alert(self, zone_name: str, object_class: str, confidence: float, current_frame_for_snapshot: np.ndarray):
        """Trigger intrusion alert with enhanced visibility, voice alert, snapshot, and email."""
        current_time = time.time()
        
        # Set overall alert active state
        self.alert_active = True
        self.alert_start_time = current_time
        
        # Update statistics
        if zone_name == 'zone1':
            self.stats['zone1_intrusions'] += 1
        else:
            self.stats['zone2_intrusions'] += 1
        
        # Generate and play voice alert using gTTS (subject to cooldown)
        if current_time - self.last_audio_alert_time >= self.audio_alert_cooldown:
            try:
                alert_text = f"Warning! {object_class} detected in {zone_name}. Intrusion alert!"
                tts = gTTS(text=alert_text, lang='en', slow=False)
                tts.save(self.tts_audio_file)
                print(f"Generated voice alert: '{alert_text}'")
                
                playsound(self.tts_audio_file)
                
                if os.path.exists(self.tts_audio_file):
                    os.remove(self.tts_audio_file)
                self.last_audio_alert_time = current_time # Update audio alert time
            except Exception as e:
                print(f"Error generating or playing voice alert with gTTS/playsound: {e}")
        
        # Take a snapshot and send email (subject to cooldown)
        if current_time - self.last_email_alert_time >= self.email_alert_cooldown:
            snapshot_filename = f"intrusion_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(snapshot_filename, current_frame_for_snapshot)
            print(f"Intrusion snapshot saved: {snapshot_filename}")
            
            self._send_alert_email(snapshot_filename, zone_name, object_class, confidence)
            
            # Optional: Remove the snapshot after sending email to save space
            # if os.path.exists(snapshot_filename):
            #     os.remove(snapshot_filename)
            #     print(f"Snapshot {snapshot_filename} removed after sending email.")


        # Log intrusion
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'zone': zone_name,
            'object_class': object_class,
            'confidence': confidence,
            'zone_scale': self.zones[zone_name]['scale_factor'],
            'email_sent': (current_time - self.last_email_alert_time < self.email_alert_cooldown) # True if email was just sent
        }
        
        self.intrusion_logs.append(log_entry)
        self._save_log() # Save log to file

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
            
            # Add background for text
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
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Only process target classes
                    if class_id not in self.target_classes:
                        continue
                    
                    object_class = self.target_classes[class_id]
                    
                    # Make all drone/small object sizes consistent
                    if object_class in ['bird', 'cat', 'dog']:  # Treating these as potential drones
                        # Normalize drone size to consistent dimensions
                        drone_size = 60  # Standard drone detection box size
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        x1 = center_x - drone_size // 2
                        x2 = center_x + drone_size // 2
                        y1 = center_y - drone_size // 2
                        y2 = center_y + drone_size // 2
                        
                        # Rename for display
                        object_class = 'drone'
                    
                    # Calculate centroid
                    centroid_x = int((x1 + x2) / 2)
                    centroid_y = int((y1 + y2) / 2)
                    centroid = (centroid_x, centroid_y)
                    
                    # Check if centroid is in any forbidden zone
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
                    
                    # Draw on the frame_with_detections_and_zones for the snapshot
                    cv2.rectangle(frame_with_detections_and_zones, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    
                    # Draw centroid with larger size for visibility
                    cv2.circle(frame_with_detections_and_zones, centroid, 8, color, -1)
                    cv2.circle(frame_with_detections_and_zones, centroid, 10, (255, 255, 255), 2)  # White border
                    
                    # Draw label with background for better visibility
                    label = f"{object_class}: {confidence:.2f}"
                    if intrusion_detected:
                        label += f" [INTRUSION - {intruded_zone.upper()}]"
                    
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame_with_detections_and_zones, (int(x1), int(y1) - 30), 
                                  (int(x1) + text_size[0] + 10, int(y1) - 5), (0, 0, 0), -1)
                    cv2.putText(frame_with_detections_and_zones, label, (int(x1) + 5, int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    self.stats['total_detections'] += 1
        
        # Assign the modified frame back to the original frame for display
        frame[:] = frame_with_detections_and_zones[:] 

    def _draw_alert_overlay(self, frame: np.ndarray):
        """Draw enhanced alert overlay when intrusion is detected"""
        if not self.alert_active:
            return
        
        current_time = time.time()
        if current_time - self.alert_start_time > self.alert_duration:
            self.alert_active = False
            return
        
        # Create intense pulsing red overlay
        alpha = 0.4 * (1 + np.sin(current_time * 15))  # Faster pulsing
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), (0, 0, 255), -1)
        cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0, frame)
        
        # Draw flashing border
        border_thickness = 20
        cv2.rectangle(frame, (0, 0), (self.frame_width, self.frame_height), 
                      (0, 0, 255), border_thickness)
        
        # Draw large alert text
        alert_text = "🚨 INTRUSION DETECTED 🚨"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)[0]
        text_x = (self.frame_width - text_size[0]) // 2
        text_y = (self.frame_height + text_size[1]) // 2
        
        # Add text background
        cv2.rectangle(frame, (text_x - 20, text_y - text_size[1] - 10),
                      (text_x + text_size[0] + 20, text_y + 10), (0, 0, 0), -1)
        
        cv2.putText(frame, alert_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
        
        # Add secondary warning text
        warning_text = "UNAUTHORIZED ACCESS DETECTED"
        warning_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        warning_x = (self.frame_width - warning_size[0]) // 2
        warning_y = text_y + 80
        
        cv2.putText(frame, warning_text, (warning_x, warning_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    def _draw_ui(self, frame: np.ndarray):
        """Draw user interface elements"""
        # Draw statistics
        stats_text = [
            f"FPS: {self.stats['fps']:.1f}",
            f"Total Detections: {self.stats['total_detections']}",
            f"Zone 1 Intrusions: {self.stats['zone1_intrusions']}",
            f"Zone 2 Intrusions: {self.stats['zone2_intrusions']}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw controls help
        if self.show_help:
            help_text = [
                "CONTROLS:",
                "h - Toggle help",
                "q - Quit",
                "1/2 - Toggle zone 1/2",
                "i/k - Increase/Decrease zone 1",
                "o/l - Increase/Decrease zone 2",
                "r - Reset zones",
                "s - Save screenshot"
            ]
            
            # Draw help background
            help_height = len(help_text) * 25 + 20
            cv2.rectangle(frame, (self.frame_width - 250, 10),
                          (self.frame_width - 10, help_height), (0, 0, 0), -1)
            
            for i, text in enumerate(help_text):
                cv2.putText(frame, text, (self.frame_width - 240, 35 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _handle_keyboard_input(self, key: int):
        """Handle keyboard input for zone control"""
        if key == ord('h'):
            self.show_help = not self.show_help
        elif key == ord('q'):
            self.running = False
        elif key == ord('1'):
            self.zones['zone1']['active'] = not self.zones['zone1']['active']
            print(f"Zone 1 {'activated' if self.zones['zone1']['active'] else 'deactivated'}")
        elif key == ord('2'):
            self.zones['zone2']['active'] = not self.zones['zone2']['active']
            print(f"Zone 2 {'activated' if self.zones['zone2']['active'] else 'deactivated'}")
        elif key == ord('i'):
            self._scale_zone('zone1', 1.1)
            print(f"Zone 1 increased (scale: {self.zones['zone1']['scale_factor']:.2f})")
        elif key == ord('k'):
            self._scale_zone('zone1', 0.9)
            print(f"Zone 1 decreased (scale: {self.zones['zone1']['scale_factor']:.2f})")
        elif key == ord('o'):
            self._scale_zone('zone2', 1.1)
            print(f"Zone 2 increased (scale: {self.zones['zone2']['scale_factor']:.2f})")
        elif key == ord('l'):
            self._scale_zone('zone2', 0.9)
            print(f"Zone 2 decreased (scale: {self.zones['zone2']['scale_factor']:.2f})")
        elif key == ord('r'):
            self._reset_zones()
            print("Zones reset to default")
        elif key == ord('s'):
            self._save_screenshot()
    
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
    
    def _save_screenshot(self):
        """Save current frame as screenshot (manual trigger)"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"manual_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, self.current_frame)
        print(f"Manual screenshot saved as {filename}")
    
    def run(self):
        """Main detection loop"""
        print("Starting Forbidden Zone Intrusion Detection System...")
        print("Press 'h' for help and controls")
        
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from video source")
                break
            
            # Store the current raw frame before drawing for manual screenshot
            self.current_frame = frame.copy() 
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            # Draw zones first
            self._draw_zones(frame)
            
            # Process detections and trigger alerts/emails. This also modifies 'frame'
            self._draw_detections(frame, results) # This method now passes 'frame' for snapshot
            
            # Draw alert overlay
            self._draw_alert_overlay(frame)
            
            # Draw UI (statistics, help)
            self._draw_ui(frame)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 30:
                fps_end_time = time.time()
                self.stats['fps'] = fps_counter / (fps_end_time - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # Display frame
            cv2.imshow('Forbidden Zone Intrusion Detection System', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # Key pressed
                self._handle_keyboard_input(key)
            
            # Check for window close
            if cv2.getWindowProperty('Forbidden Zone Intrusion Detection System', cv2.WND_PROP_VISIBLE) < 1:
                break
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Shutting down Forbidden Zone IDS...")
        
        # Save final log
        self._save_log()
        
        # Clean up temporary gTTS audio file if it exists
        if os.path.exists(self.tts_audio_file):
            try:
                os.remove(self.tts_audio_file)
                print(f"Cleaned up temporary audio file: {self.tts_audio_file}")
            except Exception as e:
                print(f"Error cleaning up temporary audio file: {e}")
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n=== FINAL STATISTICS ===")
        print(f"Total Detections: {self.stats['total_detections']}")
        print(f"Zone 1 Intrusions: {self.stats['zone1_intrusions']}")
        print(f"Zone 2 Intrusions: {self.stats['zone2_intrusions']}")
        print(f"Total Intrusions: {self.stats['zone1_intrusions'] + self.stats['zone2_intrusions']}")
        print(f"Log file saved as: {self.log_file}")
        print("System shutdown complete.")

def main():
    """Main function to run the Forbidden Zone IDS"""
    try:
        # Initialize the system
        ids = ForbiddenZoneIDS(
            model_path="yolov8x.pt",  # Will download if not present
            video_source=0  # Use webcam, change to video file path if needed
        )
        
        # Run the system
        ids.run()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user...")
    except Exception as e:
        print(f"An unhandled error occurred: {e}")
    finally:
        try:
            # Ensure cleanup is called even if an error occurs
            if 'ids' in locals() and isinstance(ids, ForbiddenZoneIDS):
                 ids.cleanup()
            else:
                print("IDS object not fully initialized or accessible, skipping cleanup.")
        except Exception as e:
            print(f"Error during final cleanup: {e}")

if __name__ == "__main__":
    main()