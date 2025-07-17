import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque, defaultdict
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import math
import os
import threading
import queue # Import queue for inter-thread communication

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    center: Tuple[int, int]
    assigned_person_id: Optional[int] = None

class TrajectoryTracker:
    def __init__(self, max_trail_length=30):
        self.trails = defaultdict(lambda: deque(maxlen=max_trail_length))
        self.colors = {}
        logger.debug("TrajectoryTracker initialized.")

    def update(self, track_id: int, center: Tuple[int, int]):
        self.trails[track_id].append(center)
        if track_id not in self.colors:
            np.random.seed(track_id)
            self.colors[track_id] = (
                int(np.random.randint(0, 255)),
                int(np.random.randint(0, 255)),
                int(np.random.randint(0, 255))
            )

    def draw_trails(self, frame: np.ndarray):
        for track_id, trail in list(self.trails.items()):
            if not trail: 
                continue
            if len(trail) > 1:
                color = self.colors.get(track_id, (255, 255, 255)) 
                points = np.array(trail, dtype=np.int32)
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    thickness = max(1, int(3 * alpha))
                    try:
                        cv2.line(frame, tuple(points[i-1]), tuple(points[i]), color, thickness)
                    except Exception as e:
                        logger.error(f"Error drawing trail line for track {track_id}: {e}")
                if len(points) >= 2:
                    try:
                        self.draw_arrow(frame, points[-2], points[-1], color)
                    except Exception as e:
                        logger.error(f"Error drawing trail arrow for track {track_id}: {e}")

    def draw_arrow(self, frame: np.ndarray, start: np.ndarray, end: np.ndarray, color: tuple):
        start_pt = tuple(map(int, start))
        end_pt = tuple(map(int, end))

        angle = math.atan2(end_pt[1] - start_pt[1], end_pt[0] - start_pt[0])
        arrow_length = 10
        arrow_angle = math.pi / 6
        x1 = int(end_pt[0] - arrow_length * math.cos(angle - arrow_angle))
        y1 = int(end_pt[1] - arrow_length * math.sin(angle - arrow_angle))
        x2 = int(end_pt[0] - arrow_length * math.cos(angle + arrow_angle))
        y2 = int(end_pt[1] - arrow_length * math.sin(angle + arrow_angle))
        cv2.arrowedLine(frame, start_pt, end_pt, color, 2)
        cv2.line(frame, end_pt, (x1, y1), color, 2)
        cv2.line(frame, end_pt, (x2, y2), color, 2)

class ShankasoochiCamera: # Renamed from PersonBagTracker for consistency with app.py
    def __init__(self, source=0, config=None, alert_queue: Optional[queue.Queue] = None):
        self.source = source
        self.config = config if config else {}
        self.cap = None
        self.thread = None
        self.frame = None # Stores the latest processed frame bytes
        self.running = False
        self.lock = threading.Lock() # Lock for accessing self.frame
        self.alert_queue = alert_queue # Queue to send alerts to app.py

        # Initialize tracker parameters from config
        self.model_path = self.config.get('model_path')
        self.screenshot_folder = self.config.get('screenshot_folder')
        self.alert_sound_path = self.config.get('alert_sound_path') # Now only for path check, not playing
        self.tracker_type = self.config.get('tracker', 'bytetrack.yaml')
        self.confidence_threshold = self.config.get('conf', 0.25)
        self.iou_threshold = self.config.get('iou', 0.7)
        self.agnostic_nms = self.config.get('agnostic_nms', False)
        self.max_association_distance = self.config.get('max_assoc_dist', 150)
        self.holding_margin = self.config.get('holding_margin', 30)
        self.sticky_frames = self.config.get('sticky_frames', 15)
        self.alert_threshold = self.config.get('alert_frames', 10) # Frames for unattended alert

        # Load YOLO model
        if not os.path.exists(self.model_path):
            logger.error(f"YOLO model not found at specified path: {self.model_path}")
            raise FileNotFoundError(f"YOLO model '{self.model_path}' not found. Please ensure it's in the correct directory.")
        self.model = YOLO(self.model_path)
        logger.info(f"YOLO model loaded from: {self.model_path}")

        # Initialize tracking components
        self.trajectory_tracker = TrajectoryTracker()
        self.person_class_id = 0 # YOLO class ID for 'person'
        self.bag_class_ids = {24: 'handbag', 26: 'backpack', 28: 'suitcase'} # YOLO class IDs for bags
        
        # State variables for tracking logic
        self.active_persons: Set[int] = set()
        self.active_bags: Set[int] = set()
        self.frame_count = 0
        self.bag_id_mapping: Dict[int, int] = {}
        self.bag_last_person_id: Dict[int, int] = {}
        self.bag_association_frames: Dict[int, int] = defaultdict(int)
        self.unattended_bags: Dict[int, int] = {} # Stores {bag_id: frame_started_being_unattended}
        self.alerts_logged: Set[int] = set() # To prevent repeated alerts for the same bag

        # Ensure screenshot folder exists (app.py also ensures this, but good to have here)
        os.makedirs(self.screenshot_folder, exist_ok=True)
        
        logger.info(f"ShankasoochiCamera initialized with source: {self.source}")
        logger.info(f"Shankasoochi Config: {self.config}")
        logger.info(f"Model Path: {self.model_path}")
        logger.info(f"Screenshot Folder: {self.screenshot_folder}")
        logger.info(f"Alert Sound Path (for check): {self.alert_sound_path}")
        logger.info(f"Tracker initialized with: conf={self.confidence_threshold}, iou={self.iou_threshold}, tracker={self.tracker_type}")
        logger.info(f"Association parameters: max_dist={self.max_association_distance}, holding_margin={self.holding_margin}, sticky_frames={self.sticky_frames}, alert_frames={self.alert_threshold}")

    def start(self):
        """Starts the video capture and processing thread."""
        if self.running:
            logger.info("ShankasoochiCamera is already running.")
            return

        logger.info(f"Starting ShankasoochiCamera from source: {self.source}")
        # Convert source to int if it's a digit string for OpenCV
        try:
            cap_source = int(self.source) if str(self.source).isdigit() else self.source
            self.cap = cv2.VideoCapture(cap_source)
        except ValueError:
            logger.error(f"Invalid camera source: {self.source}. Must be an integer or valid video path.")
            self.running = False
            return

        if not self.cap.isOpened():
            logger.error(f"Error: Could not open video source {self.source}")
            self.running = False
            return

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        logger.info("ShankasoochiCamera processing thread started.")

    def _update(self):
        """Internal method to continuously read frames and process them."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"Failed to read frame from source {self.source}. Attempting to re-open or stopping.")
                self.cap.release()
                try: # Attempt to re-open if it's a file or camera that might recover
                    cap_source = int(self.source) if str(self.source).isdigit() else self.source
                    self.cap = cv2.VideoCapture(cap_source)
                    if not self.cap.isOpened():
                        logger.error(f"Failed to re-open video source {self.source}. Stopping ShankasoochiCamera.")
                        self.running = False
                except Exception as e:
                    logger.error(f"Error during re-opening video source: {e}. Stopping ShankasoochiCamera.")
                    self.running = False
                if not self.running: # If still not running after re-attempt
                    break
                time.sleep(1) # Give some time before next read attempt
                continue

            # Process the frame and get alert info
            processed_frame, alert_info = self._process_single_frame(frame)
            
            # Convert processed frame to JPEG bytes
            ret, jpeg = cv2.imencode('.jpg', processed_frame)
            if not ret:
                logger.error("Failed to encode processed frame to JPEG.")
                continue

            with self.lock:
                self.frame = jpeg.tobytes() # Store for Flask video feed

            # If an alert was triggered, put it in the queue for app.py
            if alert_info and self.alert_queue:
                self.alert_queue.put(alert_info)
                logger.info(f"Alert put into queue: {alert_info.get('message')}")

            time.sleep(0.01) # Simulate processing time / control frame rate

        self.cap.release()
        self.model = None # Release model resources
        logger.info(f"ShankasoochiCamera processing thread stopped for source: {self.source}")

    def get_frame(self) -> Optional[bytes]:
        """Returns the latest processed frame as JPEG bytes."""
        with self.lock:
            return self.frame

    def stop(self):
        """Stops the video capture and processing thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5) # Wait for thread to finish
            if self.thread.is_alive():
                logger.warning("ShankasoochiCamera thread did not terminate gracefully.")
        logger.info("ShankasoochiCamera stop command issued.")

    def _process_single_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Processes a single frame for detection, tracking, and unattended bag alerts.
        Returns the annotated frame and alert information if an alert is triggered.
        """
        self.frame_count += 1

        detections = self.detect_and_track(frame)
        self.update_associations(detections)
        
        # Check for unattended bags and get alert info
        alert_info = self.check_unattended_bags(frame.copy()) # Pass a copy for screenshot
        
        # Draw visualizations
        self.trajectory_tracker.draw_trails(frame)
        self.draw_detections(frame, detections)
        self.draw_associations(frame, detections)
        self.draw_info_panel(frame)

        return frame, alert_info

    # --- Methods from original PersonBagTracker (kept mostly as-is) ---
    @staticmethod
    def calculate_distance(center1: Tuple[int, int], center2: Tuple[int, int]) -> float:
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    @staticmethod
    def is_bag_held_by_person(person_bbox, bag_bbox, margin=30):
        px1, py1, px2, py2 = person_bbox
        bx1, by1, bx2, by2 = bag_bbox
        bag_center_x = (bx1 + bx2) // 2
        bag_center_y = (by1 + by2) // 2
        
        is_within = (px1 - margin <= bag_center_x <= px2 + margin and
                     py1 - margin <= bag_center_y <= py2 + margin)
        return is_within

    def detect_and_track(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.track(
            frame, 
            persist=True, 
            verbose=False,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            tracker=self.tracker_type,
            agnostic_nms=self.agnostic_nms
        )
        detections = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            current_active_persons = set()
            current_active_bags = set()

            for box, track_id, class_id, conf in zip(boxes, ids, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                class_name = self.model.names[class_id]

                if class_id == self.person_class_id:
                    detection = Detection(
                        track_id=track_id, class_id=class_id, class_name=class_name,
                        bbox=(x1, y1, x2, y2), confidence=conf, center=center,
                        assigned_person_id=track_id
                    )
                    detections.append(detection)
                    self.trajectory_tracker.update(track_id, center)
                    current_active_persons.add(track_id)
                elif class_id in self.bag_class_ids:
                    detection = Detection(
                        track_id=track_id, class_id=class_id, class_name=class_name,
                        bbox=(x1, y1, x2, y2), confidence=conf, center=center,
                        assigned_person_id=None
                    )
                    detections.append(detection)
                    self.trajectory_tracker.update(track_id, center)
                    current_active_bags.add(track_id)

            self.active_persons = current_active_persons
            self.active_bags = current_active_bags

        return detections

    def update_associations(self, detections: List[Detection]):
        persons_in_frame = {d.track_id: d for d in detections if d.class_id == self.person_class_id}
        bags_in_frame = {d.track_id: d for d in detections if d.class_id in self.bag_class_ids}

        new_bag_id_mapping = {} 
        
        for bag_id, bag_det in bags_in_frame.items():
            best_person_id = None
            best_dist = float('inf')
            is_currently_held = False 

            for person_id, person_det in persons_in_frame.items():
                if self.is_bag_held_by_person(person_det.bbox, bag_det.bbox, self.holding_margin):
                    is_currently_held = True
                    best_person_id = person_id
                    best_dist = self.calculate_distance(person_det.center, bag_det.center)
                    break 

            if not is_currently_held:
                for person_id, person_det in persons_in_frame.items():
                    dist = self.calculate_distance(person_det.center, bag_det.center)
                    if dist < self.max_association_distance and dist < best_dist:
                        best_person_id = person_id
                        best_dist = dist
            
            if best_person_id is not None:
                new_bag_id_mapping[bag_id] = best_person_id
                self.bag_last_person_id[bag_id] = best_person_id 
                
                self.bag_association_frames[bag_id] = self.sticky_frames * 2 if is_currently_held else self.sticky_frames
            else:
                self.bag_association_frames[bag_id] -= 1
                if self.bag_association_frames[bag_id] <= 0:
                    new_bag_id_mapping[bag_id] = bag_id 
                else:
                    if bag_id in self.bag_id_mapping:
                            new_bag_id_mapping[bag_id] = self.bag_id_mapping[bag_id]
                    else: 
                        new_bag_id_mapping[bag_id] = bag_id

        bags_to_remove = set(self.bag_id_mapping.keys()) - set(bags_in_frame.keys())
        for bag_id in bags_to_remove:
            if bag_id in self.bag_id_mapping: del self.bag_id_mapping[bag_id]
            if bag_id in self.bag_association_frames: del self.bag_association_frames[bag_id]
            if bag_id in self.bag_last_person_id: del self.bag_last_person_id[bag_id]

        self.bag_id_mapping.update(new_bag_id_mapping)

        for det in detections:
            if det.class_id in self.bag_class_ids:
                det.assigned_person_id = self.bag_id_mapping.get(det.track_id, det.track_id) 

    def save_screenshot(self, frame: np.ndarray, bag_id: int, person_id: int) -> Optional[str]:
        """
        Saves a screenshot and returns its relative URL.
        """
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"unattended_bag_alert_bag_{bag_id}_owner_{person_id}_{timestamp}.png"
            full_path = os.path.join(self.screenshot_folder, filename)
            if cv2.imwrite(full_path, frame):
                logger.info(f"Screenshot saved: {full_path}")
                # Return URL relative to static folder
                return f'/static/screenshots/{filename}'
            else:
                logger.error(f"Failed to save screenshot: {full_path}. Check path and permissions, or image format support.")
                return None
        except Exception as e:
            logger.exception(f"Error saving screenshot for bag {bag_id}: {e}")
            return None

    def check_unattended_bags(self, current_frame_for_screenshot: np.ndarray) -> Optional[Dict]:
        """
        Checks for unattended bags and returns alert information if triggered.
        """
        alert_info = None
        for bag_id in self.active_bags:
            assigned_person_id = self.bag_id_mapping.get(bag_id, bag_id) 
            
            if assigned_person_id == bag_id or assigned_person_id not in self.active_persons:
                # Bag is currently unattended or its assigned person is out of frame
                if bag_id not in self.unattended_bags:
                    self.unattended_bags[bag_id] = self.frame_count # Mark start of unattended period
                
                frames_unattended = self.frame_count - self.unattended_bags[bag_id]
                
                if frames_unattended >= self.alert_threshold and bag_id not in self.alerts_logged:
                    owner_id_for_alert = self.bag_last_person_id.get(bag_id, "UNKNOWN")
                    
                    screenshot_url = self.save_screenshot(current_frame_for_screenshot, bag_id, owner_id_for_alert)
                    
                    alert_info = {
                        "type": "Unattended Luggage",
                        "description": f"Bag ID {bag_id} left unattended by owner {owner_id_for_alert}.",
                        "severity": "High", # Or determine dynamically
                        "timestamp": datetime.now().isoformat(),
                        "screenshot_url": screenshot_url
                    }
                    logger.warning(f"🚨 ALERT: {alert_info['description']}")
                    self.alerts_logged.add(bag_id) # Mark this bag as alerted
                    return alert_info # Return the first alert encountered in this frame
            else:
                # Bag is now associated with an active person
                if bag_id in self.unattended_bags:
                    logger.info(f"Frame {self.frame_count}: Bag {bag_id} is no longer unattended (re-associated with {assigned_person_id}).")
                    del self.unattended_bags[bag_id]
                    if bag_id in self.alerts_logged:
                        self.alerts_logged.remove(bag_id)

        # Clean up unattended_bags list for bags that have left the frame
        for bag_id in list(self.unattended_bags.keys()): 
            if bag_id not in self.active_bags: 
                logger.info(f"Frame {self.frame_count}: Bag {bag_id} left frame, removing from unattended list.")
                del self.unattended_bags[bag_id]
                if bag_id in self.alerts_logged:
                    self.alerts_logged.remove(bag_id)
        return alert_info

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]):
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            color = (200, 200, 200) 
            display_id = detection.track_id 

            if detection.class_id == self.person_class_id:
                color = (0, 255, 0) 
                display_id = detection.track_id 
            elif detection.class_id in self.bag_class_ids:
                display_id = detection.assigned_person_id 
                if detection.track_id in self.unattended_bags:
                    color = (0, 0, 255) # Red for unattended
                elif detection.assigned_person_id is not None and detection.assigned_person_id != detection.track_id:
                    color = (255, 0, 0) # Blue for associated
                else:
                    color = (0, 255, 255) # Yellow for unassociated bag

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label_parts = [detection.class_name]
            if detection.class_id == self.person_class_id:
                label_parts.append(f"ID:{display_id}")
            elif detection.class_id in self.bag_class_ids:
                label_parts.append(f"Owner:{display_id if display_id != detection.track_id else 'None'}") 
                if detection.track_id in self.unattended_bags:
                    frames_unattended = self.frame_count - self.unattended_bags[detection.track_id]
                    label_parts.append(f"UNATTENDED:{frames_unattended}")
            label_parts.append(f"(YOLO:{detection.track_id})") 

            label = " ".join(label_parts)

            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def draw_associations(self, frame: np.ndarray, detections: List[Detection]):
        person_detections = {d.track_id: d for d in detections if d.class_id == self.person_class_id}
        bag_detections = {d.track_id: d for d in detections if d.class_id in self.bag_class_ids}
        
        for bag_id, assigned_person_id in self.bag_id_mapping.items():
            if bag_id in bag_detections and assigned_person_id in person_detections and bag_id != assigned_person_id:
                person_center = person_detections[assigned_person_id].center
                bag_center = bag_detections[bag_id].center
                cv2.line(frame, person_center, bag_center, (255, 255, 0), 2) 
                cv2.circle(frame, person_center, 8, (255, 255, 0), -1)
                cv2.circle(frame, bag_center, 8, (255, 255, 0), -1)

    def draw_info_panel(self, frame: np.ndarray):
        info_y = 30
        line_height = 25 

        cv2.putText(frame, f"Frame: {self.frame_count}", (10, info_y),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += line_height
        cv2.putText(frame, f"Active Persons: {len(self.active_persons)}", (10, info_y),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
        info_y += line_height
        cv2.putText(frame, f"Active Bags (YOLO ID): {len(self.active_bags)}", (10, info_y),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += line_height
        cv2.putText(frame, f"Bags with Assigned IDs: {len([bid for bid, pid in self.bag_id_mapping.items() if bid != pid])}", (10, info_y),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) 
        info_y += line_height
        cv2.putText(frame, f"Unattended Bags (YOLO ID): {len(self.unattended_bags)}", (10, info_y),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
        
        if self.unattended_bags:
            info_y += line_height
            cv2.putText(frame, "🚨 ALERT: Unattended luggage detected!", (10, info_y),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)