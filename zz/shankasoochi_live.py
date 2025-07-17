import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque, defaultdict
import logging
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import math
import os
from playsound import playsound 

# --- Configuration ---
ALERT_SOUND_FILENAME = "alert.wav" 
SCREENSHOT_FOLDER_NAME = "screenshots"
DEFAULT_YOLO_MODEL = 'yolov8x.pt' 

# Configure logging
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

class PersonBagTracker:
    def __init__(self, model_path: str, screenshot_folder: str, alert_sound_path: str,
                 # New tracking parameters
                 tracker_type: str = 'bytetrack.yaml',
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.7,
                 agnostic_nms: bool = False,
                 # Your custom association parameters
                 max_association_distance: int = 150,
                 holding_margin: int = 30,
                 sticky_frames: int = 15
                ):
        
        if not os.path.exists(model_path):
            logger.error(f"YOLO model not found at specified path: {model_path}")
            raise FileNotFoundError(f"YOLO model '{model_path}' not found. Please ensure it's in the correct directory or provide a full path.")
        
        self.model = YOLO(model_path)
        logger.info(f"YOLO model loaded from: {model_path}")

        self.trajectory_tracker = TrajectoryTracker()
        self.person_class_id = 0
        self.bag_class_ids = {24: 'handbag', 26: 'backpack', 28: 'suitcase'}
        
        self.active_persons: Set[int] = set()
        self.active_bags: Set[int] = set()
        self.frame_count = 0
        
        self.bag_id_mapping: Dict[int, int] = {}
        self.bag_last_person_id: Dict[int, int] = {}
        self.bag_association_frames: Dict[int, int] = defaultdict(int)
        self.unattended_bags: Dict[int, int] = {}
        
        self.alert_threshold = 30  # Will be set by args.alert_frames later
        self.alerts_logged: Set[int] = set()
        
        # Store custom association parameters
        self.max_association_distance = max_association_distance
        self.holding_margin = holding_margin
        self.sticky_frames = sticky_frames

        # Store YOLO tracking parameters
        self.tracker_type = tracker_type
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.agnostic_nms = agnostic_nms

        self.screenshot_folder = screenshot_folder
        self._ensure_screenshot_folder_exists()
        
        self.alert_sound_path = alert_sound_path
        if not os.path.exists(self.alert_sound_path):
            logger.warning(f"Alert sound file not found at: {self.alert_sound_path}. Audio alerts will be disabled.")
            self.alert_sound_path = None
        else:
            logger.info(f"Alert sound loaded from: {self.alert_sound_path}")
            
        logger.info(f"Tracker initialized with: conf={self.confidence_threshold}, iou={self.iou_threshold}, tracker={self.tracker_type}")
        logger.info(f"Association parameters: max_dist={self.max_association_distance}, holding_margin={self.holding_margin}, sticky_frames={self.sticky_frames}")


    def _ensure_screenshot_folder_exists(self):
        try:
            if not os.path.exists(self.screenshot_folder):
                os.makedirs(self.screenshot_folder)
                logger.info(f"Created screenshot folder: {self.screenshot_folder}")
        except Exception as e:
            logger.error(f"Failed to create screenshot folder {self.screenshot_folder}: {e}")

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
        # Pass tracking parameters to model.track()
        results = self.model.track(
            frame, 
            persist=True, 
            verbose=False, # Set to True for detailed YOLO tracking output
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

    def save_screenshot(self, frame: np.ndarray, bag_id: int, person_id: int):
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self.screenshot_folder, f"unattended_bag_alert_bag_{bag_id}_owner_{person_id}_{timestamp}.png")
            if cv2.imwrite(filename, frame):
                logger.info(f"Screenshot saved: {filename}")
            else:
                logger.error(f"Failed to save screenshot: {filename}. Check path and permissions, or image format support.")
        except Exception as e:
            logger.exception(f"Error saving screenshot for bag {bag_id}: {e}")

    def play_alert_sound(self):
        if self.alert_sound_path and os.path.exists(self.alert_sound_path):
            try:
                playsound(self.alert_sound_path, block=False)
                logger.info(f"Played alert sound: {self.alert_sound_path}")
            except Exception as e:
                logger.error(f"Failed to play sound {self.alert_sound_path}: {e}. Ensure playsound dependencies are met.")
        else:
            logger.warning("Alert sound path not set or file not found, cannot play sound.")

    def check_unattended_bags(self, current_frame_for_screenshot: np.ndarray):
        
        for bag_id in self.active_bags:
            assigned_person_id = self.bag_id_mapping.get(bag_id, bag_id) 
            
            if assigned_person_id == bag_id or assigned_person_id not in self.active_persons:
                
                if bag_id not in self.unattended_bags:
                    self.unattended_bags[bag_id] = self.frame_count
                
                frames_unattended = self.frame_count - self.unattended_bags[bag_id]
                
                if frames_unattended >= self.alert_threshold and bag_id not in self.alerts_logged:
                    owner_id_for_alert = self.bag_last_person_id.get(bag_id, "UNKNOWN")
                    self.trigger_alert(bag_id, owner_id_for_alert, current_frame_for_screenshot)
                    self.alerts_logged.add(bag_id)
            else:
                if bag_id in self.unattended_bags:
                    logger.info(f"Frame {self.frame_count}: Bag {bag_id} is no longer unattended (re-associated with {assigned_person_id}).")
                    del self.unattended_bags[bag_id]
                    if bag_id in self.alerts_logged:
                        self.alerts_logged.remove(bag_id)

        for bag_id in list(self.unattended_bags.keys()): 
            if bag_id not in self.active_bags: 
                logger.info(f"Frame {self.frame_count}: Bag {bag_id} left frame, removing from unattended list.")
                del self.unattended_bags[bag_id]
                if bag_id in self.alerts_logged:
                    self.alerts_logged.remove(bag_id)


    def trigger_alert(self, bag_id: int, person_id: int, frame: np.ndarray):
        alert_msg = f"🚨 ALERT: Unattended bag detected! Bag Original ID: {bag_id}, Last Owner ID: {person_id}. Frame: {self.frame_count}"
        logger.warning(alert_msg)
        print(alert_msg) 
        self.save_screenshot(frame, bag_id, person_id)
        self.play_alert_sound() 

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
                    color = (0, 0, 255) 
                elif detection.assigned_person_id is not None and detection.assigned_person_id != detection.track_id:
                    color = (255, 0, 0) 
                else:
                    color = (0, 255, 255) 

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

    def process_video(self, input_source, output_path=None, display=True, playback_speed=1.0):
        if isinstance(input_source, str) and input_source.isdigit():
            cap = cv2.VideoCapture(int(input_source))
        else:
            cap = cv2.VideoCapture(input_source)

        if not cap.isOpened():
            logger.error(f"Error opening video source: {input_source}")
            return

        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Processing video: {width}x{height} @ {original_fps} FPS. Playback speed: {playback_speed}x")

        delay_ms = max(1, int(1000 / (original_fps * playback_speed)))
        logger.info(f"Calculated cv2.waitKey delay: {delay_ms} ms (for {playback_speed}x playback)")

        out = None
        if output_path:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                out = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))
                if not out.isOpened():
                    logger.error(f"Error creating video writer for {output_path}. Check codec or file path. Output video will not be saved.")
                    out = None 
                else:
                    logger.info(f"Output video writer created: {output_path}")
            except Exception as e:
                logger.exception(f"Exception creating video writer: {e}. Output video will not be saved.")
                out = None

        start_time = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream or error reading frame.")
                    break

                self.frame_count += 1

                detections = self.detect_and_track(frame)
                
                self.update_associations(detections)
                
                self.check_unattended_bags(frame.copy()) 

                self.trajectory_tracker.draw_trails(frame)
                self.draw_detections(frame, detections)
                self.draw_associations(frame, detections)
                self.draw_info_panel(frame)

                if out:
                    out.write(frame)

                if display:
                    cv2.imshow('Person-Bag Tracking', frame)
                    key = cv2.waitKey(delay_ms) & 0xFF
                    if key == ord('q'):
                        logger.info("User pressed 'q', exiting.")
                        break

                if self.frame_count % 100 == 0:
                    elapsed_time = time.time() - start_time
                    fps_processed = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                    logger.info(f"Processed {self.frame_count} frames. Current processing FPS: {fps_processed:.2f}")

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user (KeyboardInterrupt).")
        except Exception as e:
            logger.exception(f"An unexpected error occurred during video processing: {e}")
        finally:
            logger.info("Releasing resources...")
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            logger.info(f"Video processing completed. Total frames processed: {self.frame_count}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_abs_path = os.path.join(script_dir, DEFAULT_YOLO_MODEL)
    screenshot_abs_folder = os.path.join(script_dir, SCREENSHOT_FOLDER_NAME)
    alert_sound_abs_path = os.path.join(script_dir, ALERT_SOUND_FILENAME)

    parser = argparse.ArgumentParser(description='Person-Bag Tracking with Unattended Luggage Detection')
    parser.add_argument('--input', type=str, default='0',
                        help='Input video file path or camera index (default: 0 for webcam)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video file path (optional)')
    parser.add_argument('--model', type=str, default=model_abs_path,
                        help=f'YOLOv8 model path (default: {DEFAULT_YOLO_MODEL} in script directory)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable real-time display')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier for display (e.g., 2.0 for 2x speed). Does not affect output video FPS.')
    parser.add_argument('--screenshot-folder', type=str, default=screenshot_abs_folder,
                        help=f'Folder to save screenshots of unattended bag alerts (default: {SCREENSHOT_FOLDER_NAME} in script directory).')
    parser.add_argument('--alert-frames', type=int, default=10,
                        help='Number of frames a bag must be unattended to trigger an alert and sound.')
    parser.add_argument('--alert-sound', type=str, default=alert_sound_abs_path,
                        help=f'Path to the WAV file for alert sound (default: {ALERT_SOUND_FILENAME} in script directory).')
    
    # New arguments for YOLOv8 tracking parameters
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', choices=['bytetrack.yaml', 'botsort.yaml'],
                        help='Which YOLOv8 tracker to use (bytetrack.yaml or botsort.yaml).')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for object detections.')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='IoU threshold for Non-Maximum Suppression (NMS).')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='Perform class-agnostic NMS (can sometimes help with tracking).')

    # New arguments for custom association parameters
    parser.add_argument('--max-assoc-dist', type=int, default=150,
                        help='Max pixel distance for a bag to be associated with a person.')
    parser.add_argument('--holding-margin', type=int, default=30,
                        help='Pixel margin around person bbox for "holding" detection.')
    parser.add_argument('--sticky-frames', type=int, default=15,
                        help='Number of frames to keep association after direct conditions are lost.')

    args = parser.parse_args()

    tracker = PersonBagTracker(
        model_path=args.model, 
        screenshot_folder=args.screenshot_folder,
        alert_sound_path=args.alert_sound,
        # Pass new tracking parameters
        tracker_type=args.tracker,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        agnostic_nms=args.agnostic_nms,
        # Pass new association parameters
        max_association_distance=args.max_assoc_dist,
        holding_margin=args.holding_margin,
        sticky_frames=args.sticky_frames
    )
    
    tracker.alert_threshold = args.alert_frames

    tracker.process_video(
        input_source=args.input,
        output_path=args.output,
        display=not args.no_display,
        playback_speed=args.speed
    )

if __name__ == "__main__":
    main()