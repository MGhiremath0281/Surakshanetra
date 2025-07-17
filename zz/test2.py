import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque, defaultdict
import logging
import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Represents a single detection with tracking information"""
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center: Tuple[int, int]

@dataclass
class PersonBagPair:
    """Represents a person-bag pairing"""
    person_id: int
    bag_ids: Set[int]
    last_seen_together: int  # frame number
    association_strength: float
    assigned_bag_id: Optional[int] = None  # The bag gets same ID as person

class TrajectoryTracker:
    """Manages trajectory visualization for objects"""
    def __init__(self, max_trail_length=30):
        self.trails = defaultdict(lambda: deque(maxlen=max_trail_length))
        self.colors = {}
        
    def update(self, track_id: int, center: Tuple[int, int]):
        """Update trajectory for a tracked object"""
        self.trails[track_id].append(center)
        if track_id not in self.colors:
            # Generate consistent color for each ID
            np.random.seed(track_id)
            self.colors[track_id] = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
    
    def draw_trails(self, frame: np.ndarray):
        """Draw trajectory trails on the frame"""
        for track_id, trail in self.trails.items():
            if len(trail) > 1:
                color = self.colors[track_id]
                points = np.array(trail, dtype=np.int32)
                
                # Draw trail with decreasing opacity
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    thickness = max(1, int(3 * alpha))
                    cv2.line(frame, tuple(points[i-1]), tuple(points[i]), color, thickness)
                
                # Draw arrow at the end
                if len(points) >= 2:
                    self.draw_arrow(frame, points[-2], points[-1], color)
    
    def draw_arrow(self, frame: np.ndarray, start: np.ndarray, end: np.ndarray, color: tuple):
        """Draw arrow indicating direction of movement"""
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        arrow_length = 10
        arrow_angle = math.pi / 6
        
        # Calculate arrow head points
        x1 = int(end[0] - arrow_length * math.cos(angle - arrow_angle))
        y1 = int(end[1] - arrow_length * math.sin(angle - arrow_angle))
        x2 = int(end[0] - arrow_length * math.cos(angle + arrow_angle))
        y2 = int(end[1] - arrow_length * math.sin(angle + arrow_angle))
        
        cv2.arrowedLine(frame, tuple(start), tuple(end), color, 2)
        cv2.line(frame, tuple(end), (x1, y1), color, 2)
        cv2.line(frame, tuple(end), (x2, y2), color, 2)

class PersonBagTracker:
    """Main class for person-bag tracking and unattended luggage detection"""
    
    def __init__(self, model_path='yolov8x.pt'):
        self.model = YOLO(model_path)
        self.trajectory_tracker = TrajectoryTracker()
        
        # COCO class IDs for persons and bags
        self.person_class_id = 0
        self.bag_class_ids = {24: 'handbag', 26: 'backpack', 28: 'suitcase'}
        
        # Tracking data
        self.person_bag_pairs: Dict[int, PersonBagPair] = {}
        self.active_persons: Set[int] = set()
        self.active_bags: Set[int] = set()
        self.bag_id_mapping: Dict[int, int] = {}  # original_bag_id -> person_id
        self.person_last_positions: Dict[int, Tuple[int, int]] = {}  # person_id -> last_position
        self.frame_count = 0
        
        # Alert system - only for bags whose owners left the frame
        self.unattended_bags: Dict[int, int] = {}  # person_id -> frame_person_left
        self.alert_threshold = 30  # frames before alert
        self.alerts_logged: Set[int] = set()
        
        # Distance thresholds for association
        self.max_association_distance = 150  # pixels
        self.min_association_frames = 5
        
        logger.info("PersonBagTracker initialized with YOLOv8x model")
    
    def calculate_distance(self, center1: Tuple[int, int], center2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two centers"""
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def detect_and_track(self, frame: np.ndarray) -> List[Detection]:
        """Perform detection and tracking on a single frame"""
        results = self.model.track(frame, persist=True, verbose=False)
        detections = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, class_id, conf in zip(boxes, ids, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                class_name = self.model.names[class_id]
                
                # Only process persons and bags
                if class_id == self.person_class_id or class_id in self.bag_class_ids:
                    detection = Detection(
                        track_id=track_id,
                        class_id=class_id,
                        class_name=class_name,
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        center=center
                    )
                    detections.append(detection)
                    
                    # Update trajectory
                    self.trajectory_tracker.update(track_id, center)
        
        return detections
    
    def update_associations(self, detections: List[Detection]):
        """Update person-bag associations and assign same IDs"""
        persons = [d for d in detections if d.class_id == self.person_class_id]
        bags = [d for d in detections if d.class_id in self.bag_class_ids]
        
        # Update active sets and person positions
        self.active_persons = {p.track_id for p in persons}
        self.active_bags = {b.track_id for b in bags}
        
        # Update person positions
        for person in persons:
            self.person_last_positions[person.track_id] = person.center
        
        # Find associations based on proximity
        for person in persons:
            closest_bag = None
            min_distance = float('inf')
            
            # Find the closest bag to this person
            for bag in bags:
                distance = self.calculate_distance(person.center, bag.center)
                if distance < self.max_association_distance and distance < min_distance:
                    min_distance = distance
                    closest_bag = bag
            
            # Create or update person-bag pair
            if person.track_id not in self.person_bag_pairs:
                self.person_bag_pairs[person.track_id] = PersonBagPair(
                    person_id=person.track_id,
                    bag_ids=set(),
                    last_seen_together=self.frame_count,
                    association_strength=0.0,
                    assigned_bag_id=None
                )
            
            pair = self.person_bag_pairs[person.track_id]
            
            # If we found a close bag, associate it
            if closest_bag:
                # Map the bag's original ID to the person's ID
                self.bag_id_mapping[closest_bag.track_id] = person.track_id
                pair.bag_ids.add(closest_bag.track_id)
                pair.assigned_bag_id = closest_bag.track_id
                pair.last_seen_together = self.frame_count
                pair.association_strength = min(1.0, pair.association_strength + 0.1)
                
                # Update trajectory with person's ID for the bag
                self.trajectory_tracker.update(person.track_id, closest_bag.center)
            
            # Update person trajectory
            self.trajectory_tracker.update(person.track_id, person.center)
    
    def check_unattended_bags(self, detections: List[Detection]):
        """Check for unattended bags and trigger alerts only when person exits frame"""
        current_bag_ids = {d.track_id for d in detections if d.class_id in self.bag_class_ids}
        
        # Check each person-bag pair
        for person_id, pair in list(self.person_bag_pairs.items()):
            # Check if person is no longer in frame
            if person_id not in self.active_persons:
                # Person has left the frame - check if their bag is still present
                if pair.assigned_bag_id and pair.assigned_bag_id in current_bag_ids:
                    # Person left but their bag is still in frame
                    if person_id not in self.unattended_bags:
                        self.unattended_bags[person_id] = self.frame_count
                        logger.info(f"Person {person_id} left the frame, bag {pair.assigned_bag_id} still present")
                    
                    # Check if bag has been unattended long enough
                    frames_unattended = self.frame_count - self.unattended_bags[person_id]
                    if frames_unattended >= self.alert_threshold and person_id not in self.alerts_logged:
                        self.trigger_alert(person_id, pair.assigned_bag_id)
                        self.alerts_logged.add(person_id)
                else:
                    # Person left and bag is also gone - remove from tracking
                    if person_id in self.unattended_bags:
                        del self.unattended_bags[person_id]
                    if person_id in self.alerts_logged:
                        self.alerts_logged.remove(person_id)
                    # Keep the pair for a few more frames in case person returns
                    if self.frame_count - pair.last_seen_together > 100:
                        del self.person_bag_pairs[person_id]
            else:
                # Person is still in frame - remove from unattended list
                if person_id in self.unattended_bags:
                    del self.unattended_bags[person_id]
                    logger.info(f"Person {person_id} returned to their bag")
                if person_id in self.alerts_logged:
                    self.alerts_logged.remove(person_id)
    
    def trigger_alert(self, person_id: int, bag_id: int):
        """Trigger alert for unattended bag"""
        alert_msg = f"🚨 ALERT: Person {person_id} left their bag {bag_id} unattended and exited the frame!"
        logger.warning(alert_msg)
        print(alert_msg)
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]):
        """Draw bounding boxes and information on frame"""
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Determine display ID and color
            if detection.class_id == self.person_class_id:
                display_id = detection.track_id
                color = (0, 255, 0)  # Green for persons
                # Check if this person has unattended bag
                if detection.track_id in self.unattended_bags:
                    color = (0, 165, 255)  # Orange if person left bag
            else:
                # For bags, use person's ID if mapped, otherwise original ID
                if detection.track_id in self.bag_id_mapping:
                    display_id = self.bag_id_mapping[detection.track_id]
                    person_id = self.bag_id_mapping[detection.track_id]
                    if person_id in self.unattended_bags:
                        color = (0, 0, 255)  # Red for unattended bags
                    else:
                        color = (255, 0, 0)  # Blue for attended bags
                else:
                    display_id = detection.track_id
                    color = (128, 128, 128)  # Gray for unmapped bags
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{detection.class_name} ID:{display_id}"
            
            # Add unattended status for bags
            if detection.class_id in self.bag_class_ids and detection.track_id in self.bag_id_mapping:
                person_id = self.bag_id_mapping[detection.track_id]
                if person_id in self.unattended_bags:
                    frames_unattended = self.frame_count - self.unattended_bags[person_id]
                    label += f" UNATTENDED:{frames_unattended}f"
            
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def draw_associations(self, frame: np.ndarray, detections: List[Detection]):
        """Draw lines showing person-bag associations"""
        person_detections = {d.track_id: d for d in detections if d.class_id == self.person_class_id}
        bag_detections = {d.track_id: d for d in detections if d.class_id in self.bag_class_ids}
        
        for person_id, pair in self.person_bag_pairs.items():
            if person_id in person_detections and pair.assigned_bag_id:
                person_detection = person_detections[person_id]
                
                # Find the bag detection
                bag_detection = None
                for bag_id, bag_det in bag_detections.items():
                    if bag_id == pair.assigned_bag_id:
                        bag_detection = bag_det
                        break
                
                if bag_detection:
                    person_center = person_detection.center
                    bag_center = bag_detection.center
                    
                    # Choose line color based on status
                    if person_id in self.unattended_bags:
                        line_color = (0, 0, 255)  # Red for unattended
                    else:
                        line_color = (255, 255, 0)  # Yellow for normal association
                    
                    # Draw association line
                    cv2.line(frame, person_center, bag_center, line_color, 2)
                    # Draw association indicators
                    cv2.circle(frame, person_center, 8, line_color, -1)
                    cv2.circle(frame, bag_center, 8, line_color, -1)
    
    def draw_info_panel(self, frame: np.ndarray):
        """Draw information panel with statistics"""
        info_y = 30
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        info_y += 30
        cv2.putText(frame, f"Active Persons: {len(self.active_persons)}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        info_y += 30
        cv2.putText(frame, f"Active Bags: {len(self.active_bags)}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        info_y += 30
        cv2.putText(frame, f"Unattended Bags: {len(self.unattended_bags)}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if self.unattended_bags:
            info_y += 30
            cv2.putText(frame, "🚨 ALERT: Unattended luggage detected!", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def process_video(self, input_source, output_path=None, display=True):
        """Process video from file or camera"""
        # Open video source
        if isinstance(input_source, str) and input_source.isdigit():
            cap = cv2.VideoCapture(int(input_source))
        else:
            cap = cv2.VideoCapture(input_source)
        
        if not cap.isOpened():
            logger.error(f"Error opening video source: {input_source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Processing video: {width}x{height} @ {fps} FPS")
        
        # Initialize video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Detect and track objects
                detections = self.detect_and_track(frame)
                
                # Update associations
                self.update_associations(detections)
                
                # Check for unattended bags
                self.check_unattended_bags(detections)
                
                # Draw visualizations
                self.trajectory_tracker.draw_trails(frame)
                self.draw_detections(frame, detections)
                self.draw_associations(frame, detections)
                self.draw_info_panel(frame)
                
                # Write frame to output video
                if out:
                    out.write(frame)
                
                # Display frame
                if display:
                    cv2.imshow('Person-Bag Tracking', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # Log progress every 100 frames
                if self.frame_count % 100 == 0:
                    logger.info(f"Processed {self.frame_count} frames")
        
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            logger.info(f"Video processing completed. Total frames: {self.frame_count}")

def main():
    """Main function to run the tracker"""
    parser = argparse.ArgumentParser(description='Person-Bag Tracking with Unattended Luggage Detection')
    parser.add_argument('--input', type=str, default='0', 
                       help='Input video file path or camera index (default: 0 for webcam)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file path (optional)')
    parser.add_argument('--model', type=str, default='yolov8x.pt',
                       help='YOLOv8 model path (default: yolov8x.pt)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable real-time display')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = PersonBagTracker(model_path=args.model)
    
    # Process video
    tracker.process_video(
        input_source=args.input,
        output_path=args.output,
        display=not args.no_display
    )

if __name__ == "__main__":
    main()