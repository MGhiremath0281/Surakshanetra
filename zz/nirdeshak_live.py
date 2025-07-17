#!/usr/bin/env python3
"""
Ultra-advanced Asset tracker with dynamic circular thresholds,
enhanced logging, immediate (interruptible) male-voice TTS alerts,
and persistent visual tracking trails for assets within the frame.

Keys while running:
  + / =  : grow / shrink the initial *zone* (for general context, not used for Asset-specific misplaced logic now)
  - / _  : shrink initial zone
  w a s d: move threshold zone
  [ ]    : decrease / increase move step
  r      : reset to default zone
  q      : quit
"""

import sys
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import supervision as sv
import time
import threading # For non-blocking TTS and logging
import random # For cool colors
import logging # For logging events
import pyttsx3 # For Text-to-Speech with voice selection

# --------------------------------------------------
# 1. Config
# --------------------------------------------------
SOURCE                   = sys.argv[1] if len(sys.argv) > 1 else 0
MODEL                    = "yolov8x-seg.pt"
BOTTLE_CLS               = 39 # Assuming 'bottle' class ID is still used for detection
CONF                     = 0.25
IOU                      = 0.5
MISSING_FRAMES           = 30            # Number of frames before an asset is declared 'MISSING' (YOLO not detecting)
SETTLE_DOWN_SECONDS      = 5             # Time for Asset to settle before defining its 'home' position
SETTLED_CIRCLE_RADIUS    = 70            # Radius of the circle for 'misplaced' threshold (pixels)
MAX_TRAIL_AGE_SECONDS    = 2.0           # How long (in seconds) to keep trail dots for an asset after last detection
DEFAULT_ZONE             = (200, 150, 600, 400) # (x1, y1, x2, y2) - Still used for OSD
STEP                     = 20            # Pixels per key press for zone adjustment
MASK_ALPHA               = 0.60          # 0-1 for mask opacity
ALERT_COOLDOWN_SECONDS   = 5             # Cooldown for TTS alerts per Asset for the same status change

# --------------------------------------------------
# 2. Init
# --------------------------------------------------
model = YOLO(MODEL)
cap   = cv2.VideoCapture(int(SOURCE) if str(SOURCE).isdigit() else SOURCE)
if not cap.isOpened():
    raise SystemExit("Cannot open video source")

# Annotators
mask_annot  = sv.MaskAnnotator(color=sv.Color.BLUE, opacity=MASK_ALPHA)
label_annot = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_thickness=2)
box_annot   = sv.BoxAnnotator(thickness=2)

track_history = {} # Stores list of (cx, cy, timestamp) for each tid
asset_status = {} # To store 'OK', 'MISPLACED', 'MISSING', 'SETTLING', 'LOST' for each track_id
last_detection_time = {} # To store the last time an asset was *detected* by YOLO
last_alert_time = {} # To store last alert timestamp for cooldown per (tid, status_type) pair
asset_settled_info = {} # {tid: {'initial_pos': (cx, cy), 'radius': r, 'settled_time': timestamp}}
asset_colors = {} # To store unique color for each Asset ID

frame_id = 0

# Variables to control TTS thread
tts_lock = threading.Lock()
tts_event = threading.Event() # To signal the TTS thread to speak
tts_message_queue = deque() # Queue for TTS messages

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("asset_tracking.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# --- TTS Engine Management in Dedicated Thread ---
def tts_speaker_thread_function():
    """Dedicated thread to handle TTS speaking to prevent blocking the main loop.
       Initializes pyttsx3 once and explicitly manages its event loop.
    """
    engine = None
    try:
        engine = pyttsx3.init()
        # Attempt to find a male voice
        voices = engine.getProperty('voices')
        male_voice_found = False
        for voice in voices:
            if "male" in voice.name.lower() or "david" in voice.name.lower() or "alex" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                male_voice_found = True
                logger.info(f"Selected TTS Voice in speaker thread: {voice.name}")
                break
        if not male_voice_found:
            logger.warning("Could not find a specific male voice for TTS, using default system voice.")
        
        engine.setProperty('rate', 180) # Speed of speech (words per minute)
        engine.setProperty('volume', 1.0) # Max volume (0.0 to 1.0)
        
        # Start the engine's event loop in a non-blocking way
        # This allows engine.say() to queue speech and engine.iterate() to process it.
        engine.startLoop(False) 
        logger.info("TTS engine event loop started in speaker thread.")

        while True:
            # Check if there are messages to speak
            if tts_message_queue:
                with tts_lock:
                    # Clear any pending messages to prioritize the newest one
                    while tts_message_queue:
                        message = tts_message_queue.popleft()
                        try:
                            # Stop any current speech before queuing the new one
                            engine.stop() 
                            engine.say(message)
                            logger.debug(f"Queued TTS message: '{message}'")
                            # Give pyttsx3's internal loop a moment to process the 'say' command
                            time.sleep(0.01) 
                        except Exception as e:
                            logger.error(f"Error queuing TTS message '{message}': {e}")
                tts_event.clear() # Clear the event after processing messages

            # This is crucial: allow pyttsx3's internal event loop to process events
            # This makes the sound actually play.
            engine.iterate() 
            
            # Small sleep to prevent busy-waiting and yield control
            time.sleep(0.05) 

    except Exception as e:
        logger.error(f"Critical error in TTS speaker thread: {e}")
    finally:
        if engine:
            try:
                engine.stop()
                # Try to gracefully end the loop. This might still throw "run loop not started"
                # if the loop finished on its own, but it's good practice.
                if engine._inLoop: # Check if the loop is still considered active by pyttsx3
                    engine.endLoop()
                logger.info("TTS engine event loop ended.")
            except Exception as e:
                logger.error(f"Error during final TTS engine cleanup: {e}")

# Start the dedicated TTS speaker thread (daemon=True ensures it exits with the main program)
speaker_thread = threading.Thread(target=tts_speaker_thread_function, daemon=True)
speaker_thread.start()


# --------------------------------------------------
# 3. Helpers
# --------------------------------------------------
def centroid(xyxy):
    x1, y1, x2, y2 = xyxy
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def clamp(val, low, high):
    return max(low, min(val, high))

def get_random_cool_color():
    """Generates a random, visually 'cool' (blue/green/purple range) color.
       HSV Hue is scaled 0-179 for OpenCV's uint8.
    """
    h = random.randint(90, 150) # Adjusted Hue range for OpenCV (0-179 scale)
    s = random.randint(150, 255) # Saturation (0-255)
    v = random.randint(180, 255) # Value/Brightness (0-255)

    hsv_color = np.uint8([[[h, s, v]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
    return sv.Color(bgr_color[2], bgr_color[1], bgr_color[0])

def play_tts_alert(message):
    """Adds a message to the TTS queue and signals the speaker thread.
       If a message is already in the queue, it's cleared to prioritize the new one.
    """
    # We no longer check for 'global_tts_engine' here as the engine is managed by the thread itself.
    # The thread will log an error if its engine fails to initialize.
    with tts_lock: # Ensure thread-safe access to the queue
        # Clear any pending messages to prioritize the new one (if any)
        if tts_message_queue: 
            tts_message_queue.clear() 
        tts_message_queue.append(message)
    tts_event.set() # Signal the speaker thread there's a message

def draw_osd(img, rect, step, ok_count, misplaced_count, missing_count):
    x1, y1, x2, y2 = rect
    
    cv2.putText(img, f"Legacy Zone: ({x1},{y1})-({x2},{y2})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, f"Step: {step}px  [+/-] grow/shrink  wasd move  r reset  [ ] step",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"OK Assets: {ok_count}  Misplaced Assets: {misplaced_count}  Missing Assets: {missing_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Cyan for counts
    cv2.putText(img, "q quit", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# --------------------------------------------------
# 4. Dynamic threshold zone (for general context, not drawn now)
# --------------------------------------------------
zone = list(DEFAULT_ZONE)   # mutable copy

# --------------------------------------------------
# 5. Main Loop
# --------------------------------------------------
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_id += 1
    h, w = frame.shape[:2]
    current_time = time.time() # Update current_time once per frame

    # ---- Handle keys (non-blocking) ----
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in (ord('+'), ord('=')):
        zone[2] = clamp(zone[2] + STEP, zone[0] + STEP, w)
        zone[3] = clamp(zone[3] + STEP, zone[1] + STEP, h)
    elif key in (ord('-'), ord('_')):
        zone[2] = clamp(zone[2] - STEP, zone[0] + STEP, w)
        zone[3] = clamp(zone[3] - STEP, zone[1] + STEP, h)
    elif key == ord('w'):
        zone[1] = clamp(zone[1] - STEP, 0, h - 1)
        zone[3] = clamp(zone[3] - STEP, zone[1] + STEP, h)
    elif key == ord('s'):
        zone[1] = clamp(zone[1] + STEP, 0, h - 1)
        zone[3] = clamp(zone[3] + STEP, zone[1] + STEP, h)
    elif key == ord('a'):
        zone[0] = clamp(zone[0] - STEP, 0, w - 1)
        zone[2] = clamp(zone[2] - STEP, zone[0] + STEP, w)
    elif key == ord('d'):
        zone[0] = clamp(zone[0] + STEP, 0, w - 1)
        zone[2] = clamp(zone[2] + STEP, zone[0] + STEP, w)
    elif key == ord('r'):
        zone = list(DEFAULT_ZONE)
    elif key == ord('['):
        STEP = max(5, STEP - 5)
    elif key == ord(']'):
        STEP = min(100, STEP + 5)

    # ---- Inference ----
    results = model.track(source=frame, persist=True, classes=[BOTTLE_CLS],
                          conf=CONF, iou=IOU, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated = frame.copy()
    labels = []
    active_ids = set()
    ok_count = 0
    misplaced_count = 0
    
    # Track the actual count of currently missing assets for OSD
    current_missing_assets_count = 0 

    # ---- Update detection times and collect active IDs ----
    if detections.tracker_id is not None:
        for tid in detections.tracker_id:
            active_ids.add(tid)
            last_detection_time[tid] = current_time

    # ---- Process each track (including those not currently detected but still in history) ----
    # Create a combined set of all known track IDs (those detected now or with recent history)
    all_known_tids = set(track_history.keys()) | active_ids

    for tid in sorted(list(all_known_tids)): # Sort for consistent drawing order
        cx, cy = None, None
        box = None
        current_detection_exists = tid in active_ids

        # Determine current position (from detection or last known history)
        if current_detection_exists:
            idx = np.where(detections.tracker_id == tid)[0][0]
            box = detections.xyxy[idx]
            cx, cy = centroid(box)
            # Add current point to history
            if tid not in track_history:
                track_history[tid] = [] # Use list for non-fixed length
            track_history[tid].append((cx, cy, current_time))
            
            # Assign color if new ID
            if tid not in asset_colors:
                asset_colors[tid] = get_random_cool_color()

        else: # Not currently detected by YOLO
            if track_history.get(tid):
                # Use the last known position from history for display/status checks
                last_pos_in_history = track_history[tid][-1]
                cx, cy = last_pos_in_history[0], last_pos_in_history[1]
                # Synthesize a temporary box for status checks if needed (e.g., for drawing status)
                avg_size = SETTLED_CIRCLE_RADIUS * 1.5 
                box = (cx - avg_size/2, cy - avg_size/2, cx + avg_size/2, cy + avg_size/2)
            else:
                # No history and not currently detected, skip this TID as it's truly gone/never seen
                continue
        
        # --- Clean up old trail points ---
        if tid in track_history:
            # Remove points older than MAX_TRAIL_AGE_SECONDS
            track_history[tid] = [
                (px, py, ts) for px, py, ts in track_history[tid]
                if (current_time - ts) <= MAX_TRAIL_AGE_SECONDS
            ]
            # Further prune if too far out of frame (to avoid infinitely growing history for truly gone items)
            margin_x = w * 0.1
            margin_y = h * 0.1
            track_history[tid] = [
                (px, py, ts) for px, py, ts in track_history[tid]
                if (-margin_x <= px <= w + margin_x) and (-margin_y <= py <= h + margin_y)
            ]

        # Draw trail dots with fading effect
        for k, (px, py, dot_time) in enumerate(track_history[tid]):
            age_ratio = (current_time - dot_time) / MAX_TRAIL_AGE_SECONDS
            dot_opacity = max(0.1, 1.0 - age_ratio) 
            dot_radius = max(1, int(3 * dot_opacity)) 
            
            dot_color_bgr = [int(c * dot_opacity) for c in asset_colors[tid].as_bgr()]
            cv2.circle(annotated, (px, py), dot_radius, dot_color_bgr, -1)


        # Only draw masks and boxes for CURRENTLY DETECTED assets from YOLO
        if current_detection_exists:
            # Mask Annotation
            mask_idx = np.where(detections.tracker_id == tid)[0][0]
            if detections.mask is not None and mask_idx < len(detections.mask):
                mask_for_this_asset = detections.mask[mask_idx]
                single_mask_img = np.zeros_like(frame, dtype=np.uint8)
                single_mask_bgr = asset_colors[tid].as_bgr()
                single_mask_img[mask_for_this_asset > 0] = single_mask_bgr
                annotated = cv2.addWeighted(annotated, 1, single_mask_img, MASK_ALPHA, 0)
            
            # Initialize box_color for the detected asset
            box_color = sv.Color.WHITE.as_bgr() 
            # Draw bounding box (color will be updated based on status below)
            x1_box, y1_box, x2_box, y2_box = map(int, box) 
            cv2.rectangle(annotated, (x1_box, y1_box), (x2_box, y2_box), box_color, 2)


        # --- Status Logic & Alerts/Logs ---
        prev_asset_status = asset_status.get(tid, 'UNKNOWN') # Get existing status
        new_asset_status = prev_asset_status # Assume status remains the same unless changed below

        # 1. Handle new detections or re-detections
        if prev_asset_status == 'UNKNOWN' or (prev_asset_status == 'MISSING' and current_detection_exists):
            new_asset_status = 'SETTLING'
            asset_settled_info[tid] = {'initial_pos': (cx, cy), 'radius': SETTLED_CIRCLE_RADIUS, 'settled_time': current_time}
            labels.append(f"ID:{tid} Settling...")
            box_color = sv.Color.YELLOW.as_bgr()

        # 2. Handle SETTLING phase
        elif prev_asset_status == 'SETTLING':
            # Settle if time passed AND it's currently detected (stable position)
            if current_time - asset_settled_info[tid]['settled_time'] >= SETTLE_DOWN_SECONDS and current_detection_exists:
                asset_settled_info[tid]['initial_pos'] = (cx, cy) 
                asset_settled_info[tid]['radius'] = SETTLED_CIRCLE_RADIUS # Confirm radius after settling
                new_asset_status = 'OK' # It has settled and is in place
                labels.append(f"ID:{tid} OK (Settled)")
                box_color = sv.Color.GREEN.as_bgr()
            else: # Still settling or briefly lost during settling phase
                labels.append(f"ID:{tid} Settling... ({int(max(0, SETTLE_DOWN_SECONDS - (current_time - asset_settled_info[tid]['settled_time'])))}s)")
                box_color = sv.Color.YELLOW.as_bgr()
                new_asset_status = 'SETTLING' # Explicitly keep settling status

        # 3. Handle OK/MISPLACED for settled assets
        elif prev_asset_status in ['OK', 'MISPLACED']:
            if asset_settled_info.get(tid) and asset_settled_info[tid]['initial_pos']:
                initial_pos = asset_settled_info[tid]['initial_pos'] 
                radius = asset_settled_info[tid]['radius']
                
                # Check position based on current or last known centroid
                check_cx, check_cy = (cx, cy) 

                dist = np.linalg.norm(np.array((check_cx, check_cy)) - np.array(initial_pos))
                
                if dist > radius:
                    new_asset_status = "MISPLACED"
                    box_color = sv.Color.RED.as_bgr()
                    misplaced_count += 1
                else:
                    new_asset_status = "OK"
                    box_color = sv.Color.GREEN.as_bgr()
                    ok_count += 1

                # Draw the circular threshold for settled Assets
                cv2.circle(annotated, initial_pos, radius, (100, 200, 255), 2) # Light blue circle
                cv2.circle(annotated, initial_pos, 5, (100, 200, 255), -1) # Center dot
                labels.append(f"ID:{tid} {new_asset_status}")
            else: # Fallback if initial_pos is somehow missing for an OK/MISPLACED asset
                new_asset_status = 'UNKNOWN' # Treat as unknown if critical data is missing
                labels.append(f"ID:{tid} Error Status")
                box_color = sv.Color.WHITE.as_bgr()
                logger.error(f"Asset ID {tid} in OK/MISPLACED state but missing initial_pos.")

        # 4. Handle MISSING status (set in cleanup block, reflected here)
        elif prev_asset_status == 'MISSING':
            labels.append(f"ID:{tid} MISSING")
            box_color = sv.Color.BLACK.as_bgr() # Or some distinct color for missing
            current_missing_assets_count += 1 # Count for OSD

        # --- Update status and trigger TTS/Log if status changed ---
        if prev_asset_status != new_asset_status:
            asset_status[tid] = new_asset_status # Update the actual status
            alert_key = (tid, new_asset_status)
            if current_time - last_alert_time.get(alert_key, 0) > ALERT_COOLDOWN_SECONDS:
                play_tts_alert(f"Asset {tid} is {new_asset_status}.")
                
                if new_asset_status == "MISPLACED":
                    logger.warning(f"Asset ID {tid} is MISPLACED at ({cx}, {cy}).")
                elif new_asset_status == "OK" and prev_asset_status == "MISPLACED":
                    logger.info(f"Asset ID {tid} is now OK at ({cx}, {cy}).")
                elif new_asset_status == "SETTLING": # For initial settle alert
                    logger.info(f"Asset ID {tid} detected and settling.")
                # MISSING alert is handled in the cleanup section
                last_alert_time[alert_key] = current_time

        # Draw bounding box and arrow for CURRENTLY DETECTED assets
        if current_detection_exists and box is not None:
            x1_box, y1_box, x2_box, y2_box = map(int, box)
            # Use the determined box_color for the current detection
            cv2.rectangle(annotated, (x1_box, y1_box), (x2_box, y2_box), box_color, 2)
            cv2.arrowedLine(annotated, (w // 2, h // 2), (cx, cy),
                            asset_colors[tid].as_bgr(), 3, tipLength=0.03, line_type=cv2.LINE_AA)
        
    # Annotate labels for all currently detected assets
    if detections.tracker_id is not None:
        current_detection_labels = []
        for det_idx, det_tid in enumerate(detections.tracker_id):
            status_text = asset_status.get(det_tid, "Initializing...")
            if status_text == 'SETTLING':
                settling_seconds_left = int(max(0, SETTLE_DOWN_SECONDS - (current_time - asset_settled_info[det_tid]['settled_time'])))
                current_detection_labels.append(f"ID:{det_tid} Settling... ({settling_seconds_left}s)")
            else:
                current_detection_labels.append(f"ID:{det_tid} {status_text}")
        
        annotated = label_annot.annotate(annotated, detections=detections, labels=current_detection_labels)

    # --- Missing Asset Cleanup (when not detected for MISSING_FRAMES or trail is completely gone) ---
    for tid in list(all_known_tids): # Iterate over a copy of keys to safely modify dicts during iteration
        if tid not in active_ids: # Asset not detected in current frame by YOLO
            fps = cap.get(cv2.CAP_PROP_FPS)
            # Default to 30 FPS if cap.get(cv2.CAP_PROP_FPS) is 0 or problematic
            missing_time_threshold = MISSING_FRAMES / fps if fps > 0 else MISSING_FRAMES / 30.0

            time_since_last_yolo_detection = current_time - last_detection_time.get(tid, 0)
            
            # If it hasn't been detected by YOLO for `missing_time_threshold` seconds
            if time_since_last_yolo_detection > missing_time_threshold:
                if asset_status.get(tid) != 'MISSING':
                    # Status change from anything else to MISSING
                    asset_status[tid] = 'MISSING'
                    alert_key = (tid, 'MISSING')
                    if current_time - last_alert_time.get(alert_key, 0) > ALERT_COOLDOWN_SECONDS:
                        play_tts_alert(f"Asset {tid} is missing from the frame.")
                        logger.error(f"Asset ID {tid} is MISSING from the frame.")
                        last_alert_time[alert_key] = current_time
            
        # Finally, if asset's trail is completely empty, remove all its data
        if not track_history.get(tid): 
            # This means the asset has either been missing for longer than MAX_TRAIL_AGE_SECONDS,
            # or it has moved completely out of frame. This is where it's truly "forgotten".
            if asset_status.get(tid) != 'FORGOTTEN': # Introduce a 'FORGOTTEN' state for logging
                logger.info(f"Asset ID {tid} tracking data cleared (trail empty/long-term missing).")
                asset_status[tid] = 'FORGOTTEN' # Mark as forgotten
            
            # Clean up all associated data for the truly forgotten asset
            track_history.pop(tid, None)
            asset_status.pop(tid, None)
            asset_settled_info.pop(tid, None)
            asset_colors.pop(tid, None)
            last_detection_time.pop(tid, None)
            # Clear all alert cooldowns for this TID
            keys_to_delete = [k for k in last_alert_time if k[0] == tid]
            for k in keys_to_delete:
                last_alert_time.pop(k)

    # ---- OSD & show ----
    # Re-calculate missing_count for OSD based on current `asset_status`
    final_missing_count_for_osd = sum(1 for status in asset_status.values() if status == 'MISSING')

    draw_osd(annotated, zone, STEP, ok_count, misplaced_count, final_missing_count_for_osd)
    cv2.imshow("Ultra-Advanced Asset Tracker", annotated)

cap.release()
cv2.destroyAllWindows()
# The daemon TTS thread will automatically exit when the main program terminates.
# No explicit engine.stop() or endLoop() here is needed for the main thread.
logger.info("Application terminated.")