import cv2
import numpy as np
import threading
import time
from collections import deque
from playsound import playsound
from ultralytics import YOLO

# --- Configuration ---
YOLO_MODEL_PATH = r'c:\Users\adity\Desktop\OBJDETECT\yolov8x.pt'
CAMERA_SOURCE = 0
BIRD_CLASS_NAME = 'bird'  # <-- Only birds
BIRD_CLASS_ID = -1

# Tracking & collision parameters
MAX_TRACKING_AGE = 30
CRITICAL_PROXIMITY_THRESHOLD_METERS = 50
WARNING_PROXIMITY_THRESHOLD_METERS = 150
COLLISION_TTC_THRESHOLD_SECONDS = 3

# Alert settings
ALERT_DISPLAY_DURATION_SECONDS = 2
ALERT_SOUND_PATH = 'alert.mp3'
FONT = cv2.FONT_HERSHEY_SIMPLEX
ALERT_TEXT_SCALE = 1.5
ALERT_TEXT_THICKNESS = 3
ALERT_TEXT_COLOR = (255, 255, 255)

# --- Global state ---
current_alert_message = ""
alert_display_start_time = 0
ir_mode_active = False

# --- Sound utility ---
def play_alert_sound(sound_file):
    try:
        playsound(sound_file)
    except Exception as e:
        print(f"Error playing sound: {e}")

# --- Centroid calculation helper ---
def compute_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

# --- Main function ---
def run_system():
    global BIRD_CLASS_ID, current_alert_message, alert_display_start_time, ir_mode_active

    # Load YOLO model
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"Model loaded from {YOLO_MODEL_PATH}")
        for class_id, class_name in model.names.items():
            if class_name.strip().lower() == BIRD_CLASS_NAME.lower():
                BIRD_CLASS_ID = class_id
                break
        if BIRD_CLASS_ID == -1:
            print(f"Error: '{BIRD_CLASS_NAME}' not found in model classes. Available: {model.names}")
            return
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"Error: Cannot open source {CAMERA_SOURCE}")
        return

    tracks = {}
    next_track_id = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    last_alert_sound_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or failed to grab frame.")
            break

        fps_frame_count += 1
        display_frame = frame.copy()

        # --- Black-hot IR mode ---
        if ir_mode_active:
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            inverted = cv2.bitwise_not(gray)
            high_contrast = cv2.normalize(inverted, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            display_frame = cv2.cvtColor(high_contrast, cv2.COLOR_GRAY2BGR)

        # YOLO detection: ONLY BIRDS
        results = model(frame, conf=0.5, classes=[BIRD_CLASS_ID])
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append({'bbox': [x1, y1, x2, y2], 'conf': conf})

        # Tracking logic
        new_tracks = {}
        matched = [False] * len(detections)

        for tid, info in tracks.items():
            best_idx, min_dist = -1, float('inf')
            prev_centroid = info['centroid']

            for i, det in enumerate(detections):
                if matched[i]:
                    continue
                det_centroid = compute_centroid(det['bbox'])
                dist = np.linalg.norm(np.array(prev_centroid) - np.array(det_centroid))
                if dist < 100 and dist < min_dist:
                    best_idx, min_dist = i, dist

            if best_idx != -1:
                matched[best_idx] = True
                det = detections[best_idx]
                curr_centroid = compute_centroid(det['bbox'])
                vx, vy = curr_centroid[0] - prev_centroid[0], curr_centroid[1] - prev_centroid[1]

                info.update({
                    'bbox': det['bbox'],
                    'centroid': curr_centroid,
                    'last_seen': fps_frame_count,
                    'velocity': (vx, vy)
                })
                info['history'].append(curr_centroid)
                if len(info['history']) > 10:
                    info['history'].popleft()
                new_tracks[tid] = info

        for i, det in enumerate(detections):
            if not matched[i]:
                centroid = compute_centroid(det['bbox'])
                new_tracks[next_track_id] = {
                    'bbox': det['bbox'],
                    'centroid': centroid,
                    'last_seen': fps_frame_count,
                    'velocity': (0, 0),
                    'history': deque([centroid])
                }
                next_track_id += 1

        # Remove stale tracks
        stale = [tid for tid, info in new_tracks.items() if (fps_frame_count - info['last_seen']) > MAX_TRACKING_AGE]
        for tid in stale:
            del new_tracks[tid]

        tracks = new_tracks

        # Alert logic
        collision = False
        warning = False

        for tid, info in tracks.items():
            x1, y1, x2, y2 = info['bbox']
            centroid_x, centroid_y = info['centroid']
            vx, vy = info['velocity']

            est_dist = 1000 / (y2 - y1 + 1)
            if est_dist < CRITICAL_PROXIMITY_THRESHOLD_METERS:
                collision = True
            elif est_dist < WARNING_PROXIMITY_THRESHOLD_METERS:
                warning = True

            # Draw
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_frame, f'{BIRD_CLASS_NAME} {tid}', (x1, y1 - 10), FONT, 0.7, (255, 0, 0), 2)
            cv2.circle(display_frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
            cv2.arrowedLine(display_frame, (centroid_x, centroid_y), (int(centroid_x + vx * 5), int(centroid_y + vy * 5)), (255, 0, 0), 2)
            for i in range(1, len(info['history'])):
                cv2.line(display_frame, info['history'][i-1], info['history'][i], (0, 255, 255), 1)

        # Alerts
        new_alert = False
        bg_color = (0, 0, 0)

        if collision:
            msg = "RUNWAY NOT CLEAR"
            bg_color = (0, 0, 255)
        elif warning:
            msg = "WARNING: Bird close"
            bg_color = (0, 165, 255)
        else:
            msg = ""

        if msg != current_alert_message:
            current_alert_message = msg
            alert_display_start_time = time.time()
            new_alert = True

        if new_alert and (time.time() - last_alert_sound_time > ALERT_DISPLAY_DURATION_SECONDS / 2):
            threading.Thread(target=play_alert_sound, args=(ALERT_SOUND_PATH,), daemon=True).start()
            last_alert_sound_time = time.time()

        if current_alert_message:
            text_size = cv2.getTextSize(current_alert_message, FONT, ALERT_TEXT_SCALE, ALERT_TEXT_THICKNESS)[0]
            x = (display_frame.shape[1] - text_size[0]) // 2
            y = 70
            cv2.rectangle(display_frame, (x - 10, y - text_size[1] - 10), (x + text_size[0] + 10, y + 10), bg_color, -1)
            cv2.putText(display_frame, current_alert_message, (x, y), FONT, ALERT_TEXT_SCALE, ALERT_TEXT_COLOR, ALERT_TEXT_THICKNESS)

        # FPS
        if time.time() - fps_start_time >= 1.0:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()
        cv2.putText(display_frame, f'FPS: {fps:.2f}', (10, 30), FONT, 1, (0, 255, 0), 2)

        # Mode and keys
        mode_text = "Black-hot IR Mode" if ir_mode_active else "Normal Mode"
        cv2.putText(display_frame, mode_text, (10, 70), FONT, 1, (255, 255, 0), 2)
        cv2.putText(display_frame, "Keys: q=Quit, i=IR ON, e=IR OFF", (10, display_frame.shape[0] - 20), FONT, 0.7, (200, 200, 200), 2)

        cv2.imshow('IR Bird Detection', display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('i'):
            ir_mode_active = True
        elif key == ord('e'):
            ir_mode_active = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("--- IR Bird Detection System ---")
    print(f"Model path: {YOLO_MODEL_PATH}")
    print(f"Alert sound: {ALERT_SOUND_PATH}")
    print("Press 'q' to quit, 'i' to enable IR mode, 'e' to disable IR mode.")
    run_system()
    print("--- System Shut Down ---")
