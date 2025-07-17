import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import threading
from playsound import playsound

# --- Configuration ---
YOLO_MODEL_PATH = r'c:\Users\adity\Desktop\OBJDETECT\yolo11n.pt'
VIDEO_PATH = r'c:\Users\adity\Desktop\OBJDETECT\data\birds2.mp4'
CAMERA_SOURCE = VIDEO_PATH

BIRD_CLASS_NAME = 'bird'
BIRD_CLASS_ID = -1

MAX_TRACKING_AGE = 30
CRITICAL_PROXIMITY_THRESHOLD_METERS = 50
WARNING_PROXIMITY_THRESHOLD_METERS = 150

ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 300, 200, 900, 600
ROI_COLOR = (0, 255, 255)
ROI_THICKNESS = 2

ALERT_DISPLAY_DURATION_SECONDS = 2
ALERT_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
ALERT_TEXT_SCALE = 1.5
ALERT_TEXT_THICKNESS = 3
ALERT_TEXT_COLOR = (255, 255, 255)
ALERT_BACKGROUND_COLOR_CRITICAL = (0, 0, 255)
ALERT_BACKGROUND_COLOR_WARNING = (0, 165, 255)
ALERT_BACKGROUND_COLOR_ROI = (255, 0, 0)

ALERT_SOUND_PATH = 'alert.mp3'

current_alert_message = ""
alert_display_start_time = 0
ir_mode_active = False
last_alert_sound_time = 0
last_log_time = time.time()

def play_alert_sound(sound_file):
    try:
        playsound(sound_file)
    except Exception as e:
        print(f"Error playing sound: {e}")

def run_bird_avoidance_system():
    global BIRD_CLASS_ID, current_alert_message, alert_display_start_time, ir_mode_active, last_alert_sound_time, last_log_time

    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"YOLO model loaded from {YOLO_MODEL_PATH}")

        for class_id, class_name in model.names.items():
            if class_name.strip().lower() == BIRD_CLASS_NAME.lower():
                BIRD_CLASS_ID = class_id
                break

        if BIRD_CLASS_ID == -1:
            print(f"Error: '{BIRD_CLASS_NAME}' not found in model classes. Available: {model.names}")
            return
        else:
            print(f"Detected class ID for '{BIRD_CLASS_NAME}': {BIRD_CLASS_ID}")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {CAMERA_SOURCE}")
        return

    frame_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    tracks = {}
    next_track_id = 0

    print("Starting bird detection... Press 'q' or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or failed to grab frame.")
            break

        frame_count += 1
        fps_frame_count += 1
        display_frame = frame.copy()

        if ir_mode_active:
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            inverted = cv2.bitwise_not(gray)
            high_contrast = cv2.normalize(inverted, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            display_frame = cv2.cvtColor(high_contrast, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(display_frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), ROI_COLOR, ROI_THICKNESS)
        cv2.putText(display_frame, 'ROI', (ROI_X1, ROI_Y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROI_COLOR, 2)

        results = model(frame, verbose=False, conf=0.5, classes=[BIRD_CLASS_ID])

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if cls == BIRD_CLASS_ID:
                    detections.append({'bbox': [x1, y1, x2, y2], 'conf': conf})

        new_tracks = {}
        matched_detections = [False] * len(detections)

        for track_id, track_info in tracks.items():
            best_match_idx = -1
            min_dist = float('inf')
            track_centroid = track_info['centroid']

            for i, det in enumerate(detections):
                if matched_detections[i]:
                    continue
                det_centroid = ((det['bbox'][0] + det['bbox'][2]) // 2, (det['bbox'][1] + det['bbox'][3]) // 2)
                dist = np.linalg.norm(np.array(track_centroid) - np.array(det_centroid))
                if dist < 100 and dist < min_dist:
                    min_dist = dist
                    best_match_idx = i

            if best_match_idx != -1:
                matched_detections[best_match_idx] = True
                matched_det = detections[best_match_idx]
                prev_centroid = track_info['centroid']
                current_centroid = ((matched_det['bbox'][0] + matched_det['bbox'][2]) // 2, (matched_det['bbox'][1] + matched_det['bbox'][3]) // 2)

                vx = current_centroid[0] - prev_centroid[0]
                vy = current_centroid[1] - prev_centroid[1]

                track_info['bbox'] = matched_det['bbox']
                track_info['centroid'] = current_centroid
                track_info['last_seen'] = frame_count
                track_info['velocity'] = (vx, vy)
                track_info['history'].append(current_centroid)
                if len(track_info['history']) > 10:
                    track_info['history'].popleft()
                new_tracks[track_id] = track_info

        for i, det in enumerate(detections):
            if not matched_detections[i]:
                centroid = ((det['bbox'][0] + det['bbox'][2]) // 2, (det['bbox'][1] + det['bbox'][3]) // 2)
                new_tracks[next_track_id] = {
                    'bbox': det['bbox'],
                    'centroid': centroid,
                    'last_seen': frame_count,
                    'velocity': (0, 0),
                    'history': deque([centroid])
                }
                next_track_id += 1

        tracks = {tid: info for tid, info in new_tracks.items() if (frame_count - info['last_seen']) <= MAX_TRACKING_AGE}

        collision_imminent = False
        warning_active = False
        bird_in_roi = False

        for track_id, track_info in tracks.items():
            x1, y1, x2, y2 = track_info['bbox']
            centroid_x, centroid_y = track_info['centroid']
            vx, vy = track_info['velocity']

            bird_height_pixels = y2 - y1
            estimated_distance_meters = (1000 / (bird_height_pixels + 1))

            if estimated_distance_meters < CRITICAL_PROXIMITY_THRESHOLD_METERS:
                collision_imminent = True
            elif estimated_distance_meters < WARNING_PROXIMITY_THRESHOLD_METERS:
                warning_active = True

            if (x1 < ROI_X2 and x2 > ROI_X1 and y1 < ROI_Y2 and y2 > ROI_Y1):
                bird_in_roi = True
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), ALERT_BACKGROUND_COLOR_ROI, 2)
                cv2.putText(display_frame, 'IN ROI', (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ALERT_BACKGROUND_COLOR_ROI, 2)
            else:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # --- Advanced arrow visualization ---
            cv2.circle(display_frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  # Red point
            cv2.arrowedLine(display_frame, (centroid_x, centroid_y),
                            (int(centroid_x + vx * 5), int(centroid_y + vy * 5)),
                            (0, 255, 0), 2)

        now = time.time()
        if now - last_log_time >= 3:
            if bird_in_roi:
                print("⚠️ Bird inside ROI area!")
            elif collision_imminent:
                print("🔥 Collision imminent detected!")
            elif warning_active:
                print("⚠️ Bird detected nearby (warning)!")
            elif tracks:
                print("✅ Bird detected.")
            else:
                print("No birds detected.")
            last_log_time = now

        temp_alert_message = ""
        temp_alert_color = (0, 0, 0)

        if bird_in_roi:
            temp_alert_message = "BIRD IN RESTRICTED AREA!"
            temp_alert_color = ALERT_BACKGROUND_COLOR_ROI
        elif collision_imminent:
            temp_alert_message = "CRITICAL: COLLISION IMMINENT!"
            temp_alert_color = ALERT_BACKGROUND_COLOR_CRITICAL
        elif warning_active:
            temp_alert_message = "WARNING: Bird detected close!"
            temp_alert_color = ALERT_BACKGROUND_COLOR_WARNING

        if temp_alert_message != current_alert_message:
            current_alert_message = temp_alert_message
            alert_background_color = temp_alert_color
            alert_display_start_time = time.time()
            if current_alert_message and (time.time() - last_alert_sound_time > ALERT_DISPLAY_DURATION_SECONDS / 2):
                threading.Thread(target=play_alert_sound, args=(ALERT_SOUND_PATH,)).start()
                last_alert_sound_time = time.time()
        elif current_alert_message and (time.time() - alert_display_start_time >= ALERT_DISPLAY_DURATION_SECONDS):
            if not (bird_in_roi or collision_imminent or warning_active):
                current_alert_message = ""

        if current_alert_message:
            text_size = cv2.getTextSize(current_alert_message, ALERT_TEXT_FONT, ALERT_TEXT_SCALE, ALERT_TEXT_THICKNESS)[0]
            text_x = (display_frame.shape[1] - text_size[0]) // 2
            text_y = 70
            cv2.rectangle(display_frame, (text_x - 10, text_y - text_size[1] - 10),
                          (text_x + text_size[0] + 10, text_y + 10), alert_background_color, -1)
            cv2.putText(display_frame, current_alert_message, (text_x, text_y),
                        ALERT_TEXT_FONT, ALERT_TEXT_SCALE, ALERT_TEXT_COLOR, ALERT_TEXT_THICKNESS)

        if time.time() - fps_start_time >= 1.0:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()
        cv2.putText(display_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        mode_text = "Black-hot IR Mode" if ir_mode_active else "Normal Mode"
        cv2.putText(display_frame, mode_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(display_frame, "Keys: q=Quit, ESC=Quit, i=IR ON, e=IR OFF", (10, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow('Bird Detection & Tracking', display_frame)
        key = cv2.waitKey(int(1000 / fps_video) if fps_video > 0 else 1) & 0xFF
        if key == ord('q') or key == 27:  # ESC key
            print("Quit key pressed. Exiting gracefully.")
            break
        elif key == ord('i'):
            ir_mode_active = True
            print("Black-hot IR Mode Activated.")
        elif key == ord('e'):
            ir_mode_active = False
            print("Normal Mode Activated.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("--- Bird Avoidance System Initializing ---")
    print("Video Source:", VIDEO_PATH)
    run_bird_avoidance_system()
    print("--- System Shut Down ---")
