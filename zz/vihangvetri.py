import warnings
warnings.simplefilter(action='ignore', category=Warning)
import cv2
import torch
import numpy as np
from PIL import Image
import smtplib
from email.message import EmailMessage
import ssl
import pyttsx3
import os

# ==================== Email Configuration ====================
EMAIL_ADDRESS = "muttuh028@gmail.com"
EMAIL_PASSWORD = "kcxj atas xddm bbqr"  # <-- replace with your Gmail app password
TO_EMAIL = "hmuktanandg@gmail.com"

# ==================== TTS Setup ====================
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # speed of speech

# ==================== Load YOLOv5 ====================
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='github', verbose=False)

# ==================== Video source ====================
video_path = r"C:\Users\adity\Desktop\OBJDETECT\data\dronenew.mp4"
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    print("Please update 'video_path' to the correct location of your video.")
    exit()

cap = cv2.VideoCapture(video_path)

# Read the first frame to get dimensions for initial zone centering
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame from the video.")
    exit()

frame_height, frame_width, _ = frame.shape
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video to the beginning for the main loop

# ==================== Zone Configuration ====================
# Define desired width and height for the initial zone
initial_zone_width = 600
initial_zone_height = 400

# Calculate x and y to center the zone
zone_x = (frame_width - initial_zone_width) // 2
zone_y = (frame_height - initial_zone_height) // 2

zone = {'x': zone_x, 'y': zone_y, 'width': initial_zone_width, 'height': initial_zone_height} # <--- CHANGED HERE
dragging_zone = False
resizing_zone = False
resize_handle = -1
start_drag_x, start_drag_y = 0, 0

# Constants for interaction
HANDLE_SIZE = 10
ZONE_COLOR_NORMAL = (0, 255, 0)
ZONE_COLOR_ACTIVE = (0, 255, 255)
ZONE_THICKNESS = 2

def get_zone_corners(zone):
    x, y, w, h = zone['x'], zone['y'], zone['width'], zone['height']
    return [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h)
    ]

def is_inside_zone(zone, mx, my):
    return zone['x'] <= mx <= zone['x'] + zone['width'] and \
           zone['y'] <= my <= zone['y'] + zone['height']

def get_resize_handle(zone, mx, my, handle_size):
    x, y, w, h = zone['x'], zone['y'], zone['width'], zone['height']

    if abs(mx - x) <= handle_size and abs(my - y) <= handle_size: return 0
    if abs(mx - (x + w)) <= handle_size and abs(my - y) <= handle_size: return 1
    if abs(mx - (x + w)) <= handle_size and abs(my - (y + h)) <= handle_size: return 2
    if abs(mx - x) <= handle_size and abs(my - (y + h)) <= handle_size: return 3

    if x + handle_size < mx < x + w - handle_size:
        if abs(my - y) <= handle_size: return 4
        if abs(my - (y + h)) <= handle_size: return 6
    if y + handle_size < my < y + h - handle_size:
        if abs(mx - x) <= handle_size: return 7
        if abs(mx - (x + w)) <= handle_size: return 5

    return -1

def mouse_event(event, x, y, flags, param):
    global zone, dragging_zone, resizing_zone, resize_handle, start_drag_x, start_drag_y

    if event == cv2.EVENT_LBUTTONDOWN:
        resize_handle = get_resize_handle(zone, x, y, HANDLE_SIZE)
        if resize_handle != -1:
            resizing_zone = True
            start_drag_x, start_drag_y = x, y
        elif is_inside_zone(zone, x, y):
            dragging_zone = True
            start_drag_x, start_drag_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        dragging_zone = False
        resizing_zone = False
        resize_handle = -1

        if zone['width'] < 0:
            zone['x'] += zone['width']
            zone['width'] = abs(zone['width'])
        if zone['height'] < 0:
            zone['y'] += zone['height']
            zone['height'] = abs(zone['height'])

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_zone:
            dx = x - start_drag_x
            dy = y - start_drag_y
            zone['x'] += dx
            zone['y'] += dy
            start_drag_x, start_drag_y = x, y
        elif resizing_zone:
            dx = x - start_drag_x
            dy = y - start_drag_y

            current_x, current_y, current_w, current_h = zone['x'], zone['y'], zone['width'], zone['height']

            if resize_handle == 0: # Top-left
                zone['x'] = x
                zone['y'] = y
                zone['width'] = current_w - dx
                zone['height'] = current_h - dy
            elif resize_handle == 1: # Top-right
                zone['y'] = y
                zone['width'] = current_w + dx
                zone['height'] = current_h - dy
            elif resize_handle == 2: # Bottom-right
                zone['width'] = current_w + dx
                zone['height'] = current_h + dy
            elif resize_handle == 3: # Bottom-left
                zone['x'] = x
                zone['width'] = current_w - dx
                zone['height'] = current_h + dy
            elif resize_handle == 4: # Top edge
                zone['y'] = y
                zone['height'] = current_h - dy
            elif resize_handle == 5: # Right edge
                zone['width'] = current_w + dx
            elif resize_handle == 6: # Bottom edge
                zone['height'] = current_h + dy
            elif resize_handle == 7: # Left edge
                zone['x'] = x
                zone['width'] = current_w - dx

            start_drag_x, start_drag_y = x, y

        cursor_type = cv2.EVENT_MOUSEMOVE
        if get_resize_handle(zone, x, y, HANDLE_SIZE) != -1:
            cursor_type = cv2.EVENT_LBUTTONDOWN
        elif is_inside_zone(zone, x, y):
            cursor_type = cv2.EVENT_FLAG_LBUTTON


cv2.namedWindow('frame')
cv2.setMouseCallback('frame', mouse_event)

alert_sent = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or video cannot be read.")
        break

    # Get current zone boundaries
    rect_x_min = zone['x']
    rect_y_min = zone['y']
    rect_x_max = zone['x'] + zone['width']
    rect_y_max = zone['y'] + zone['height']

    if zone['width'] < 0:
        rect_x_min, rect_x_max = rect_x_max, rect_x_min
    if zone['height'] < 0:
        rect_y_min, rect_y_max = rect_y_max, rect_y_min

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = model(img, size=640)

    detections = results.xyxy[0].tolist()

    drone_in_zone = False

    for result in detections:
        x1, y1, x2, y2, conf, cls = result
        if conf > 0.5:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            if rect_x_min <= cx <= rect_x_max and rect_y_min <= cy <= rect_y_max:
                drone_in_zone = True
                cv2.putText(frame, "Drone in Zone!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if not alert_sent:
                    screenshot_path = "frame_screenshot.jpg"
                    cv2.imwrite(screenshot_path, frame)

                    drone_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    drone_path = "drone_crop.jpg"
                    cv2.imwrite(drone_path, drone_crop)

                    subject = "ALERT: Drone Detected in Zone"
                    body = f"Drone detected at coordinates: ({cx}, {cy}). Screenshot and cropped image attached."

                    msg = EmailMessage()
                    msg["From"] = EMAIL_ADDRESS
                    msg["To"] = TO_EMAIL
                    msg["Subject"] = subject
                    msg.set_content(body)

                    with open(screenshot_path, 'rb') as f:
                        msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename="screenshot.jpg")
                    with open(drone_path, 'rb') as f:
                        msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename="drone.jpg")

                    try:
                        context = ssl.create_default_context()
                        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
                            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                            smtp.send_message(msg)

                        print("Drone detected in zone!")
                        print("Alert email sent")
                        engine.say("Warning! Drone detected in the zone.")
                        engine.runAndWait()

                    except Exception as e:
                        print(f"Error sending email: {e}")
                        pass

                    alert_sent = True

    if not drone_in_zone:
        alert_sent = False

    zone_color = ZONE_COLOR_ACTIVE if dragging_zone or resizing_zone else ZONE_COLOR_NORMAL
    cv2.rectangle(frame, (zone['x'], zone['y']),
                  (zone['x'] + zone['width'], zone['y'] + zone['height']),
                  zone_color, ZONE_THICKNESS)

    for corner in get_zone_corners(zone):
        cv2.circle(frame, corner, 5, zone_color, -1)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()