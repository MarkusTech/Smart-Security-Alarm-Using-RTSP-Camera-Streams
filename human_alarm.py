import cv2
from ultralytics import YOLO
import simpleaudio as sa
import time

# -------------------------------
# CONFIGURATION
# -------------------------------
ALARM_SOUND_FILE = "alarm.wav"
RTSP_URL = "rtsp://username:password@camera-ip:554/cam/realmonitor?channel=1&subtype=0"
ALARM_COOLDOWN = 5      # seconds between alarms
MIN_BOX_AREA = 5000     # filter tiny boxes

# -------------------------------
# LOAD YOLOv8 MODEL
# -------------------------------
model = YOLO("yolov8n.pt")

# -------------------------------
# LOAD ALARM SOUND
# -------------------------------
alarm_sound = sa.WaveObject.from_wave_file(ALARM_SOUND_FILE)

# -------------------------------
# OPEN RTSP CAMERA
# -------------------------------
cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    print("ERROR: Camera not detected or wrong RTSP URL")
    exit()

print("Starting human detection. Press 'q' to quit.")

last_alarm_time = 0
human_present = False

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Resize frame for faster detection
    small_frame = cv2.resize(frame, (640, 360))
    results = model(small_frame)

    human_detected = False

    # Draw boxes around humans
    for r in results:
        for obj in r.boxes:
            cls = int(obj.cls[0])
            label = model.names[cls]
            if label == "person":
                x1, y1, x2, y2 = map(int, obj.xyxy[0])
                box_area = (x2 - x1) * (y2 - y1)
                if box_area < MIN_BOX_AREA:
                    continue
                human_detected = True
                x_scale = frame.shape[1] / small_frame.shape[1]
                y_scale = frame.shape[0] / small_frame.shape[0]
                x1 = int(x1 * x_scale)
                x2 = int(x2 * x_scale)
                y1 = int(y1 * y_scale)
                y2 = int(y2 * y_scale)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    current_time = time.time()
    if human_detected and not human_present:
        if (current_time - last_alarm_time) > ALARM_COOLDOWN:
            print("🚨 HUMAN DETECTED!")
            alarm_sound.play()
            last_alarm_time = current_time
        human_present = True
    elif not human_detected:
        human_present = False

    cv2.imshow("Smart Security Alarm", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
