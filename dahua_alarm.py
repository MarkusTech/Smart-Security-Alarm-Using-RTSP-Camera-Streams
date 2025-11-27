import cv2
from ultralytics import YOLO
import time
import os
import csv
import winsound
from dotenv import load_dotenv

load_dotenv()

CAMERA_ID = os.getenv("CAMERA_ID", "CAM1")
RTSP_URL = os.getenv("RTSP_URL")
ALARM_COOLDOWN = int(os.getenv("ALARM_COOLDOWN", 5))
DETECTION_FRAMES_REQUIRED = int(os.getenv("DETECTION_FRAMES_REQUIRED", 2))
CAPTURE_FOLDER = os.getenv("CAPTURE_FOLDER", "captures")
LOG_FILE = os.getenv("LOG_FILE", "detection_log.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")

os.makedirs(CAPTURE_FOLDER, exist_ok=True)
model = YOLO(MODEL_PATH)

# OPEN CAMERA
cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    print(f"ERROR: Cannot connect to Dahua camera {CAMERA_ID}")
    exit()

# LOG FILE SETUP
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "CameraID", "ImageFile"])

print(f"Starting human detection on {CAMERA_ID}. Press 'q' to quit.")

last_alarm_time = 0
human_present = False
detection_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    results = model(frame)

    human_detected = False

    for r in results:
        for obj in r.boxes:
            cls = int(obj.cls[0])
            if model.names[cls] == "person":
                human_detected = True
                x1, y1, x2, y2 = map(int, obj.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Person {CAMERA_ID}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S"),
                            (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Consecutive detection logic
    if human_detected:
        detection_counter += 1
    else:
        detection_counter = 0

    if detection_counter >= DETECTION_FRAMES_REQUIRED and not human_present:
        current_time = time.time()
        if (current_time - last_alarm_time) > ALARM_COOLDOWN:
            print(f"HUMAN DETECTED! [{CAMERA_ID}]")
            winsound.Beep(1000, 400)
            last_alarm_time = current_time

            for r in results:
                for obj in r.boxes:
                    cls = int(obj.cls[0])
                    if model.names[cls] == "person":
                        x1, y1, x2, y2 = map(int, obj.xyxy[0])
                        cropped = frame[y1:y2, x1:x2]
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(CAPTURE_FOLDER, f"{CAMERA_ID}_human_{timestamp}.jpg")
                        cv2.imwrite(filename, cropped)
                        print(f"Captured cropped image: {filename}")

                        with open(LOG_FILE, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([timestamp, CAMERA_ID, filename])

        human_present = True
    elif detection_counter == 0:
        human_present = False

    cv2.imshow("Dahua CCTV Human Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program terminated.")
