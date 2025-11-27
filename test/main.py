import cv2
import simpleaudio as sa
import time
import os
import csv

# CONFIGURATION
ALARM_SOUND_FILE = "alarm.wav"
CAMERA_INDEX = 0
ALARM_COOLDOWN = 5 
MIN_BOX_HEIGHT = 50
FRAME_DOWNSCALE = 0.6
DETECTION_FRAMES_REQUIRED = 2
SKIP_FRAMES = 1
CAMERA_ID = "CAM1"

# Directories for saving data
CAPTURE_FOLDER = "captures"
os.makedirs(CAPTURE_FOLDER, exist_ok=True)
LOG_FILE = "detection_log.csv"

# LOAD ALARM SOUND
alarm_sound = sa.WaveObject.from_wave_file(ALARM_SOUND_FILE)

# OPEN CAMERA
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("ERROR: Camera not detected")
    exit()

print("Starting professional human detection. Press 'q' to quit.")

# SETUP HOG HUMAN DETECTOR
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

last_alarm_time = 0
human_present = False
detection_counter = 0
frame_count = 0

# LOG FILE SETUP
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "CameraID", "ImageFile"])

# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1

    if frame_count % SKIP_FRAMES == 0:
        # Resize frame for faster detection
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_DOWNSCALE, fy=FRAME_DOWNSCALE)

        # Detect humans
        boxes, weights = hog.detectMultiScale(
            small_frame,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.1
        )

        human_detected = False

        for (x, y, w, h) in boxes:
            if h < MIN_BOX_HEIGHT:
                continue

            human_detected = True

            # Scale to original frame
            x1 = int(x / FRAME_DOWNSCALE)
            y1 = int(y / FRAME_DOWNSCALE)
            x2 = int((x + w) / FRAME_DOWNSCALE)
            y2 = int((y + h) / FRAME_DOWNSCALE)

            # Draw bounding box and overlay info
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Person {CAMERA_ID}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S"), (10, frame.shape[0] - 10),
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
                alarm_sound.play()
                last_alarm_time = current_time

                # AUTO CAPTURE CROPPED HUMAN IMAGE
                for (x, y, w, h) in boxes:
                    if h < MIN_BOX_HEIGHT:
                        continue
                    x1 = int(x / FRAME_DOWNSCALE)
                    y1 = int(y / FRAME_DOWNSCALE)
                    x2 = int((x + w) / FRAME_DOWNSCALE)
                    y2 = int((y + h) / FRAME_DOWNSCALE)
                    cropped = frame[y1:y2, x1:x2]
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(CAPTURE_FOLDER, f"{CAMERA_ID}_human_{timestamp}.jpg")
                    cv2.imwrite(filename, cropped)
                    print(f"Captured cropped image: {filename}")

                    # Log detection
                    with open(LOG_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp, CAMERA_ID, filename])

            human_present = True
        elif detection_counter == 0:
            human_present = False

    # Show frame
    cv2.imshow("Professional Human Detection", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("Program terminated.")
