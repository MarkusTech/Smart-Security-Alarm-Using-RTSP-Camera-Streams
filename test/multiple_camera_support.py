import cv2
import simpleaudio as sa
import time
import os
import csv

# CONFIGURATION
ALARM_SOUND_FILE = "alarm.wav"
CAMERA_INDICES = [0, 1, 2, 3]  # 4 cameras
ALARM_COOLDOWN = 5 
MIN_BOX_HEIGHT = 50
FRAME_DOWNSCALE = 0.6
DETECTION_FRAMES_REQUIRED = 2
SKIP_FRAMES = 1
CAMERA_IDS = ["CAM1", "CAM2", "CAM3", "CAM4"]

# Directories for saving data
CAPTURE_FOLDER = "captures"
os.makedirs(CAPTURE_FOLDER, exist_ok=True)
LOG_FILE = "detection_log.csv"

# LOAD ALARM SOUND
alarm_sound = sa.WaveObject.from_wave_file(ALARM_SOUND_FILE)

# SETUP HOG HUMAN DETECTOR
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# INITIALIZE CAMERAS
caps = [cv2.VideoCapture(idx) for idx in CAMERA_INDICES]
for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"ERROR: Camera {CAMERA_IDS[i]} not detected")
        exit()

# STATE PER CAMERA
last_alarm_times = [0] * len(CAMERA_INDICES)
human_present_flags = [False] * len(CAMERA_INDICES)
detection_counters = [0] * len(CAMERA_INDICES)
frame_counts = [0] * len(CAMERA_INDICES)

# LOG FILE SETUP
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "CameraID", "ImageFile"])

print("Starting professional human detection on 4 cameras. Press 'q' to quit.")

# MAIN LOOP
while True:
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            continue

        frame_counts[i] += 1

        if frame_counts[i] % SKIP_FRAMES == 0:
            # Resize frame
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
                # Draw boxes
                x1, y1, x2, y2 = int(x / FRAME_DOWNSCALE), int(y / FRAME_DOWNSCALE), int((x + w) / FRAME_DOWNSCALE), int((y + h) / FRAME_DOWNSCALE)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Person {CAMERA_IDS[i]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S"), (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Consecutive detection logic
            if human_detected:
                detection_counters[i] += 1
            else:
                detection_counters[i] = 0

            if detection_counters[i] >= DETECTION_FRAMES_REQUIRED and not human_present_flags[i]:
                current_time = time.time()
                if (current_time - last_alarm_times[i]) > ALARM_COOLDOWN:
                    print(f"HUMAN DETECTED! [{CAMERA_IDS[i]}]")
                    alarm_sound.play()
                    last_alarm_times[i] = current_time

                    # Capture cropped images
                    for (x, y, w, h) in boxes:
                        if h < MIN_BOX_HEIGHT:
                            continue
                        x1, y1, x2, y2 = int(x / FRAME_DOWNSCALE), int(y / FRAME_DOWNSCALE), int((x + w) / FRAME_DOWNSCALE), int((y + h) / FRAME_DOWNSCALE)
                        cropped = frame[y1:y2, x1:x2]
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(CAPTURE_FOLDER, f"{CAMERA_IDS[i]}_human_{timestamp}.jpg")
                        cv2.imwrite(filename, cropped)
                        print(f"Captured cropped image: {filename}")

                        # Log detection
                        with open(LOG_FILE, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([timestamp, CAMERA_IDS[i], filename])

                human_present_flags[i] = True
            elif detection_counters[i] == 0:
                human_present_flags[i] = False

        # Show frame for each camera
        cv2.imshow(f"Camera {CAMERA_IDS[i]}", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
print("Program terminated.")
