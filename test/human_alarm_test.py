import cv2
import simpleaudio as sa
import time

# -------------------------------
# CONFIGURATION
# -------------------------------
ALARM_SOUND_FILE = "alarm.wav"
CAMERA_INDEX = 0
ALARM_COOLDOWN = 5          # seconds between alarms
MIN_BOX_HEIGHT = 50
FRAME_DOWNSCALE = 0.6       # slightly larger frame for better detection
DETECTION_FRAMES_REQUIRED = 2  # require consecutive detections
SKIP_FRAMES = 1             # run detection every N frames

# -------------------------------
# LOAD ALARM SOUND
# -------------------------------
alarm_sound = sa.WaveObject.from_wave_file(ALARM_SOUND_FILE)

# -------------------------------
# OPEN CAMERA
# -------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("ERROR: Camera not detected")
    exit()

print("Starting human detection. Press 'q' to quit.")

# -------------------------------
# SETUP HOG HUMAN DETECTOR
# -------------------------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

last_alarm_time = 0
human_present = False
detection_counter = 0
frame_count = 0

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1

    # Only run detection on every SKIP_FRAMES-th frame
    if frame_count % SKIP_FRAMES == 0:
        # Resize frame for detection
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_DOWNSCALE, fy=FRAME_DOWNSCALE)

        boxes, weights = hog.detectMultiScale(
            small_frame,
            winStride=(4, 4),  # smaller stride = more accurate
            padding=(8, 8),
            scale=1.1          # larger scale = catch humans of different sizes
        )

        human_detected = False

        for (x, y, w, h) in boxes:
            if h < MIN_BOX_HEIGHT:
                continue

            human_detected = True
            x1 = int(x / FRAME_DOWNSCALE)
            y1 = int(y / FRAME_DOWNSCALE)
            x2 = int((x + w) / FRAME_DOWNSCALE)
            y2 = int((y + h) / FRAME_DOWNSCALE)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Require consecutive detections to avoid misses
        if human_detected:
            detection_counter += 1
        else:
            detection_counter = 0

        if detection_counter >= DETECTION_FRAMES_REQUIRED and not human_present:
            current_time = time.time()
            if (current_time - last_alarm_time) > ALARM_COOLDOWN:
                print("🚨 HUMAN DETECTED!")
                alarm_sound.play()
                last_alarm_time = current_time
            human_present = True
        elif detection_counter == 0:
            human_present = False

    # Show frame
    cv2.imshow("Human Detection", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
cv2.destroyAllWindows()
print("Program terminated.")
