import cv2
import simpleaudio as sa
import time

# -------------------------------
# CONFIGURATION
# -------------------------------
ALARM_SOUND_FILE = "alarm.wav"
CAMERA_INDEX = 0         # 0 = default laptop camera
ALARM_COOLDOWN = 5       # seconds between alarms
MIN_BOX_HEIGHT = 50      # filter very small detections
FRAME_DOWNSCALE = 0.5    # resize factor for faster detection

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

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Resize frame for faster detection
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_DOWNSCALE, fy=FRAME_DOWNSCALE)

    # Detect humans (returns list of rectangles)
    boxes, weights = hog.detectMultiScale(
        small_frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    human_detected = False

    # Draw bounding boxes
    for (x, y, w, h) in boxes:
        if h < MIN_BOX_HEIGHT:
            continue  # skip tiny false positives

        human_detected = True

        # Scale box to original frame size
        x1 = int(x / FRAME_DOWNSCALE)
        y1 = int(y / FRAME_DOWNSCALE)
        x2 = int((x + w) / FRAME_DOWNSCALE)
        y2 = int((y + h) / FRAME_DOWNSCALE)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "Person", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Trigger alarm only when a new human appears
    current_time = time.time()
    if human_detected and not human_present:
        if (current_time - last_alarm_time) > ALARM_COOLDOWN:
            print("🚨 HUMAN DETECTED!")
            alarm_sound.play()
            last_alarm_time = current_time
        human_present = True
    elif not human_detected:
        human_present = False

    # Show frame
    cv2.imshow("Human Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
cv2.destroyAllWindows()
print("Program terminated.")
