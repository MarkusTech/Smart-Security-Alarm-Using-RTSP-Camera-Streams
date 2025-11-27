import cv2
from ultralytics import YOLO
import simpleaudio as sa
import time

# -------------------------------
# CONFIGURATION
# -------------------------------
ALARM_SOUND_FILE = "alarm.wav"
CAMERA_INDEX = 0
ALARM_COOLDOWN = 5      # seconds
MIN_BOX_AREA = 5000
FRAME_SKIP = 2          # process every 2nd frame for efficiency

# -------------------------------
# LOAD YOLO MODEL
# -------------------------------
model = YOLO("yolov8n.pt")  # small model for speed

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

last_alarm_time = 0
human_present = False
frame_count = 0

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    # Skip frames to save CPU
    if frame_count % FRAME_SKIP != 0:
        cv2.imshow("Human Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Resize frame smaller for detection (faster & less memory)
    small_frame = cv2.resize(frame, (320, 180))

    results = model(small_frame, verbose=False)

    human_detected = False

    # Draw bounding boxes
    for r in results:
        for obj in r.boxes:
            cls = int(obj.cls[0])
            label = model.names[cls]
            if label != "person":
                continue

            x1, y1, x2, y2 = map(int, obj.xyxy[0])
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < MIN_BOX_AREA / 4:  # scale for smaller frame
                continue

            human_detected = True

            # Scale box to original frame
            x_scale = frame.shape[1] / small_frame.shape[1]
            y_scale = frame.shape[0] / small_frame.shape[0]
            x1 = int(x1 * x_scale)
            x2 = int(x2 * x_scale)
            y1 = int(y1 * y_scale)
            y2 = int(y2 * y_scale)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Trigger alarm once per new human detection
    current_time = time.time()
    if human_detected and not human_present:
        if (current_time - last_alarm_time) > ALARM_COOLDOWN:
            print("🚨 HUMAN DETECTED!")
            # Play alarm asynchronously
            alarm_sound.play()
            last_alarm_time = current_time
        human_present = True
    elif not human_detected:
        human_present = False

    # Show frame
    cv2.imshow("Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
cv2.destroyAllWindows()
print("Program terminated.")
