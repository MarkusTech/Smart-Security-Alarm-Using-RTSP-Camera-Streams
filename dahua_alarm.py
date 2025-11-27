import cv2
from ultralytics import YOLO
import winsound

# ---- Load YOLO Model ----
model = YOLO("yolov8n.pt") 

# ---- Dahua RTSP Link ----
rtsp_url = "rtsp://admin:password@192.168.1.120:554/cam/realmonitor?channel=1&subtype=0"

cap = cv2.VideoCapture(rtsp_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    # Run detection
    results = model(frame)

    # Check for human
    for r in results:
        for obj in r.boxes:
            cls = int(obj.cls[0])
            if model.names[cls] == "person":
                print("HUMAN DETECTED!")
                winsound.Beep(1000, 400)  # Alarm on PC speaker

    # Show live video
    cv2.imshow("Dahua CCTV Human Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
