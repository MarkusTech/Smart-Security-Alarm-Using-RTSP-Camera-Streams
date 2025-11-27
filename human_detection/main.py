import os
import csv
import yaml
import threading
import cv2
from detector import HumanDetector
from alarm import AlarmManager
from logger import setup_logger
from camera_handler import CameraHandler

# --- Load Config ---
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

logger = setup_logger(cfg["log_folder"])
os.makedirs(cfg["capture_folder"], exist_ok=True)
log_csv_path = os.path.join(cfg["log_folder"], cfg["log_file"])
if not os.path.exists(log_csv_path):
    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "CameraID", "ImageFile"])

# --- Initialize detector and alarm ---
detector = HumanDetector(cfg["yolo_model_path"], cfg["detection_confidence"])
alarm = AlarmManager(cfg["alarm_sound_file"])

# --- Event to stop all cameras ---
stop_event = threading.Event()

# --- Start all cameras ---
camera_threads = []
for cam_cfg in cfg["cameras"]:
    cam_thread = CameraHandler(cam_cfg, cfg, detector, alarm, log_csv_path, logger, stop_event)
    cam_thread.start()
    camera_threads.append(cam_thread)

logger.info("All cameras started. Close any window or press ESC to quit.")

# --- Global quit handling ---
try:
    while not stop_event.is_set():
        key = cv2.waitKey(100)
        if key == 27:  # ESC key
            logger.info("ESC pressed. Exiting...")
            stop_event.set()
            break
except KeyboardInterrupt:
    logger.info("KeyboardInterrupt detected. Exiting...")
    stop_event.set()

# --- Wait for all threads to finish ---
for t in camera_threads:
    t.join()

cv2.destroyAllWindows()
logger.info("All cameras stopped. Program terminated.")
