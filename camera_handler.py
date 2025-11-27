import os
import cv2
import time
import csv
import threading

class CameraHandler(threading.Thread):
    def __init__(self, camera_config, cfg, detector, alarm, log_csv_path, logger, stop_event):
        super().__init__()
        self.index = camera_config["index"]
        self.camera_id = camera_config["id"]
        self.cfg = cfg
        self.detector = detector
        self.alarm = alarm
        self.log_csv_path = log_csv_path
        self.logger = logger
        self.stop_event = stop_event

        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            self.logger.error(f"Camera {self.camera_id} not detected")
            raise Exception(f"Camera {self.camera_id} not detected")

        self.last_alarm_time = 0
        self.human_present = False
        self.detection_counter = 0
        self.frame_count = 0

    def run(self):
        self.logger.info(f"Camera {self.camera_id} started")
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue

            self.frame_count += 1
            if self.frame_count % self.cfg["skip_frames"] != 0:
                continue

            small_frame = cv2.resize(frame, (0,0), fx=self.cfg["frame_downscale"], fy=self.cfg["frame_downscale"])
            humans = self.detector.detect(small_frame)
            humans = [
                (
                    int(x1 / self.cfg["frame_downscale"]),
                    int(y1 / self.cfg["frame_downscale"]),
                    int(x2 / self.cfg["frame_downscale"]),
                    int(y2 / self.cfg["frame_downscale"])
                )
                for (x1, y1, x2, y2) in humans
                if (y2 - y1) >= self.cfg["min_box_height"]
            ]

            human_detected = len(humans) > 0

            # Draw boxes
            for (x1, y1, x2, y2) in humans:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, f"Person {self.camera_id}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S"),
                        (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Detection logic
            self.detection_counter = self.detection_counter + 1 if human_detected else 0

            if self.detection_counter >= self.cfg["detection_frames_required"] and not self.human_present:
                current_time = time.time()
                if (current_time - self.last_alarm_time) > self.cfg["alarm_cooldown"]:
                    self.logger.info(f"HUMAN DETECTED! [{self.camera_id}]")
                    self.alarm.play_alarm()
                    self.last_alarm_time = current_time

                    # Capture cropped images
                    for (x1, y1, x2, y2) in humans:
                        cropped = frame[y1:y2, x1:x2]
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(self.cfg["capture_folder"], f"{self.camera_id}_human_{timestamp}.jpg")
                        cv2.imwrite(filename, cropped)
                        self.logger.info(f"Cropped image saved: {filename}")
                        with open(self.log_csv_path, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([timestamp, self.camera_id, filename])

                self.human_present = True
            elif self.detection_counter == 0:
                self.human_present = False

            cv2.imshow(f"Camera {self.camera_id}", frame)

            # Check if window was closed
            if cv2.getWindowProperty(f"Camera {self.camera_id}", cv2.WND_PROP_VISIBLE) < 1:
                self.logger.info(f"Window closed for {self.camera_id}, stopping thread.")
                self.stop_event.set()
                break

            # Small delay to reduce CPU
            cv2.waitKey(1)

        self.cap.release()
        cv2.destroyWindow(f"Camera {self.camera_id}")
        self.logger.info(f"Camera {self.camera_id} stopped")
