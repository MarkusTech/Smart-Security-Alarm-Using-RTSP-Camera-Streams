import cv2
import threading

class Camera:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise Exception("Camera not detected")
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self.update_frames, daemon=True).start()

    def update_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.cap.release()
