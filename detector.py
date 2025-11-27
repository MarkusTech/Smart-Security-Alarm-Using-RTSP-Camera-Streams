from ultralytics import YOLO

class HumanDetector:
    def __init__(self, model_path, confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        humans = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf >= self.confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    humans.append((x1, y1, x2, y2))
        return humans
