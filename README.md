# Smart Security Alarm Using RTSP Camera Streams

This project detects humans in real-time from an RTSP camera stream and triggers an alarm when a person is detected. It uses YOLOv8 for detection and plays an alarm sound through your PC speakers.

---

## Features

- Real-time human detection from RTSP camera streams
- Bounding boxes drawn around detected humans
- Alarm triggers only when a human appears (cooldown applied)
- Minimal memory usage with resized frames
- Compatible with laptop cameras for testing

---

## Requirements

- Python 3.9+
- Windows / Linux / MacOS
- RTSP-enabled IP camera (like Dahua, Hikvision, etc.)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/MarkusTech/Smart-Security-Alarm-Using-RTSP-Camera-Streams.git
cd Smart-Security-Alarm-Using-RTSP-Camera-Streams
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```
3. Rup App
 ``` bash
python main.py
 ```
4. Run Test

```bash
cd test
python main.py
```
