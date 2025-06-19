import cv2
from app.services.detector import Detector
from app.services.tracker import Tracker

detector = Detector("yolov8x.pt", 0.5)
tracker = Tracker()


def process_video(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {input_path}")

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracked_frame = tracker.track(detections, frame)
        out.write(tracked_frame)

    cap.release()
    out.release()
