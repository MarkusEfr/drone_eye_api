import time
import cv2
from app.services.detector import Detector
from app.services.tracker import Tracker


def process_video(
    input_path: str, output_path: str, detector: Detector, tracker: Tracker
):
    video_in = cv2.VideoCapture(input_path)
    if not video_in.isOpened():
        raise ValueError(f"❌ Cannot open video file: {input_path}")

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    fps = video_in.get(cv2.CAP_PROP_FPS)
    width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    prev_time = time.time()

    while True:
        success, frame = video_in.read()
        if not success:
            break

        detections = detector.detect(frame)

        current_time = time.time()
        frame_fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        tracked_frame, _ = tracker.track(
            detections, frame, return_tracking_data=True, show_fps=frame_fps
        )

        video_out.write(tracked_frame)

    video_in.release()
    video_out.release()
    print(f"✅ Tracking complete. Output saved to: {output_path}")
