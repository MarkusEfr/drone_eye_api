from fastapi import APIRouter, File, UploadFile, Depends
from fastapi.responses import FileResponse

import uuid

from app.services.detector import Detector
from app.services.tracker import Tracker
from app.services.video_processor import process_video

router = APIRouter()

# Singleton-like cache
_cached_detector = None
_cached_tracker = None


def get_detector() -> Detector:
    global _cached_detector
    if _cached_detector is None:
        _cached_detector = Detector(
            model_path="yolov8x.pt",
            confidence_threshold=0.5,
            allowed_labels=["car", "truck", "bus", "person"],
        )
    return _cached_detector


def get_tracker() -> Tracker:
    global _cached_tracker
    if _cached_tracker is None:
        _cached_tracker = Tracker(
            max_track_lifetime=60,
            min_confirmed_frames=2,
            max_trail_length=30,
            allowed_labels={"car", "truck", "person"},
        )
    return _cached_tracker


@router.post("/track/video")
async def track_video(
    file: UploadFile = File(...),
    detector: Detector = Depends(get_detector),
    tracker: Tracker = Depends(get_tracker),
):
    filename = file.filename or "input.mp4"
    file_ext = filename.split(".")[-1]
    video_id = str(uuid.uuid4())
    input_path = f"/tmp/{video_id}_in.{file_ext}"
    output_path = f"/tmp/{video_id}_out.mp4"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    process_video(input_path, output_path, detector=detector, tracker=tracker)

    return FileResponse(
        output_path, media_type="video/mp4", filename="tracked_output.mp4"
    )
