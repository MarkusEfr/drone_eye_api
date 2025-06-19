from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, Response, FileResponse
import cv2
import numpy as np
import os
import uuid

from app.services.detector import Detector
from app.services.tracker import Tracker
from app.services.video_processor import process_video

router = APIRouter()
detector = Detector("yolov8x.pt", 0.5)
tracker = Tracker()


def read_image_from_upload(file: UploadFile):
    image_bytes = file.file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


@router.post("/detect/")
async def detect(file: UploadFile = File(...)):
    image = read_image_from_upload(file)
    results = detector.detect(image)
    return JSONResponse(content={"detections": results})


@router.post("/detect/annotated")
async def detect_annotated(file: UploadFile = File(...)):
    image = read_image_from_upload(file)
    frame = detector.detect_and_draw(image)
    _, encoded_image = cv2.imencode(".jpg", frame)
    return Response(content=encoded_image.tobytes(), media_type="image/jpeg")


@router.post("/track/video")
async def track_video(file: UploadFile = File(...)):
    filename = file.filename or "input.mp4"
    file_ext = filename.split(".")[-1]
    video_id = str(uuid.uuid4())
    input_path = f"/tmp/{video_id}_in.{file_ext}"
    output_path = f"/tmp/{video_id}_out.mp4"

    with open(input_path, "wb") as f:
        f.write(file.file.read())

    process_video(input_path, output_path)
    return FileResponse(
        output_path, media_type="video/mp4", filename="tracked_output.mp4"
    )
