from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, Response
import cv2
import numpy as np
from app.services.detector import Detector

router = APIRouter()
detector = Detector("yolov8x.pt", 0.5)


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
