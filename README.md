# Drone Eye API

API приложение на FastAPI с YOLOv8 для детекции объектов

## Запуск
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
uvicorn app.main:app --reload
```
