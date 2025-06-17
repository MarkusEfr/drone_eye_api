from ultralytics import YOLO
import cv2


class Detector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names

    def _get_filtered_boxes(self, frame):
        results = self.model(frame, verbose=False)[0]
        return [
            {
                "conf": float(box.conf[0]),
                "cls_id": int(box.cls[0]),
                "xyxy": box.xyxy[0].cpu().numpy(),
            }
            for box in results.boxes
            if float(box.conf[0]) >= self.conf_threshold
        ]

    def detect(self, frame):
        detections = []
        for item in self._get_filtered_boxes(frame):
            x1, y1, x2, y2 = item["xyxy"]
            label = self.class_names[item["cls_id"]]
            bbox = [x1, y1, x2 - x1, y2 - y1]
            detections.append(
                {
                    "label": label,
                    "confidence": round(item["conf"], 2),
                    "bbox": [round(float(x), 2) for x in bbox],
                }
            )
        return detections

    def detect_and_draw(self, frame):
        for item in self._get_filtered_boxes(frame):
            x1, y1, x2, y2 = item["xyxy"].astype(int)
            label = f"{self.class_names[item['cls_id']]} {item['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        return frame
