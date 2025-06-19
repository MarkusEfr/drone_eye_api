from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2


class Tracker:
    def __init__(self, max_age=30, n_init=1):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    def track(self, detections, frame):
        results = self.tracker.update_tracks(
            [(d["bbox"], d["confidence"], d["label"]) for d in detections], frame=frame
        )

        for track in results:
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        return frame
