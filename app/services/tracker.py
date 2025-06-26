from deep_sort_realtime.deepsort_tracker import DeepSort
from app.services.painter import Painter


class Tracker:
    def __init__(
        self,
        max_track_lifetime=60,
        min_confirmed_frames=2,
        max_trail_length=30,
        allowed_labels=None,
    ):
        self.deep_sort = DeepSort(
            max_age=max_track_lifetime, n_init=min_confirmed_frames
        )
        self.motion_trails = {}  # track_id → list of center points
        self.max_trail_length = max_trail_length
        self.allowed_labels = allowed_labels or {"car", "truck", "person"}
        self.painter = Painter()

    def track(self, detections, frame, return_tracking_data=False, show_fps=None):
        # Prepare input for DeepSort
        trackable_detections = [
            (det["bbox"], det["confidence"], det["label"])
            for det in detections
            if det["label"] in self.allowed_labels
        ]

        updated_tracks = self.deep_sort.update_tracks(trackable_detections, frame=frame)
        tracking_results = []

        for track in updated_tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            label_name = track.get_det_class()
            left, top, right, bottom = map(int, track.to_ltrb())
            center_x, center_y = (left + right) // 2, (top + bottom) // 2

            # Draw bounding box with label and ID using Painter
            self.painter.draw_bbox(
                frame,
                bbox=[left, top, right, bottom],
                label=label_name,
                track_id=track_id,
            )

            # Draw trail for this object
            self.motion_trails.setdefault(track_id, []).append((center_x, center_y))
            if len(self.motion_trails[track_id]) > self.max_trail_length:
                self.motion_trails[track_id].pop(0)

            self.painter.draw_trail(frame, self.motion_trails[track_id])

            tracking_results.append(
                {
                    "id": track_id,
                    "label": label_name,
                    "bbox": [left, top, right, bottom],
                }
            )

        # Optional FPS overlay
        if show_fps is not None:
            self.painter.draw_fps(frame, show_fps)

        return (frame, tracking_results) if return_tracking_data else frame
