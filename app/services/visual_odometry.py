import cv2
import numpy as np


class VisualOdometry:
    def __init__(self, focal_length, principal_point):
        self.orb = cv2.ORB.create(nfeatures=3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.focal = focal_length
        self.pp = principal_point

        self.prev_kp = None
        self.prev_des = None
        self.prev_frame = None
        self.cur_pose = np.eye(4)
        self.trajectory = [np.zeros(3)]

    def process_frame(self, frame_gray):
        # 1. Detect keypoints and descriptors
        mask = np.ones_like(frame_gray, dtype=np.uint8)
        kp, des = self.orb.detectAndCompute(frame_gray, mask)

        if self._is_first_frame(des):
            self._update_reference(kp, des, frame_gray)
            return self.cur_pose[:3, 3], self.trajectory

        # 2. Match descriptors
        matches = self._get_matches(des)
        if len(matches) < 8:
            self._update_reference(kp, des, frame_gray)
            return self.cur_pose[:3, 3], self.trajectory

        # 3. Extract matched keypoints
        pts1, pts2 = self._extract_matched_points(matches, self.prev_kp, kp)

        # 4. Estimate motion
        E, _ = cv2.findEssentialMat(pts2, pts1, focal=self.focal, pp=self.pp)
        _, R, t, _ = cv2.recoverPose(E, pts2, pts1, focal=self.focal, pp=self.pp)

        # 5. Update pose
        self._update_pose(R, t)

        # 6. Update reference frame and keypoints
        self._update_reference(kp, des, frame_gray)

        return self.cur_pose[:3, 3], self.trajectory

    def _is_first_frame(self, des):
        return (
            self.prev_frame is None
            or des is None
            or self.prev_des is None
            or len(des) < 2
            or len(self.prev_des) < 2
        )

    def _get_matches(self, des):
        if self.prev_des is None or des is None:
            return []

        if len(self.prev_des) < 2 or len(des) < 2:
            return []

        return self.bf.knnMatch(self.prev_des, des, k=2)

    def _extract_matched_points(self, matches, kp1, kp2):
        valid = [m for m in matches if m.queryIdx < len(kp1) and m.trainIdx < len(kp2)]
        pts1 = np.array([kp1[m.queryIdx].pt for m in valid], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in valid], dtype=np.float32)
        return pts1, pts2

    def _update_pose(self, R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()
        self.cur_pose = self.cur_pose @ np.linalg.inv(T)
        self.trajectory.append(self.cur_pose[:3, 3].copy())

    def _update_reference(self, kp, des, frame):
        self.prev_kp = kp
        self.prev_des = des
        self.prev_frame = frame
