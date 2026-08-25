from enum import Enum
import cv2
import numpy as np

from src.datastructures.frame import Frame
from src.datastructures.keyframe import Keyframe
from src.datastructures.map_point import MapPoint
from src.features.feature_matcher import FeatureMatcher
from src.geometry.initialization import TwoFrameInitializer
from src.optimization.bundle_adjustment import BundleAdjustment


class TrackingState(Enum):
    NOT_INITIALIZED = 0
    OK = 1
    LOST = 2


class Tracking:
    """
    Main tracking thread pipeline. Responsible for initialization, constant
    velocity propagation, 3D MapPoint projection matching, PnP pose estimation
    with Motion-Only Bundle Adjustment refinement, and Keyframe creation.
    """

    def __init__(self, K, min_keyframe_matches=5):
        self.K = np.array(K, dtype=np.float32)
        self.state = TrackingState.NOT_INITIALIZED

        # Core modules
        self.matcher = FeatureMatcher(ratio_thresh=0.9, max_hamming_dist=70)
        self.initializer = TwoFrameInitializer(K)
        self.optimizer = BundleAdjustment(K)

        # Frame history & velocity model
        self.current_frame = None
        self.last_frame = None
        self.init_reference_frame = None
        self.reference_keyframe = None
        self.velocity = np.eye(4, dtype=np.float32)

        # Active tracking structures
        self.keyframes = []
        self.map_points = []
        self.min_keyframe_matches = min_keyframe_matches

    def process_frame(self, frame):
        """
        Main entry point for incoming camera frames.
        frame: src.datastructures.frame.Frame
        """
        self.current_frame = frame

        if self.state == TrackingState.NOT_INITIALIZED:
            success = self._initialize_monocular()
            if success:
                self.state = TrackingState.OK
                self.last_frame = self.current_frame
                print(f"[TRACKING] Monocular Initialization Successful! Map Points: {len(self.map_points)}")
        elif self.state == TrackingState.OK:
            tracking_ok = self._track_with_motion_model()

            if not tracking_ok:
                tracking_ok = self._track_reference_keyframe()

            if tracking_ok:
                if self._need_new_keyframe():
                    self._create_new_keyframe()
            else:
                self.state = TrackingState.LOST
                print("[TRACKING WARNING] Tracking Lost!")

            self.last_frame = self.current_frame
        elif self.state == TrackingState.LOST:
            # Try to relocalize against past keyframes
            reloc_ok = self._relocalize()
            if reloc_ok:
                self.state = TrackingState.OK
                print("[TRACKING] Relocalization Successful!")
            self.last_frame = self.current_frame

        return self.state

    def _relocalize(self):
        """Attempts to recover pose by matching against all historical Keyframes."""
        if not self.keyframes or self.current_frame.des is None:
            return False

        # Try keyframes in reverse order (most recent first)
        for kf in reversed(self.keyframes):
            if kf.des is None:
                continue
            matches = self.matcher.match_descriptors(kf.des, self.current_frame.des)
            if len(matches) < 20:
                continue

            # Populate candidate MapPoints from the matching keyframe
            self.current_frame.map_points = [None] * len(self.current_frame.kps)
            for m in matches:
                mp = kf.map_points[m.queryIdx]
                if mp is not None and not getattr(mp, 'is_bad', False):
                    self.current_frame.map_points[m.trainIdx] = mp

            # Attempt PnP solve
            if self._estimate_pose_pnp():
                self.reference_keyframe = kf
                self.velocity = np.eye(4, dtype=np.float32)
                return True

        return False

    def _initialize_monocular(self):
        """Initializes 3D map from two frame views with sufficient parallax."""
        if self.init_reference_frame is None:
            if self.current_frame.des is not None and len(self.current_frame.des) > 50:
                self.init_reference_frame = self.current_frame
            return False

        # Match current frame against the fixed INITIAL reference frame
        matches = self.matcher.match_descriptors(self.init_reference_frame.des, self.current_frame.des)
        print(f"[DEBUG INIT] Match count: {len(matches)}")
        if len(matches) < 40:
            self.init_reference_frame = self.current_frame
            return False

        # Extract matched 2D point coordinates
        pts1 = np.array([self.init_reference_frame.kps[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([self.current_frame.kps[m.trainIdx].pt for m in matches], dtype=np.float32)

        # Check for sufficient parallax (median pixel disparity >= 6.0 px)
        disparities = np.linalg.norm(pts1 - pts2, axis=1)
        median_disparity = np.median(disparities)
        print(f"[DEBUG] Init Phase 1: Matches={len(matches)}, Median Disparity={median_disparity:.2f} px")
        if median_disparity < 3.0:
            return False

        # Compute initial relative pose and triangulate 3D points matching signature
        success, T_21, pts3D, valid_mask = self.initializer.initialize(
            self.init_reference_frame.kps, self.current_frame.kps, matches
        )

        if not success or valid_mask is None or np.sum(valid_mask) < 25:
            return False

        # Set initial frame poses
        self.init_reference_frame.T_cw = np.eye(4, dtype=np.float32)
        self.current_frame.T_cw = T_21.astype(np.float32)

        # Spawn initial Keyframes
        kf1 = Keyframe(self.init_reference_frame)
        kf2 = Keyframe(self.current_frame)
        self.keyframes.extend([kf1, kf2])
        self.reference_keyframe = kf2

        # Instantiate initial MapPoints
        inlier_indices = np.where(valid_mask)[0]
        for inl_idx in inlier_indices:
            p3D = pts3D[inl_idx]
            match = matches[inl_idx]

            mp = MapPoint(p3D, kf1, match.queryIdx)
            mp.add_observation(kf2, match.trainIdx)

            if hasattr(mp, 'update_distinctive_descriptor'):
                mp.update_distinctive_descriptor()

            kf1.map_points[match.queryIdx] = mp
            kf2.map_points[match.trainIdx] = mp
            self.current_frame.map_points[match.trainIdx] = mp

            self.map_points.append(mp)

        n_inliers = int(np.sum(valid_mask))
        kf1.add_connection(kf2, n_inliers)
        kf2.add_connection(kf1, n_inliers)

        # Initial motion velocity
        self.velocity = self.current_frame.T_cw @ np.linalg.inv(self.init_reference_frame.T_cw)
        return True

    def _track_with_motion_model(self):
        """Predicts pose via constant velocity and projects 3D map points into current frame."""
        T_pred = self.velocity @ self.last_frame.T_cw
        self.current_frame.T_cw = T_pred.astype(np.float32)

        valid_mps = [mp for mp in self.last_frame.map_points if mp is not None and not getattr(mp, 'is_bad', False)]
        if len(valid_mps) < 10:
            return False

        pts_3d = np.array([mp.pos for mp in valid_mps], dtype=np.float32)
        R_pred = self.current_frame.T_cw[:3, :3]
        t_pred = self.current_frame.T_cw[:3, 3]

        pts3D_cam = (R_pred @ pts_3d.T).T + t_pred
        in_front = pts3D_cam[:, 2] > 0.1
        if np.sum(in_front) < 10:
            return False

        proj = (self.K @ pts3D_cam.T).T
        proj_2d = proj[:, :2] / proj[:, 2:]

        curr_kps_pts = np.float32([kp.pt for kp in self.current_frame.kps])
        search_radius = 40.0

        for i, mp in enumerate(valid_mps):
            if not in_front[i]:
                continue

            target_pt = proj_2d[i]
            diffs = np.linalg.norm(curr_kps_pts - target_pt, axis=1)
            candidate_indices = np.where(diffs < search_radius)[0]

            if len(candidate_indices) == 0:
                continue

            cand_des = self.current_frame.des[candidate_indices]
            dists = FeatureMatcher.compute_hamming_distances_vectorized(mp.distinctive_des, cand_des)
            best_idx = np.argmin(dists)

            if dists[best_idx] < self.matcher.max_hamming_dist:
                match_kp_idx = candidate_indices[best_idx]
                self.current_frame.map_points[match_kp_idx] = mp

        return self._estimate_pose_pnp()

    def _track_reference_keyframe(self):
        """Fallback tracking against the nearest Reference Keyframe."""
        if self.reference_keyframe is None:
            return False

        matches = self.matcher.match_descriptors(self.reference_keyframe.des, self.current_frame.des)
        if len(matches) < 15:
            return False

        for m in matches:
            mp = self.reference_keyframe.map_points[m.queryIdx]
            if mp is not None and not getattr(mp, 'is_bad', False):
                self.current_frame.map_points[m.trainIdx] = mp

        return self._estimate_pose_pnp()

    def _estimate_pose_pnp(self):
        """PnP RANSAC outlier filtering followed by Motion-Only Bundle Adjustment."""
        pts_3d = []
        pts_2d = []
        indices = []

        for idx, mp in enumerate(self.current_frame.map_points):
            if mp is not None and not getattr(mp, 'is_bad', False):
                pts_3d.append(mp.pos)
                pts_2d.append(self.current_frame.kps[idx].pt)
                indices.append(idx)

        if len(pts_3d) < 10:
            return False

        pts_3d = np.float32(pts_3d)
        pts_2d = np.float32(pts_2d)

        # 1. Solve initial pose using EPNP + RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d,
            pts_2d,
            self.K,
            distCoeffs=None,
            iterationsCount=150,
            reprojectionError=4.0,
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success or inliers is None or len(inliers) < 10:
            return False

        R, _ = cv2.Rodrigues(rvec)
        self.current_frame.T_cw[:3, :3] = R.astype(np.float32)
        self.current_frame.T_cw[:3, 3] = tvec.squeeze().astype(np.float32)

        # Null out outlier MapPoints
        inlier_set = set(inliers.ravel())
        for i, idx in enumerate(indices):
            if i not in inlier_set:
                self.current_frame.map_points[idx] = None

        # 2. Refine 6-DoF pose using Motion-Only Bundle Adjustment
        self.optimizer.motion_only_ba(self.current_frame, max_iterations=15)

        # Update Motion Model Velocity
        if self.last_frame is not None:
            T_last_inv = np.eye(4, dtype=np.float32)
            R_last_T = self.last_frame.T_cw[:3, :3].T
            T_last_inv[:3, :3] = R_last_T
            T_last_inv[:3, 3] = -R_last_T @ self.last_frame.T_cw[:3, 3]
            self.velocity = self.current_frame.T_cw @ T_last_inv

        return True

    def _need_new_keyframe(self):
        """Determines whether current frame should become a Keyframe."""
        if self.reference_keyframe is None:
            return True
        n_tracked_mp = sum(1 for mp in self.current_frame.map_points if mp is not None and not getattr(mp, 'is_bad', False))
        n_ref_mp = sum(1 for mp in self.reference_keyframe.map_points if mp is not None and not getattr(mp, 'is_bad', False))
        if n_ref_mp == 0:
            return True
        return (n_tracked_mp < 0.3 * n_ref_mp) and (n_tracked_mp > 10)

    def _create_new_keyframe(self):
        """Converts current frame into a keyframe."""
        kf = Keyframe(self.current_frame)
        self.keyframes.append(kf)
        self.reference_keyframe = kf

        for idx, mp in enumerate(self.current_frame.map_points):
            if mp is not None and not getattr(mp, 'is_bad', False):
                mp.add_observation(kf, idx)

        frame_id = getattr(self.current_frame, 'id', 'N/A')
        print(f"[TRACKING] Keyframe #{kf.id} Inserted (from Frame #{frame_id}).")