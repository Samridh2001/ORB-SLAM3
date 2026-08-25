import numpy as np
import cv2

from src.features.feature_matcher import FeatureMatcher
from src.optimization.bundle_adjustment import BundleAdjustment

class LoopClosing:
    """
    Loop Closing thread for ORB-SLAM3.
    Detects place recognition loops, computes 7-DoF Sim(3) alignment via Horn's method,
    corrects intra-map loops, and delegates map merging to Atlas.
    """

    def __init__(self, atlas, K, min_inliers=20):
        self.atlas = atlas
        self.K = np.array(K, dtype=np.float32)
        self.matcher = FeatureMatcher(ratio_thresh=0.75, max_hamming_dist=50)
        self.optimizer = BundleAdjustment(K)
        self.min_inliers = min_inliers
        self.loop_queue = []
    
    def add_keyframe(self, keyframe):
        """Adds a keyframe to the loop detection queue."""
        self.loop_queue.append(keyframe)

    def process(self):
        """
        Main execution step.
        Pops the next keyframe from the queue and searches for loop closures.
        Returns: bool indicating whether a loop or merge was executed.
        """
        if not self.loop_queue:
            return False
        current_kf = self.loop_queue.pop(0)
        if getattr(current_kf, 'is_bad', False):
            return False
        # 1. Detect loop candidate keyframes across active and inactive maps
        candidate_kf, candidate_map = self._detect_loop_candidates(current_kf)
        if candidate_kf is None:
            return False
        # 2. Match 3D map points between current KF and candidate KF
        matches, pts3d_curr, pts3d_cand = self._find_3d_correspondences(current_kf, candidate_kf)
        if len(matches) < self.min_inliers:
            return False

        # 3. Estimate 7-DoF Sim(3) relative pose via Horn's alignment with RANSAC
        success, S_cand_curr, inliers = self._compute_sim3_ransac(pts3d_curr, pts3d_cand)
        if not success or len(inliers) < self.min_inliers:
            return False

        # 4. Intra-Map Loop vs. Inter-Map Merge
        active_map = self.atlas.get_active_map()
        if candidate_map == active_map:
            print(f"[LOOP CLOSING] Loop detected in active map between KF #{current_kf.id} and KF #{candidate_kf.id}")
            self._correct_loop(current_kf, candidate_kf, matches, inliers)
        else:
            print(f"[LOOP CLOSING] Map merge triggered between Active Map #{active_map.id} and Map #{candidate_map.id}")
            self._merge_maps(active_map, candidate_map, current_kf, candidate_kf, S_cand_curr, matches, inliers)

        return True

    def _detect_loop_candidates(self, current_kf):
        """Finds the best historical keyframe candidate."""
        covisible_neighbors = set(current_kf.connected_keyframes.keys())
        covisible_neighbors.add(current_kf)

        best_candidate = None
        best_candidate_map = None
        max_matches = 0

        for submap in self.atlas.get_all_maps():
            for kf in submap.get_keyframes():
                if kf in covisible_neighbors or getattr(kf, 'is_bad', False):
                    continue

                matches = self.matcher.match_descriptors(current_kf.des, kf.des)
                if len(matches) > max_matches and len(matches) >= self.min_inliers:
                    max_matches = len(matches)
                    best_candidate = kf
                    best_candidate_map = submap

        return best_candidate, best_candidate_map

    def _find_3d_correspondences(self, kf1, kf2):
        """Extracts 3D-3D point pairs for matching 2D features between two keyframes."""
        matches = self.matcher.match_descriptors(kf1.des, kf2.des)
        pts3d_1, pts3d_2 = [], []
        valid_matches = []

        for m in matches:
            mp1 = kf1.map_points[m.queryIdx]
            mp2 = kf2.map_points[m.trainIdx]

            if mp1 is not None and mp2 is not None and not getattr(mp1, 'is_bad', False) and not getattr(mp2, 'is_bad', False):
                pts3d_1.append(mp1.pos)
                pts3d_2.append(mp2.pos)
                valid_matches.append(m)

        return valid_matches, np.array(pts3d_1, dtype=np.float32), np.array(pts3d_2, dtype=np.float32)

    def _compute_sim3_ransac(self, pts1, pts2, max_iterations=100, inlier_thresh=0.15):
        """Computes 7-DoF Sim(3) transformation S_21 aligning pts1 to pts2 using Horn's method + RANSAC."""
        n_pts = len(pts1)
        if n_pts < 4:
            return False, None, []

        best_inliers = []
        best_S = None

        for _ in range(max_iterations):
            sample_idx = np.random.choice(n_pts, 3, replace=False)
            p1_sample = pts1[sample_idx]
            p2_sample = pts2[sample_idx]

            S_candidate = self._horn_sim3(p1_sample, p2_sample)
            if S_candidate is None:
                continue

            s = np.cbrt(np.linalg.det(S_candidate[:3, :3]))
            R = S_candidate[:3, :3] / s
            t = S_candidate[:3, 3]

            pts1_trans = (s * (R @ pts1.T)).T + t
            errors = np.linalg.norm(pts1_trans - pts2, axis=1)

            inliers = np.where(errors < inlier_thresh)[0]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_S = S_candidate

            if len(best_inliers) > 0.8 * n_pts:
                break

        if len(best_inliers) < self.min_inliers:
            return False, None, []

        refined_S = self._horn_sim3(pts1[best_inliers], pts2[best_inliers])
        return True, refined_S if refined_S is not None else best_S, best_inliers

    @staticmethod
    def _horn_sim3(pts1, pts2):
        """Closed-form 7-DoF absolute orientation solution with scale (Horn's Method)."""
        n = len(pts1)
        if n < 3:
            return None

        mu1 = np.mean(pts1, axis=0)
        mu2 = np.mean(pts2, axis=0)

        p1_cent = pts1 - mu1
        p2_cent = pts2 - mu2

        var1 = np.sum(p1_cent ** 2)
        if var1 < 1e-7:
            return None

        H = p1_cent.T @ p2_cent / n
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        s = np.trace(p2_cent.T @ (p1_cent @ R.T)) / var1
        if s <= 1e-4:
            s = 1.0

        t = mu2 - s * (R @ mu1)

        S = np.eye(4, dtype=np.float32)
        S[:3, :3] = (s * R).astype(np.float32)
        S[:3, 3] = t.astype(np.float32)
        return S

    def _correct_loop(self, current_kf, candidate_kf, matches, inliers):
        """Corrects intra-map loop by fusing duplicate points and running Global BA."""
        # 1. Fuse duplicate MapPoints
        for inlier_idx in inliers:
            m = matches[inlier_idx]
            mp_curr = current_kf.map_points[m.queryIdx]
            mp_cand = candidate_kf.map_points[m.trainIdx]

            if mp_curr is not None and mp_cand is not None and mp_curr != mp_cand:
                for obs_kf, kp_idx in list(mp_curr.observations.items()):
                    obs_kf.map_points[kp_idx] = mp_cand
                    mp_cand.add_observation(obs_kf, kp_idx)
                mp_curr.is_bad = True

        # 2. Global BA across active map
        active_map = self.atlas.get_active_map()
        print(f"[LOOP CLOSING] Running Global BA on Active Map #{active_map.id}...")
        self.optimizer.global_bundle_adjustment(
            keyframes=active_map.get_keyframes(),
            map_points=active_map.get_map_points(),
            max_iterations=25
        )

    def _merge_maps(self, active_map, matched_map, current_kf, candidate_kf, S_matched_active, matches, inliers):
        """Delegates complete map merging (transform + point fusion + Welded BA) to Atlas."""
        self.atlas.merge_maps(
            target_map=matched_map,
            source_map=active_map,
            S_target_source=S_matched_active,
            current_kf=current_kf,
            candidate_kf=candidate_kf,
            matches=matches,
            inliers=inliers
        )

        # Global BA on the unified map
        print(f"[LOOP CLOSING] Running Global BA across unified Map #{matched_map.id}...")
        self.optimizer.global_bundle_adjustment(
            keyframes=matched_map.get_keyframes(),
            map_points=matched_map.get_map_points(),
            max_iterations=25
        )