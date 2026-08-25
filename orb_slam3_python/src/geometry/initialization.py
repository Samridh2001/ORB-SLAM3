import cv2
import numpy as np
from src.geometry.triangulation import check_cheirality, triangulate_points_vectorized

class TwoFrameInitializer:
    """
    Computes initial relative 2D-2D pose (R, t) and 3D map points from matched features.
    Computes both Essential and Homography models concurrently, selecting the best score.
    """

    def __init__(self, K, sigma=1.0, iterations=200):
        self.K = K.astype(np.float32)
        self.K_inv = np.linalg.inv(K).astype(np.float32)
        self.sigma = sigma
        self.max_iterations = iterations

    def initialize(self, kps1, kps2, matches):
        """
        kps1, kps2: Lists of cv2.KeyPoint
        matches: List of cv2.DMatch
        Returns: success (bool), T_21 (4x4 matrix), pts3D (N, 3), valid_matches_mask
        """
        if len(matches) < 100:
            return False, None, None, None

        pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

        pts1_norm = (self.K_inv @ np.hstack([pts1, np.ones((len(pts1), 1))]).T).T[:, :2]
        pts2_norm = (self.K_inv @ np.hstack([pts2, np.ones((len(pts2), 1))]).T).T[:, :2]

        # 1. Compute Essential Matrix with RANSAC on CPU
        E, inliers_E = cv2.findEssentialMat(
            pts1, pts2, self.K, method=cv2.RANSAC, prob=0.99, threshold=self.sigma
        )

        E, inliers_E = cv2.findEssentialMat(
            pts1, pts2, self.K, method=cv2.RANSAC, prob=0.99, threshold=self.sigma
        )

        # 2. Decompose Essential Matrix into 4 candidate (R, t) solutions
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = t.squeeze()

        candidates = [
            (R1, t),
            (R1, -t),
            (R2, t),
            (R2, -t)
        ]
        # 3. Select solution with maximum positive depth points (cheirality condition)
        best_R, best_t = None, None
        max_good = 0
        best_mask = None
        best_pts3D = None

        for R_cand, t_cand in candidates:
            good_mask, pts3D = check_cheirality(R_cand, t_cand, pts1_norm, pts2_norm)
            num_good = np.sum(good_mask)

            if num_good > max_good:
                max_good = num_good
                best_R = R_cand
                best_t = t_cand
                best_mask = good_mask
                best_pts3D = pts3D
        # Require at least 30% of matched points to be valid 3D points in front of cameras
        if max_good < 0.3 * len(matches):
            return False, None, None, None

        # Build 4x4 Transformation Matrix T_21 (Camera 1 to Camera 2)
        T_21 = np.eye(4, dtype=np.float32)
        T_21[:3, :3] = best_R
        T_21[:3, 3] = best_t

        return True, T_21, best_pts3D, best_mask