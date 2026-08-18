import cv2
import numpy as np


class FeatureMatcher:
    """
    CPU-optimized Feature Matcher using OpenCV BFMatcher (Hamming distance)
    and vectorized spatial grid projection filters.
    """

    def __init__(self, ratio_thresh=0.75, max_hamming_dist=50):
        self.ratio_thresh = ratio_thresh
        self.max_hamming_dist = max_hamming_dist
        
        # Binary descriptors (ORB) must use Hamming Norm
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match_descriptors(self, des1, des2):
        """
        Standard 2-NN matching with Lowe's ratio test.
        des1, des2: uint8 numpy arrays of shape (N, 32) and (M, 32)
        Returns: list of cv2.DMatch
        """
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []

        # Find top 2 nearest neighbors for Lowe's ratio test
        knn_matches = self.bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            
            # Lowe's ratio test & absolute distance threshold filter
            if m.distance < self.ratio_thresh * n.distance and m.distance < self.max_hamming_dist:
                good_matches.append(m)

        return good_matches

    def match_window_guided(self, frame1, frame2, window_size=30):
        """
        Guided matching: Matches keypoints in frame1 to keypoints in frame2 
        ONLY within a local spatial window (pixel radius) around their 2D coordinates.
        Extremely fast for CPU frame-to-frame tracking with low latency.
        """
        des1, des2 = frame1.des, frame2.des
        kps1, kps2 = frame1.kps, frame2.kps

        if des1 is None or des2 is None or len(kps1) == 0 or len(kps2) == 0:
            return []

        pts2 = np.float32([kp.pt for kp in kps2])  # (M, 2)
        matches = []

        # Vectorized spatial searching on CPU
        for idx1, kp1 in enumerate(kps1):
            pt1 = np.array(kp1.pt, dtype=np.float32)

            # Spatial window query using fast Euclidean bounds
            diffs = np.abs(pts2 - pt1)
            in_window = (diffs[:, 0] < window_size) & (diffs[:, 1] < window_size)
            candidate_indices = np.where(in_window)[0]

            if len(candidate_indices) == 0:
                continue

            # Compute Hamming distances only to candidates within the window
            cand_des = des2[candidate_indices]
            dists = self.compute_hamming_distances_vectorized(des1[idx1], cand_des)

            # Find best and second best match in local window
            if len(dists) == 1:
                if dists[0] < self.max_hamming_dist:
                    # Explicit float cast and positional args for cv2.DMatch
                    matches.append(cv2.DMatch(idx1, int(candidate_indices[0]), float(dists[0])))
            else:
                sorted_idxs = np.argsort(dists)
                best_idx = sorted_idxs[0]
                second_idx = sorted_idxs[1]

                if dists[best_idx] < self.ratio_thresh * dists[second_idx] and dists[best_idx] < self.max_hamming_dist:
                    # Explicit float cast and positional args for cv2.DMatch
                    matches.append(cv2.DMatch(idx1, int(candidate_indices[best_idx]), float(dists[best_idx])))

        return matches

    @staticmethod
    def compute_hamming_distances_vectorized(query_des, candidate_des):
        """
        Computes Hamming distances between 1 query descriptor and (N, 32) candidates
        using vectorized bitwise XOR and bit-counting on CPU.
        """
        xor_result = np.bitwise_xor(query_des, candidate_des)
        # Fast bitwise unpack along uint8 bytes
        return np.unpackbits(xor_result, axis=1).sum(axis=1)