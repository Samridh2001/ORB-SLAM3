import os
import sys
import numpy as np
import cv2

# Dynamic path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datastructures.frame import Frame
from src.features.feature_matcher import FeatureMatcher


def test_matcher():
    np.random.seed(42)

    # 1. Create mock keypoints and descriptors for Frame 1
    kps1 = [cv2.KeyPoint(x=10.0 * i, y=10.0 * i, size=31) for i in range(20)]
    des1 = np.random.randint(0, 255, (20, 32), dtype=np.uint8)

    # 2. Create Frame 2 with small spatial shift (simulating smooth camera movement)
    kps2 = [cv2.KeyPoint(x=10.0 * i + 2.0, y=10.0 * i + 2.0, size=31) for i in range(20)]
    des2 = np.copy(des1)

    # Mutate 50% of Frame 2 descriptors to simulate noise/texture changes
    des2[10:] = np.random.randint(0, 255, (10, 32), dtype=np.uint8)

    frame1 = Frame(kps1, des1)
    frame2 = Frame(kps2, des2)

    matcher = FeatureMatcher(ratio_thresh=0.8, max_hamming_dist=50)

    # Test 1: Standard Lowe's Ratio Matching
    matches_knn = matcher.match_descriptors(frame1.des, frame2.des)
    print(f"[TEST KNN MATCHING]: {len(matches_knn)} matches found.")

    # Test 2: Window-Guided Matching
    matches_guided = matcher.match_window_guided(frame1, frame2, window_size=15)
    print(f"[TEST GUIDED MATCHING]: {len(matches_guided)} matches found.")

    assert len(matches_knn) > 0, "KNN Matching failed!"
    assert len(matches_guided) > 0, "Guided Matching failed!"
    print("[TEST MATCHER SUCCESS]")


if __name__ == "__main__":
    test_matcher()