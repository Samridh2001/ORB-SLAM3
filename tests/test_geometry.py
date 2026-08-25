import os
import sys
import numpy as np
import cv2

# Dynamic path resolution to avoid setting PYTHONPATH manually
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.geometry.initialization import TwoFrameInitializer
from src.geometry.triangulation import triangulate_points_vectorized

def test_geometry():
    # Intrinsic Matrix K
    K = np.array([[800, 0, 320],
                  [0, 800, 240],
                  [0,   0,   1]], dtype=np.float32)

    # Synthetic Ground Truth 3D Points
    np.random.seed(42)
    gt_pts3D = np.random.uniform(-2, 2, (150, 3)).astype(np.float32)
    gt_pts3D[:, 2] += 5.0  # Move in front of camera (Z > 0)

    # True Motion: Move 0.5m along X-axis
    T_gt = np.eye(4, dtype=np.float32)
    T_gt[0, 3] = 0.5

    # Project to Frame 1 and Frame 2
    proj1 = (K @ gt_pts3D.T).T
    pts1 = proj1[:, :2] / proj1[:, 2:]

    pts3D_f2 = (gt_pts3D - T_gt[:3, 3])
    proj2 = (K @ pts3D_f2.T).T
    pts2 = proj2[:, :2] / proj2[:, 2:]

    # Mock cv2.KeyPoint objects and DMatches
    kps1 = [cv2.KeyPoint(p[0], p[1], 1.0) for p in pts1]
    kps2 = [cv2.KeyPoint(p[0], p[1], 1.0) for p in pts2]
    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(pts1))]

    # Run Initializer
    initializer = TwoFrameInitializer(K)
    success, T_est, pts3D_est, mask = initializer.initialize(kps1, kps2, matches)

    print(f"[TEST GEOMETRY SUCCESS]: {success}")
    if success:
        print("Estimated Translation Vector (Normalized scale):\n", T_est[:3, 3])
        print(f"Valid Triangulated Points: {np.sum(mask)} / {len(gt_pts3D)}")

if __name__ == "__main__":
    test_geometry()