import os
import sys
import numpy as np
import cv2

# Dynamic path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datastructures.frame import Frame
from src.datastructures.keyframe import Keyframe
from src.datastructures.map_point import MapPoint
from src.optimization.bundle_adjustment import BundleAdjustment


def test_bundle_adjustment():
    K = np.array([[800, 0, 320],
                  [0, 800, 240],
                  [0,   0,   1]], dtype=np.float32)

    ba = BundleAdjustment(K)

    # 1. Generate 3D ground truth scene (30 points)
    np.random.seed(42)
    pts3D_gt = np.random.uniform(-1.5, 1.5, (30, 3)).astype(np.float32)
    pts3D_gt[:, 2] += 4.5

    # 2. Camera poses: KF0 at origin, KF1 and KF2 along X-axis
    T0 = np.eye(4, dtype=np.float32)
    T1 = np.eye(4, dtype=np.float32); T1[0, 3] = -0.3
    T2 = np.eye(4, dtype=np.float32); T2[0, 3] = -0.6

    # Project ground truth points into each frame
    kps0, kps1, kps2 = [], [], []
    for P in pts3D_gt:
        p0 = (K @ P)[:2] / P[2]
        p1_c = T1[:3, :3] @ P + T1[:3, 3]
        p1 = (K @ p1_c)[:2] / p1_c[2]
        p2_c = T2[:3, :3] @ P + T2[:3, 3]
        p2 = (K @ p2_c)[:2] / p2_c[2]

        kps0.append(cv2.KeyPoint(float(p0[0]), float(p0[1]), 31))
        kps1.append(cv2.KeyPoint(float(p1[0]), float(p1[1]), 31))
        kps2.append(cv2.KeyPoint(float(p2[0]), float(p2[1]), 31))

    f0 = Frame(kps0, None); f0.T_cw = np.copy(T0); kf0 = Keyframe(f0)
    
    # Introduce initial error on KF1 and KF2 (+0.05m translation noise)
    f1 = Frame(kps1, None); f1.T_cw = np.copy(T1); f1.T_cw[0, 3] += 0.05; kf1 = Keyframe(f1)
    f2 = Frame(kps2, None); f2.T_cw = np.copy(T2); f2.T_cw[0, 3] += 0.05; kf2 = Keyframe(f2)

    # 3. Create MapPoints with 3D noise and link observations
    map_points = []
    for idx, P in enumerate(pts3D_gt):
        noisy_P = P + np.random.normal(0, 0.02, 3).astype(np.float32)
        mp = MapPoint(noisy_P, kf0, idx)
        mp.add_observation(kf1, idx)
        mp.add_observation(kf2, idx)
        kf0.map_points[idx] = mp
        kf1.map_points[idx] = mp
        kf2.map_points[idx] = mp
        map_points.append(mp)

    init_err1 = abs(kf1.T_cw[0, 3] - T1[0, 3])
    init_err2 = abs(kf2.T_cw[0, 3] - T2[0, 3])
    print(f"Pre-BA KF1 Error: {init_err1:.4f} m, KF2 Error: {init_err2:.4f} m")

    # Run Local BA: optimize [kf1, kf2], anchor [kf0]
    ba.local_bundle_adjustment(
        local_keyframes=[kf1, kf2],
        fixed_keyframes=[kf0],
        local_map_points=map_points,
        max_iterations=30
    )

    final_err1 = abs(kf1.T_cw[0, 3] - T1[0, 3])
    final_err2 = abs(kf2.T_cw[0, 3] - T2[0, 3])
    print(f"Post-BA KF1 Error: {final_err1:.4f} m, KF2 Error: {final_err2:.4f} m")

    assert final_err1 < init_err1 and final_err2 < init_err2, "Bundle Adjustment failed to reduce error!"
    print("[TEST BUNDLE ADJUSTMENT SUCCESS]")


if __name__ == "__main__":
    test_bundle_adjustment()