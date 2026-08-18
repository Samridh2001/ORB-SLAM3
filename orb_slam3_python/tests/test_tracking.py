import os
import sys
import cv2
import numpy as np

# Dynamic path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datastructures.frame import Frame
from src.threads.tracking import Tracking, TrackingState


def test_tracking_pipeline():
    # Camera Intrinsics
    K = np.array([[800, 0, 320],
                  [0, 800, 240],
                  [0,   0,   1]], dtype=np.float32)

    tracker = Tracking(K)

    # Generate 3D landmark points
    np.random.seed(42)
    n_points = 150
    pts3D_gt = np.random.uniform(-1.5, 1.5, (n_points, 3)).astype(np.float32)
    pts3D_gt[:, 2] += 4.0  # Depth between 2.5m and 5.5m

    # Unique descriptors for landmarks
    gt_descriptors = np.random.randint(0, 256, (n_points, 32), dtype=np.uint8)

    # True camera positions moving smoothly: 0.15m per frame along X-axis
    poses_T_cw = []
    for i in range(3):
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = -i * 0.15
        poses_T_cw.append(T)

    frames = []
    for i, T_cw in enumerate(poses_T_cw):
        R = T_cw[:3, :3]
        t = T_cw[:3, 3]

        # Project 3D points to 2D
        pts3D_cam = (R @ pts3D_gt.T).T + t
        proj = (K @ pts3D_cam.T).T
        pts2D = proj[:, :2] / proj[:, 2:]

        kps = [cv2.KeyPoint(float(p[0]), float(p[1]), 31.0) for p in pts2D]
        des = np.copy(gt_descriptors)

        frame = Frame(kps, des, timestamp=float(i))
        frames.append(frame)

    # Frame 0: Startup
    state0 = tracker.process_frame(frames[0])
    print(f"Frame 0 State: {state0}")

    # Frame 1: Monocular Initialization
    state1 = tracker.process_frame(frames[1])
    print(f"Frame 1 State: {state1}")
    assert state1 == TrackingState.OK, "Monocular Initialization failed on Frame 1!"

    # Frame 2: Velocity Tracking + PnP
    state2 = tracker.process_frame(frames[2])
    print(f"Frame 2 State: {state2}")
    assert state2 == TrackingState.OK, "Motion Model Tracking failed on Frame 2!"

    print("[TEST TRACKING PIPELINE SUCCESS]")


if __name__ == "__main__":
    test_tracking_pipeline()