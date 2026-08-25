import os
import sys
import numpy as np
import cv2

# Dynamic path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datastructures.frame import Frame
from src.threads.tracking import Tracking, TrackingState
from src.threads.local_mapping import LocalMapping


def test_full_tracking_mapping_pipeline():
    K = np.array([[800, 0, 320],
                  [0, 800, 240],
                  [0,   0,   1]], dtype=np.float32)

    tracker = Tracking(K)
    local_mapper = LocalMapping(K, min_shared_points=5)

    # 1. Generate 3D synthetic landmark points
    np.random.seed(42)
    n_points = 120
    pts3D_gt = np.random.uniform(-1.5, 1.5, (n_points, 3)).astype(np.float32)
    pts3D_gt[:, 2] += 4.0

    gt_descriptors = np.random.randint(0, 256, (n_points, 32), dtype=np.uint8)

    # 2. Simulate 5 camera frames moving smoothly along X-axis
    poses_T_cw = []
    for i in range(5):
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = -i * 0.15
        poses_T_cw.append(T)

    frames = []
    for i, T_cw in enumerate(poses_T_cw):
        R = T_cw[:3, :3]
        t = T_cw[:3, 3]

        pts3D_cam = (R @ pts3D_gt.T).T + t
        proj = (K @ pts3D_cam.T).T
        pts2D = proj[:, :2] / proj[:, 2:]

        kps = [cv2.KeyPoint(float(p[0]), float(p[1]), 31.0) for p in pts2D]
        des = np.copy(gt_descriptors)

        frame = Frame(kps, des, timestamp=float(i))
        frames.append(frame)

    # Process all frames through tracking and local mapping
    for i, frame in enumerate(frames):
        state = tracker.process_frame(frame)
        print(f"Frame {i} Tracking State: {state}")

        # If keyframe was added, feed it to local mapping
        if tracker.keyframes and (not local_mapper.keyframe_queue or local_mapper.keyframe_queue[-1] != tracker.keyframes[-1]):
            latest_kf = tracker.keyframes[-1]
            local_mapper.add_keyframe(latest_kf)
            local_mapper.process(tracker.map_points, tracker.keyframes)

    print(f"Total Keyframes: {len(tracker.keyframes)}")
    print(f"Total MapPoints: {len(tracker.map_points)}")
    assert tracker.state == TrackingState.OK, "Tracking failed!"
    print("[TEST TRACKING + LOCAL MAPPING INTEGRATION SUCCESS]")


if __name__ == "__main__":
    test_full_tracking_mapping_pipeline()