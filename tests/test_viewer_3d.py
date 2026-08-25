import os
import sys
import time
import numpy as np
import cv2

# Dynamic path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datastructures.frame import Frame
from src.datastructures.keyframe import Keyframe
from src.datastructures.map_point import MapPoint
from visualization.viewer_3d import Viewer3D


def test_viewer_3d():
    viewer = Viewer3D(window_name="ORB-SLAM3 Test Visualizer", width=800, height=600)

    # 1. Create synthetic MapPoints
    np.random.seed(42)
    n_points = 100
    pts3D = np.random.uniform(-2.0, 2.0, (n_points, 3)).astype(np.float32)
    pts3D[:, 2] += 4.0

    dummy_kps = [cv2.KeyPoint(100, 100, 31)]
    dummy_des = np.zeros((1, 32), dtype=np.uint8)
    dummy_frame = Frame(dummy_kps, dummy_des)
    dummy_kf = Keyframe(dummy_frame)

    map_points = [MapPoint(pos, dummy_kf, 0) for pos in pts3D]

    # 2. Create synthetic Keyframe trajectory
    keyframes = []
    for i in range(10):
        f = Frame(dummy_kps, dummy_des, timestamp=float(i))
        f.T_cw = np.eye(4, dtype=np.float32)
        f.T_cw[0, 3] = -i * 0.2  # Moving along X
        kf = Keyframe(f)
        keyframes.append(kf)

    # Connect consecutive keyframes in covisibility graph
    for i in range(len(keyframes) - 1):
        keyframes[i].add_connection(keyframes[i + 1], weight=30)
        keyframes[i + 1].add_connection(keyframes[i], weight=30)

    # 3. Test Threaded Update Pipeline
    viewer.start()
    print("Sending updates to 3D viewer...")

    for i in range(10):
        viewer.update(keyframes[: i + 1], map_points, current_frame=keyframes[i])
        time.sleep(0.05)

    time.sleep(0.5)
    viewer.stop()
    print("[TEST 3D VIEWER PIPELINE SUCCESS]")


if __name__ == "__main__":
    test_viewer_3d()