import os
import sys
import numpy as np
import cv2

# Dynamic path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datastructures.frame import Frame
from src.datastructures.keyframe import Keyframe
from src.datastructures.map_point import MapPoint
from src.threads.local_mapping import LocalMapping


def test_local_mapping_pipeline():
    K = np.array([[800, 0, 320],
                  [0, 800, 240],
                  [0,   0,   1]], dtype=np.float32)

    local_mapping = LocalMapping(K, min_shared_points=5)

    # 1. Create 3 synthetic 3D points
    p1 = [0.0, 0.0, 4.0]
    p2 = [0.5, 0.2, 4.5]
    p3 = [-0.5, -0.2, 3.8]

    # Create Frame 1
    kps1 = [cv2.KeyPoint(320.0, 240.0, 31), cv2.KeyPoint(400.0, 260.0, 31), cv2.KeyPoint(220.0, 210.0, 31)]
    des1 = np.random.randint(0, 256, (3, 32), dtype=np.uint8)
    f1 = Frame(kps1, des1, timestamp=0.0)
    kf1 = Keyframe(f1)

    mp1 = MapPoint(p1, kf1, 0)
    mp2 = MapPoint(p2, kf1, 1)
    mp3 = MapPoint(p3, kf1, 2)
    kf1.map_points = [mp1, mp2, mp3]

    # Create Frame 2 with displacement along X
    kps2 = [cv2.KeyPoint(300.0, 240.0, 31), cv2.KeyPoint(380.0, 260.0, 31), cv2.KeyPoint(200.0, 210.0, 31)]
    des2 = np.copy(des1)
    f2 = Frame(kps2, des2, timestamp=1.0)
    f2.T_cw[0, 3] = -0.2
    kf2 = Keyframe(f2)

    # Both keyframes observe the same 3 points
    for idx, mp in enumerate([mp1, mp2, mp3]):
        mp.add_observation(kf2, idx)
    kf2.map_points = [mp1, mp2, mp3]

    keyframes_list = [kf1, kf2]
    map_points_list = [mp1, mp2, mp3]

    # Run Local Mapping on kf2
    local_mapping.add_keyframe(kf2)
    processed = local_mapping.process(map_points_list, keyframes_list)

    print(f"[TEST LOCAL MAPPING] Processed Keyframe: {processed}")
    print(f"Connected Keyframes in kf2: {len(kf2.connected_keyframes)}")
    print(f"Total MapPoints in Map: {len(map_points_list)}")

    assert processed is True, "Local mapping processing failed!"
    print("[TEST LOCAL MAPPING SUCCESS]")


if __name__ == "__main__":
    test_local_mapping_pipeline()