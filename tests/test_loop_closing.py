import os
import sys
import numpy as np
import cv2

# Dynamic path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datastructures.frame import Frame
from src.datastructures.keyframe import Keyframe
from src.datastructures.map_point import MapPoint
from src.datastructures.atlas import Atlas
from src.threads.loop_closing import LoopClosing


def test_loop_closing_and_map_merge():
    K = np.array([[800, 0, 320],
                  [0, 800, 240],
                  [0,   0,   1]], dtype=np.float32)

    atlas = Atlas()
    loop_closer = LoopClosing(atlas, K, min_inliers=15)

    # 1. Ground Truth Scene (30 landmarks)
    np.random.seed(42)
    n_points = 30
    pts3D_gt = np.random.uniform(-1.5, 1.5, (n_points, 3)).astype(np.float32)
    pts3D_gt[:, 2] += 4.0
    des_gt = np.random.randint(0, 256, (n_points, 32), dtype=np.uint8)

    # 2. Historical Keyframe (KF0) in Map 0
    map0 = atlas.get_active_map()
    kps0 = [cv2.KeyPoint(100.0 + i * 5, 100.0 + i * 2, 31) for i in range(n_points)]
    f0 = Frame(kps0, np.copy(des_gt), timestamp=0.0)
    kf0 = Keyframe(f0)
    map0.add_keyframe(kf0)

    for i, P in enumerate(pts3D_gt):
        mp = MapPoint(P, kf0, i)
        kf0.map_points[i] = mp
        map0.add_map_point(mp)

    # 3. Simulate Lost Tracking -> Create Map 1 in Atlas
    map1 = atlas.create_new_map()
    kps1 = [cv2.KeyPoint(100.0 + i * 5, 100.0 + i * 2, 31) for i in range(n_points)]
    f1 = Frame(kps1, np.copy(des_gt), timestamp=10.0)
    kf1 = Keyframe(f1)
    map1.add_keyframe(kf1)

    # Add duplicate points with noise in Map 1
    for i, P in enumerate(pts3D_gt):
        noisy_P = P + np.random.normal(0, 0.03, 3).astype(np.float32)
        mp = MapPoint(noisy_P, kf1, i)
        kf1.map_points[i] = mp
        map1.add_map_point(mp)

    assert len(atlas.get_all_maps()) == 2, "Atlas should have 2 active/inactive maps pre-merge."

    # 4. Trigger Map Merge via Loop Closing
    loop_closer.add_keyframe(kf1)
    executed = loop_closer.process()

    print(f"Merge Executed: {executed}")
    print(f"Total Maps in Atlas Post-Merge: {len(atlas.get_all_maps())}")
    print(f"Total MapPoints in Unified Map: {atlas.get_active_map().count_map_points()}")

    assert executed is True, "Map merge failed!"
    assert len(atlas.get_all_maps()) == 1, "Submaps were not unified into a single map!"
    assert atlas.get_active_map().count_map_points() == n_points, "Duplicate landmarks were not fused!"
    print("[TEST MAP MERGING + WELDED BA SUCCESS]")


if __name__ == "__main__":
    test_loop_closing_and_map_merge()