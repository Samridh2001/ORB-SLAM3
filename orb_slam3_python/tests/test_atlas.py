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


def test_atlas_system():
    atlas = Atlas()

    # 1. Verify Active Map Creation
    map1 = atlas.get_active_map()
    assert map1 is not None, "Atlas failed to create default active map!"
    assert len(atlas.get_all_maps()) == 1

    # Add items to Map 1
    f0 = Frame([cv2.KeyPoint(100, 100, 31)], np.zeros((1, 32), dtype=np.uint8))
    kf0 = Keyframe(f0)
    mp0 = MapPoint(np.array([0.0, 0.0, 3.0], dtype=np.float32), kf0, 0)
    map1.add_keyframe(kf0)
    map1.add_map_point(mp0)

    assert map1.count_keyframes() == 1
    assert map1.count_map_points() == 1

    # 2. Simulate Tracking Loss: Create new Submap (Map 2)
    map2 = atlas.create_new_map()
    assert atlas.get_active_map() == map2
    assert len(atlas.get_all_maps()) == 2
    assert len(atlas.get_inactive_maps()) == 1

    # Add items to Map 2
    f1 = Frame([cv2.KeyPoint(200, 200, 31)], np.zeros((1, 32), dtype=np.uint8))
    kf1 = Keyframe(f1)
    mp1 = MapPoint(np.array([1.0, 0.0, 3.0], dtype=np.float32), kf1, 0)
    map2.add_keyframe(kf1)
    map2.add_map_point(mp1)

    print(f"Total Keyframes in Atlas across all maps: {atlas.count_keyframes()}")
    print(f"Total MapPoints in Atlas across all maps: {atlas.count_map_points()}")
    assert atlas.count_keyframes() == 2
    assert atlas.count_map_points() == 2

    # 3. Simulate Map Merging via Sim(3) Alignment: Merge Map 2 into Map 1
    # Identity transform with +0.5m translation along X
    S_12 = np.eye(4, dtype=np.float32)
    S_12[0, 3] = 0.5

    atlas.merge_maps(target_map=map1, source_map=map2, S_target_source=S_12)

    assert atlas.get_active_map() == map1
    assert len(atlas.get_all_maps()) == 1
    assert map1.count_keyframes() == 2
    assert map1.count_map_points() == 2

    # Verify transformed point position in merged map
    merged_mps = map1.get_map_points()
    assert np.isclose(merged_mps[1].pos[0], 1.5), "Sim(3) map point transformation mismatch!"

    print("[TEST ATLAS MULTI-MAP SYSTEM SUCCESS]")


if __name__ == "__main__":
    test_atlas_system()