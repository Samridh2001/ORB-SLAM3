import os
import sys
import numpy as np
import cv2

# Dynamic path resolution to avoid setting PYTHONPATH manually
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datastructures.frame import Frame
from src.datastructures.keyframe import Keyframe
from src.datastructures.map_point import MapPoint

def test_datastructures():
    # 1. Dummy ORB keypoints & descriptors
    kps = [cv2.KeyPoint(x=10.0 * i, y=20.0 * i, size=31) for i in range(5)]
    des = np.random.randint(0, 255, (5, 32), dtype=np.uint8)

    # 2. Instantiate Frame
    frame = Frame(kps, des, timestamp=1.0)
    
    # 3. Convert Frame to Keyframe
    keyframe = Keyframe(frame)
    
    # 4. Instantiate a MapPoint observed by Keyframe
    mp = MapPoint(position=[1.0, 2.0, 5.0], keyframe=keyframe, kp_idx=0)
    keyframe.map_points[0] = mp
    
    # 5. Compute representative descriptor
    mp.update_distinctive_descriptor()

    print(f"[TEST SUCCESS]")
    print(f"Keyframe ID: {keyframe.id}, Pose Center: {keyframe.get_camera_center()}")
    print(f"MapPoint ID: {mp.id}, Pos: {mp.pos}, Obs Count: {mp.n_obs}")

if __name__ == "__main__":
    test_datastructures()