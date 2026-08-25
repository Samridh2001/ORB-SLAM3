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
from visualization.image_drawer import ImageDrawer


def test_image_drawer():
    drawer = ImageDrawer()

    # 1. Create synthetic camera frame
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)  # Dark grey background

    # 2. Add synthetic keypoints
    kps = [
        cv2.KeyPoint(100.0, 150.0, 31),
        cv2.KeyPoint(200.0, 250.0, 31),
        cv2.KeyPoint(350.0, 180.0, 31),
        cv2.KeyPoint(500.0, 300.0, 31),
        cv2.KeyPoint(400.0, 400.0, 31)
    ]
    frame = Frame(kps, np.zeros((len(kps), 32), dtype=np.uint8))

    # 3. Associate 3 landmarks as tracked MapPoints
    kf = Keyframe(frame)
    mp1 = MapPoint(np.array([0, 0, 2], dtype=np.float32), kf, 0)
    mp2 = MapPoint(np.array([1, 0, 3], dtype=np.float32), kf, 1)
    mp3 = MapPoint(np.array([0, 1, 4], dtype=np.float32), kf, 2)

    frame.map_points[0] = mp1
    frame.map_points[1] = mp2
    frame.map_points[2] = mp3
    # Keypoints 3 and 4 remain unmatched (None)

    # 4. Atlas manager
    atlas = Atlas()
    active_map = atlas.get_active_map()
    active_map.add_keyframe(kf)
    active_map.add_map_point(mp1)
    active_map.add_map_point(mp2)
    active_map.add_map_point(mp3)

    # 5. Render canvas
    canvas = drawer.draw_frame(
        image=img,
        current_frame=frame,
        tracking_state_str="OK",
        atlas=atlas
    )

    assert canvas is not None, "Canvas rendering failed!"
    assert canvas.shape == (480, 640, 3), "Canvas shape altered during rendering!"
    
    # Verify non-empty drawing (pixel values modified from uniform 50)
    assert not np.all(canvas == 50), "Visualizer failed to draw features on canvas!"

    # Save output frame for inspection
    out_path = "tests/test_output_tracking_canvas.png"
    cv2.imwrite(out_path, canvas)
    print(f"Canvas saved to '{out_path}'")
    print("[TEST 2D IMAGE DRAWER SUCCESS]")


if __name__ == "__main__":
    test_image_drawer()