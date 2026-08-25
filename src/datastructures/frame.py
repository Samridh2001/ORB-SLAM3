import numpy as np


class Frame:
    """
    Transient processing frame representing a single image capture.
    Optimized with __slots__ to minimize CPU RAM footprint.
    """
    __slots__ = (
        'id', 'timestamp', 'kps', 'des', 'kps_undistorted', 
        'map_points', 'outliers', 'T_cw', 'N', 'grid'
    )

    _id_counter = 0

    def __init__(self, keypoints, descriptors, timestamp=0.0):
        self.id = Frame._id_counter
        Frame._id_counter += 1
        
        self.timestamp = timestamp
        self.kps = keypoints  # List of cv2.KeyPoint
        self.des = descriptors  # np.ndarray uint8 (N, 32)
        self.N = len(keypoints) if keypoints is not None else 0
        
        # Array of associated MapPoint pointers (populated during tracking)
        self.map_points = [None] * self.N
        self.outliers = [False] * self.N
        
        # 4x4 World-to-Camera Transformation SE(3) matrix (float32 for CPU RAM efficiency)
        self.T_cw = np.eye(4, dtype=np.float32)

    def get_camera_center(self):
        """Returns 3D camera position in World coordinates: -R^T * t"""
        R_cw = self.T_cw[:3, :3]
        t_cw = self.T_cw[:3, 3]
        return -R_cw.T @ t_cw

    def clear(self):
        """Allows rapid GC cleanup when frame is dropped."""
        self.kps = None
        self.des = None
        self.map_points = None