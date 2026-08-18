import numpy as np


class Keyframe:
    """
    Permanent Node in the Covisibility Graph / Atlas Map.
    Stores camera pose, linked MapPoints, and covisibility connections.
    """
    __slots__ = (
        'id', 'frame_id', 'timestamp', 'T_cw', 'kps', 'des', 
        'map_points', 'connected_keyframes', 'is_bad'
    )

    _id_counter = 0

    def __init__(self, frame):
        self.id = Keyframe._id_counter
        Keyframe._id_counter += 1

        self.frame_id = frame.id
        self.timestamp = frame.timestamp
        self.T_cw = np.copy(frame.T_cw).astype(np.float32)
        
        self.kps = frame.kps
        self.des = frame.des
        
        # Direct references to MapPoints
        self.map_points = list(frame.map_points)
        
        # Covisibility Graph connections: {Keyframe: weight_num_shared_points}
        self.connected_keyframes = {}
        self.is_bad = False

    def add_connection(self, keyframe, weight):
        self.connected_keyframes[keyframe] = weight

    def get_camera_center(self):
        R_cw = self.T_cw[:3, :3]
        t_cw = self.T_cw[:3, 3]
        return -R_cw.T @ t_cw