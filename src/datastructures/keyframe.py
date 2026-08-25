import numpy as np


class Keyframe:
    """
    Keyframe representation in ORB-SLAM3.
    Stores pose (T_cw), visual features, covisibility graph connections,
    spanning tree parent/children links, and DBoW2 visual vectors.
    """
    __slots__ = (
        'id', 'frame_id', 'timestamp', 'kps', 'des', 'map_points',
        'T_cw', 'is_bad', 'connected_keyframes', 'parent', 'children',
        'bow_vector', 'feature_vector'
    )
    _id_counter = 0

    def __init__(self, frame):
        self.id = Keyframe._id_counter
        Keyframe._id_counter += 1

        self.frame_id = getattr(frame, 'id', self.id)
        self.timestamp = getattr(frame, 'timestamp', 0.0)

        # Feature data
        self.kps = frame.kps
        self.des = frame.des
        self.map_points = [None] * len(self.kps) if self.kps else []

        # Pose: World-to-Camera transformation T_cw
        self.T_cw = np.copy(frame.T_cw).astype(np.float32) if hasattr(frame, 'T_cw') and frame.T_cw is not None else np.eye(4, dtype=np.float32)

        # Covisibility Graph & Spanning Tree structures
        self.connected_keyframes = {}  # {Keyframe: weight}
        self.parent = None             # Spanning tree parent Keyframe
        self.children = set()          # Spanning tree children Keyframes

        # DBoW2 representations
        self.bow_vector = {}           # {word_id: normalized_weight}
        self.feature_vector = {}       # {node_id: [feature_indices]}

        self.is_bad = False

    def add_connection(self, keyframe, weight):
        """Adds or updates a weighted edge in the covisibility graph."""
        if keyframe != self and not getattr(keyframe, 'is_bad', False):
            self.connected_keyframes[keyframe] = weight

    def get_camera_center(self):
        """Returns 3D optical center position in world coordinates (C_w = -R^T * t)."""
        R_cw = self.T_cw[:3, :3]
        t_cw = self.T_cw[:3, 3]
        return -R_cw.T @ t_cw

    def set_parent(self, parent_kf):
        """Sets spanning tree parent."""
        self.parent = parent_kf
        if parent_kf is not None:
            parent_kf.children.add(self)