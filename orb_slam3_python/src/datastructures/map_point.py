import numpy as np


class MapPoint:
    """
    Persistent 3D Landmark observed across multiple Keyframes.
    Uses __slots__ to allow housing tens of thousands of points in memory efficiently.
    """
    __slots__ = (
        'id', 'pos', 'observations', 'distinctive_des', 
        'mean_view_dir', 'n_obs', 'is_bad', 'ref_keyframe'
    )

    _id_counter = 0

    def __init__(self, position, keyframe, kp_idx):
        self.id = MapPoint._id_counter
        MapPoint._id_counter += 1

        # 3D Position in World Frame (x, y, z) - float32
        self.pos = np.array(position, dtype=np.float32)
        
        # Dictionary mapping Keyframe -> feature index in that Keyframe {Keyframe: int}
        self.observations = {keyframe: kp_idx}
        self.ref_keyframe = keyframe
        self.n_obs = 1
        self.is_bad = False
        
        # Representative ORB descriptor selected as median distance among observations
        self.distinctive_des = None
        self.mean_view_dir = np.zeros(3, dtype=np.float32)

    def add_observation(self, keyframe, kp_idx):
        if keyframe not in self.observations:
            self.observations[keyframe] = kp_idx
            self.n_obs += 1

    def remove_observation(self, keyframe):
        if keyframe in self.observations:
            del self.observations[keyframe]
            self.n_obs -= 1
            if self.n_obs <= 0:
                self.is_bad = True

    def update_distinctive_descriptor(self):
        """
        Calculates the descriptor with the minimum median Hamming distance 
        to all other observing descriptors (ORB-SLAM3 standard).
        """
        if not self.observations:
            return

        descriptors = []
        for kf, idx in self.observations.items():
            if not kf.is_bad and kf.des is not None and idx < len(kf.des):
                descriptors.append(kf.des[idx])

        if not descriptors:
            return

        descriptors = np.array(descriptors, dtype=np.uint8)
        if len(descriptors) == 1:
            self.distinctive_des = descriptors[0]
            return

        # Compute pairwise Hamming distances efficiently via XOR & Bitcount
        N = len(descriptors)
        distances = np.zeros((N, N), dtype=np.int32)
        for i in range(N):
            for j in range(i + 1, N):
                # Hamming distance between binary descriptors
                dist = np.count_nonzero(np.unpackbits(np.bitwise_xor(descriptors[i], descriptors[j])))
                distances[i, j] = dist
                distances[j, i] = dist

        # Find descriptor with minimum median distance
        medians = np.median(distances, axis=1)
        best_idx = np.argmin(medians)
        self.distinctive_des = descriptors[best_idx]