import numpy as np
from src.optimization.bundle_adjustment import BundleAdjustment


class Map:
    """
    Represents an individual submap in ORB-SLAM3.
    Contains its own set of Keyframes, MapPoints, and origin anchor keyframe.
    """
    __slots__ = ('id', 'keyframes', 'map_points', 'init_keyframe', 'is_bad', 'max_kf_id', 'max_mp_id')
    _id_counter = 0

    def __init__(self):
        self.id = Map._id_counter
        Map._id_counter += 1
        self.keyframes = []
        self.map_points = []
        self.init_keyframe = None
        self.is_bad = False
        self.max_kf_id = 0
        self.max_mp_id = 0

    def add_keyframe(self, keyframe):
        """Adds a keyframe to this submap."""
        self.keyframes.append(keyframe)
        if self.init_keyframe is None:
            self.init_keyframe = keyframe
        if keyframe.id > self.max_kf_id:
            self.max_kf_id = keyframe.id

    def add_map_point(self, map_point):
        """Adds a map point to this submap."""
        self.map_points.append(map_point)
        if map_point.id > self.max_mp_id:
            self.max_mp_id = map_point.id

    def get_keyframes(self):
        """Returns all non-culled keyframes in this submap."""
        return [kf for kf in self.keyframes if not getattr(kf, 'is_bad', False)]

    def get_map_points(self):
        """Returns all non-culled map points in this submap."""
        return [mp for mp in self.map_points if not getattr(mp, 'is_bad', False)]

    def count_keyframes(self):
        return sum(1 for kf in self.keyframes if not getattr(kf, 'is_bad', False))

    def count_map_points(self):
        return sum(1 for mp in self.map_points if not getattr(mp, 'is_bad', False))

    def clear(self):
        """Marks submap as bad/culled."""
        self.is_bad = True
        self.keyframes.clear()
        self.map_points.clear()


class Atlas:
    """
    Multi-map manager for ORB-SLAM3.
    Maintains one active submap alongside a repository of inactive submaps.
    Coordinates Sim(3) alignment, duplicate point fusion, and Welded Local BA upon map merge.
    """
    __slots__ = ('maps', 'active_map', 'K', 'optimizer')

    def __init__(self, K=None):
        self.maps = []
        self.active_map = None
        self.K = K if K is not None else np.eye(3, dtype=np.float32)
        self.optimizer = BundleAdjustment(self.K)
        self.create_new_map()

    def create_new_map(self):
        """
        Creates a fresh submap and assigns it as current active map.
        Called on system startup or when tracking is lost in monocular mode.
        """
        new_map = Map()
        self.maps.append(new_map)
        self.active_map = new_map
        return new_map

    def get_active_map(self):
        """Returns the active submap."""
        return self.active_map

    def get_inactive_maps(self):
        """Returns all valid submaps other than the currently active one."""
        return [m for m in self.maps if m != self.active_map and not m.is_bad]

    def get_all_maps(self):
        """Returns all valid submaps in the atlas."""
        return [m for m in self.maps if not m.is_bad]

    def set_active_map(self, map_obj):
        """Switches the active submap pointer."""
        if map_obj in self.maps and not map_obj.is_bad:
            self.active_map = map_obj

    def merge_maps(self, target_map, source_map, S_target_source, current_kf=None, candidate_kf=None, matches=None, inliers=None):
        """
        Merges source_map into target_map via:
        1. Sim(3) coordinate transformation of all source points & poses
        2. Duplicate MapPoint fusion along the seam
        3. Welded Local BA optimizing boundary keyframes & fused points
        """
        if source_map == target_map or source_map.is_bad or target_map.is_bad:
            return

        s = np.cbrt(np.linalg.det(S_target_source[:3, :3]))
        R = S_target_source[:3, :3] / s
        t = S_target_source[:3, 3]

        # 1. Transform all source MapPoints into target map frame
        for mp in source_map.get_map_points():
            mp.pos = (s * (R @ mp.pos) + t).astype(np.float32)
            target_map.add_map_point(mp)

        # 2. Transform all source Keyframe poses
        S_inv = np.linalg.inv(S_target_source)
        for kf in source_map.get_keyframes():
            kf.T_cw = (kf.T_cw @ S_inv).astype(np.float32)
            target_map.add_keyframe(kf)

        # 3. Fuse duplicate MapPoints along the merge seam
        if current_kf is not None and candidate_kf is not None and matches is not None and inliers is not None:
            self._fuse_duplicate_points(current_kf, candidate_kf, matches, inliers)

            # 4. Welded Local BA: Optimize boundary keyframes & fused map points
            self._run_welded_local_ba(target_map, current_kf, candidate_kf)

        # 5. Retire source map
        source_map.clear()
        if self.active_map == source_map:
            self.active_map = target_map

    def _fuse_duplicate_points(self, current_kf, candidate_kf, matches, inliers):
        """Fuses duplicate map points along the merge boundary."""
        for inlier_idx in inliers:
            m = matches[inlier_idx]
            mp_curr = current_kf.map_points[m.queryIdx]
            mp_cand = candidate_kf.map_points[m.trainIdx]

            if mp_curr is not None and mp_cand is not None and mp_curr != mp_cand:
                for obs_kf, kp_idx in list(mp_curr.observations.items()):
                    obs_kf.map_points[kp_idx] = mp_cand
                    mp_cand.add_observation(obs_kf, kp_idx)
                mp_curr.is_bad = True

    def _run_welded_local_ba(self, target_map, current_kf, candidate_kf):
        """Executes Welded Local BA across the boundary keyframes."""
        welded_kfs = [current_kf, candidate_kf]
        welded_mps = []

        for kf in welded_kfs:
            for mp in kf.map_points:
                if mp is not None and not getattr(mp, 'is_bad', False):
                    welded_mps.append(mp)

        # Neighbor keyframes outside the welding window are held fixed
        fixed_kfs = [kf for kf in target_map.get_keyframes() if kf not in welded_kfs]

        print("[ATLAS] Running Welded Local BA on Map Merge boundary...")
        self.optimizer.local_bundle_adjustment(
            local_keyframes=welded_kfs,
            fixed_keyframes=fixed_kfs[:3] if fixed_kfs else [],
            local_map_points=list(set(welded_mps)),
            max_iterations=20
        )

    def count_keyframes(self):
        """Total keyframe count across all active and inactive submaps."""
        return sum(m.count_keyframes() for m in self.maps if not m.is_bad)

    def count_map_points(self):
        """Total map point count across all active and inactive submaps."""
        return sum(m.count_map_points() for m in self.maps if not m.is_bad)