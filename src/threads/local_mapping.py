import numpy as np
import cv2

from src.datastructures.map_point import MapPoint
from src.datastructures.covisibility_graph import CovisibilityGraph
from src.features.feature_matcher import FeatureMatcher
from src.geometry.triangulation import triangulate_point
from src.optimization.bundle_adjustment import BundleAdjustment


class LocalMapping:
    """
    Local Mapping thread pipeline.
    Processes new Keyframes, creates new MapPoints via epipolar matching,
    executes Local Bundle Adjustment, and culls redundant landmarks & keyframes.
    """

    def __init__(self, K, min_shared_points=15):
        self.K = np.array(K, dtype=np.float32)
        self.K_inv = np.linalg.inv(K).astype(np.float32)

        self.covisibility = CovisibilityGraph(min_shared_points=min_shared_points)
        self.matcher = FeatureMatcher(ratio_thresh=0.8, max_hamming_dist=50)
        self.optimizer = BundleAdjustment(K)

        self.keyframe_queue = []
        self.recent_map_points = []
        self.current_keyframe = None

    def add_keyframe(self, keyframe):
        """Pushes a new Keyframe to the processing queue."""
        self.keyframe_queue.append(keyframe)

    def process(self, map_points_list, keyframes_list):
        """
        Main execution step. Consumes next keyframe in queue and runs local mapping.
        Returns: True if a keyframe was processed, False otherwise.
        """
        if not self.keyframe_queue:
            return False

        # 1. Pop next keyframe
        self.current_keyframe = self.keyframe_queue.pop(0)

        # 2. Update Covisibility Graph connections
        self.covisibility.update_connections(self.current_keyframe)

        # 3. MapPoint Culling
        self._cull_recent_map_points()

        # 4. Triangulate new MapPoints with neighboring keyframes
        new_mps = self._create_new_map_points(keyframes_list)
        map_points_list.extend(new_mps)
        self.recent_map_points.extend(new_mps)

        # 5. Run Local Bundle Adjustment on the active local window
        self._run_local_ba(keyframes_list)

        # 6. Keyframe Culling
        self._cull_redundant_keyframes(keyframes_list)

        return True

    def _run_local_ba(self, keyframes_list):
        """
        Gathers active keyframes (local window), fixed neighbor keyframes,
        and visible map points, then executes Local BA.
        """
        if len(keyframes_list) < 2:
            return

        local_kfs = [self.current_keyframe]
        neighbors = self.covisibility.get_covisible_keyframes(self.current_keyframe, n_top=5)
        for kf in neighbors:
            if kf not in local_kfs and not getattr(kf, 'is_bad', False):
                local_kfs.append(kf)

        fixed_kfs = set()
        local_mps = set()

        for kf in local_kfs:
            for mp in kf.map_points:
                if mp is not None and not getattr(mp, 'is_bad', False):
                    local_mps.add(mp)
                    for obs_kf in mp.observations.keys():
                        if obs_kf not in local_kfs and not getattr(obs_kf, 'is_bad', False):
                            fixed_kfs.add(obs_kf)

        # Anchor initial keyframe (id=0) to lock gauge freedom if no fixed keyframe exists
        if not fixed_kfs:
            anchor_kf = keyframes_list[0]
            if anchor_kf in local_kfs and len(local_kfs) > 1:
                local_kfs.remove(anchor_kf)
                fixed_kfs.add(anchor_kf)

        # Execute optimization
        self.optimizer.local_bundle_adjustment(
            local_keyframes=local_kfs,
            fixed_keyframes=list(fixed_kfs),
            local_map_points=list(local_mps),
            max_iterations=25
        )

    def _cull_recent_map_points(self):
        """Prunes unstable or unobserved newly created points."""
        surviving_points = []
        for mp in self.recent_map_points:
            if getattr(mp, 'is_bad', False):
                continue
            if mp.n_obs < 2:
                mp.is_bad = True
            else:
                surviving_points.append(mp)
        self.recent_map_points = surviving_points

    def _create_new_map_points(self, keyframes_list):
        """Triangulates new 3D points between current keyframe and top covisible neighbors."""
        neighbors = self.covisibility.get_covisible_keyframes(self.current_keyframe, n_top=5)
        if not neighbors and len(keyframes_list) > 1:
            neighbors = [keyframes_list[-2]]

        new_map_points = []
        curr_kf = self.current_keyframe
        P1 = self.K @ curr_kf.T_cw[:3, :]

        for neighbor in neighbors:
            if neighbor == curr_kf or getattr(neighbor, 'is_bad', False):
                continue

            baseline = np.linalg.norm(curr_kf.get_camera_center() - neighbor.get_camera_center())
            if baseline < 0.01:
                continue

            P2 = self.K @ neighbor.T_cw[:3, :]
            matches = self.matcher.match_descriptors(curr_kf.des, neighbor.des)

            for m in matches:
                idx_curr = m.queryIdx
                idx_neigh = m.trainIdx

                if curr_kf.map_points[idx_curr] is not None or neighbor.map_points[idx_neigh] is not None:
                    continue

                pt1 = curr_kf.kps[idx_curr].pt
                pt2 = neighbor.kps[idx_neigh].pt

                p3D = triangulate_point(P1, P2, pt1, pt2)
                if p3D is None:
                    continue

                p3D_c1 = curr_kf.T_cw[:3, :3] @ p3D + curr_kf.T_cw[:3, 3]
                p3D_c2 = neighbor.T_cw[:3, :3] @ p3D + neighbor.T_cw[:3, 3]

                if p3D_c1[2] <= 0.1 or p3D_c2[2] <= 0.1:
                    continue

                proj1 = (self.K @ p3D_c1)[:2] / p3D_c1[2]
                proj2 = (self.K @ p3D_c2)[:2] / p3D_c2[2]

                err1 = np.sum((proj1 - np.array(pt1)) ** 2)
                err2 = np.sum((proj2 - np.array(pt2)) ** 2)

                if err1 < 4.0 and err2 < 4.0:
                    mp = MapPoint(position=p3D, keyframe=curr_kf, kp_idx=idx_curr)
                    mp.add_observation(neighbor, idx_neigh)
                    mp.update_distinctive_descriptor()

                    curr_kf.map_points[idx_curr] = mp
                    neighbor.map_points[idx_neigh] = mp
                    new_map_points.append(mp)

        return new_map_points

    def _cull_redundant_keyframes(self, keyframes_list):
        """Culls keyframes if > 90% of observed points are seen in >= 3 other keyframes."""
        if len(keyframes_list) <= 3:
            return

        neighbors = self.covisibility.get_covisible_keyframes(self.current_keyframe, n_top=10)
        for kf in neighbors:
            if kf.id == 0 or getattr(kf, 'is_bad', False):
                continue

            n_redundant_points = 0
            n_valid_points = 0

            for mp in kf.map_points:
                if mp is None or getattr(mp, 'is_bad', False):
                    continue
                n_valid_points += 1
                if mp.n_obs >= 3:
                    n_redundant_points += 1

            if n_valid_points > 0 and (n_redundant_points / n_valid_points) > 0.90:
                kf.is_bad = True
                for mp in kf.map_points:
                    if mp is not None and not getattr(mp, 'is_bad', False):
                        mp.remove_observation(kf)