class CovisibilityGraph:
    """
    Manages connections between Keyframes in the active map.
    Two Keyframes share an edge if they observe the same MapPoints.
    Weight = number of shared MapPoints.
    """
    __slots__ = ('min_shared_points',)

    def __init__(self, min_shared_points=15):
        self.min_shared_points = min_shared_points

    def update_connections(self, keyframe):
        """
        Updates the covisibility weights for a keyframe based on current MapPoint observations.
        """
        # Count shared points with other keyframes
        kf_counter = {}

        for mp in keyframe.map_points:
            if mp is None or mp.is_bad:
                continue

            for obs_kf in mp.observations.keys():
                if obs_kf == keyframe or obs_kf.is_bad:
                    continue
                kf_counter[obs_kf] = kf_counter.get(obs_kf, 0) + 1

        if not kf_counter:
            keyframe.connected_keyframes.clear()
            return

        # Filter connections by minimum threshold
        connected = {}
        for other_kf, count in kf_counter.items():
            if count >= self.min_shared_points:
                connected[other_kf] = count
                # Symmetrically update the neighbor's connection back
                other_kf.connected_keyframes[keyframe] = count

        keyframe.connected_keyframes = connected

    @staticmethod
    def get_covisible_keyframes(keyframe, n_top=10):
        """
        Returns the top-N covisible keyframes sorted in descending order of shared points.
        """
        if not keyframe.connected_keyframes:
            return []

        sorted_neighbors = sorted(
            keyframe.connected_keyframes.items(),
            key=lambda item: item[1],
            reverse=True
        )
        return [kf for kf, _ in sorted_neighbors[:n_top]]

    staticmethod
    def get_best_covisible_keyframe(keyframe):
        """Returns the single neighbor with the highest covisibility weight."""
        if not keyframe.connected_keyframes:
            return None
        return max(keyframe.connected_keyframes.items(), key=lambda item: item[1])[0]