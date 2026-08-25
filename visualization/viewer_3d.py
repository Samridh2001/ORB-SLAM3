import threading
import time
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


class Viewer3D:
    """
    Real-time 3D Visualizer for ORB-SLAM3 using Open3D.
    Renders active 3D MapPoints, camera frustums, historical trajectory,
    and covisibility graph connections in a background thread.
    """

    def __init__(self, window_name="ORB-SLAM3 3D Viewer", width=1024, height=768):
        self.enabled = HAS_OPEN3D
        self.window_name = window_name
        self.width = width
        self.height = height

        # Visual geometries
        self.pcd_map_points = o3d.geometry.PointCloud() if HAS_OPEN3D else None
        self.lines_trajectory = o3d.geometry.LineSet() if HAS_OPEN3D else None
        self.lines_covisibility = o3d.geometry.LineSet() if HAS_OPEN3D else None
        self.frustums_keyframes = o3d.geometry.LineSet() if HAS_OPEN3D else None
        self.frustum_current = o3d.geometry.LineSet() if HAS_OPEN3D else None

        # Thread synchronization & state cache
        self.lock = threading.Lock()
        self.is_running = False
        self.thread = None

        self._cached_mp_positions = np.empty((0, 3), dtype=np.float32)
        self._cached_kf_centers = np.empty((0, 3), dtype=np.float32)
        self._cached_covis_lines = np.empty((0, 2), dtype=np.int32)
        self._cached_curr_T_cw = None

        if not self.enabled:
            print("[VIEWER WARNING] Open3D is not installed. 3D visualization disabled.")

    def start(self):
        """Starts the visualizer in a dedicated thread."""
        if not self.enabled:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the visualizer thread."""
        self.is_running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def update(self, keyframes, map_points, current_frame=None):
        """
        Thread-safe state update called from Tracking/Mapping pipeline.
        """
        if not self.enabled:
            return

        # 1. Extract valid 3D map points
        valid_mps = [mp.pos for mp in map_points if mp is not None and not getattr(mp, 'is_bad', False)]
        mp_positions = np.array(valid_mps, dtype=np.float32) if valid_mps else np.empty((0, 3), dtype=np.float32)

        # 2. Extract Keyframe camera centers and covisibility edges
        valid_kfs = [kf for kf in keyframes if not getattr(kf, 'is_bad', False)]
        kf_centers = []
        kf_to_idx = {}

        for idx, kf in enumerate(valid_kfs):
            center = kf.get_camera_center()
            kf_centers.append(center)
            kf_to_idx[kf.id] = idx

        kf_positions = np.array(kf_centers, dtype=np.float32) if kf_centers else np.empty((0, 3), dtype=np.float32)

        # Build covisibility edge list
        covis_lines = []
        for kf in valid_kfs:
            if kf.id in kf_to_idx:
                idx_a = kf_to_idx[kf.id]
                for neighbor, weight in kf.connected_keyframes.items():
                    if neighbor.id in kf_to_idx and weight >= 20 and kf.id < neighbor.id:
                        idx_b = kf_to_idx[neighbor.id]
                        covis_lines.append([idx_a, idx_b])

        covis_lines = np.array(covis_lines, dtype=np.int32) if covis_lines else np.empty((0, 2), dtype=np.int32)

        # 3. Current frame pose
        curr_T_cw = np.copy(current_frame.T_cw) if current_frame is not None and hasattr(current_frame, 'T_cw') else None

        # Lock and swap cache
        with self.lock:
            self._cached_mp_positions = mp_positions
            self._cached_kf_centers = kf_positions
            self._cached_covis_lines = covis_lines
            self._cached_curr_T_cw = curr_T_cw

    def _create_frustum_lines(self, T_cw, scale=0.1, color=(0.0, 1.0, 0.0)):
        """Builds a LineSet representing a 3D camera wireframe frustum."""
        R_wc = T_cw[:3, :3].T
        C_w = -R_wc @ T_cw[:3, 3]

        w = scale * 0.8
        h = scale * 0.6
        z = scale

        corners_c = np.array([
            [0, 0, 0],
            [-w, -h, z],
            [w, -h, z],
            [w,  h, z],
            [-w,  h, z]
        ], dtype=np.float32)

        corners_w = (R_wc @ corners_c.T).T + C_w

        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4], [4, 1]
        ]
        colors = [color for _ in range(len(lines))]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners_w)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def _run(self):
        """Open3D rendering loop."""
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.window_name, width=self.width, height=self.height)

        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.12])
        opt.point_size = 3.0

        vis.add_geometry(self.pcd_map_points)
        vis.add_geometry(self.lines_trajectory)
        vis.add_geometry(self.lines_covisibility)
        vis.add_geometry(self.frustum_current)

        first_view = True

        while self.is_running:
            with self.lock:
                mps = self._cached_mp_positions
                kfs = self._cached_kf_centers
                cov_lines = self._cached_covis_lines
                curr_T_cw = self._cached_curr_T_cw

            if len(mps) > 0:
                self.pcd_map_points.points = o3d.utility.Vector3dVector(mps)
                colors = np.full((len(mps), 3), 0.8, dtype=np.float64)
                self.pcd_map_points.colors = o3d.utility.Vector3dVector(colors)
                vis.update_geometry(self.pcd_map_points)

            if len(kfs) >= 2:
                traj_indices = [[i, i + 1] for i in range(len(kfs) - 1)]
                self.lines_trajectory.points = o3d.utility.Vector3dVector(kfs)
                self.lines_trajectory.lines = o3d.utility.Vector2iVector(traj_indices)
                traj_colors = [[0.2, 0.4, 1.0] for _ in range(len(traj_indices))]
                self.lines_trajectory.colors = o3d.utility.Vector3dVector(traj_colors)
                vis.update_geometry(self.lines_trajectory)

            if len(kfs) > 0 and len(cov_lines) > 0:
                self.lines_covisibility.points = o3d.utility.Vector3dVector(kfs)
                self.lines_covisibility.lines = o3d.utility.Vector2iVector(cov_lines)
                cov_colors = [[0.0, 0.8, 0.8] for _ in range(len(cov_lines))]
                self.lines_covisibility.colors = o3d.utility.Vector3dVector(cov_colors)
                vis.update_geometry(self.lines_covisibility)

            if curr_T_cw is not None:
                frustum = self._create_frustum_lines(curr_T_cw, scale=0.15, color=(0.0, 1.0, 0.0))
                self.frustum_current.points = frustum.points
                self.frustum_current.lines = frustum.lines
                self.frustum_current.colors = frustum.colors
                vis.update_geometry(self.frustum_current)

            if first_view and len(mps) > 10:
                vis.reset_view_point(True)
                first_view = False

            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.03)

        vis.destroy_window()