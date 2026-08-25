import numpy as np
import cv2


class ImageDrawer:
    """
    2D Feature Tracking Visualizer for ORB-SLAM3.
    Renders extracted keypoints, tracked MapPoints, optical flow vectors,
    and a real-time HUD showing SLAM state telemetry.
    """

    def __init__(self, window_name="ORB-SLAM3 2D Tracking Feed"):
        self.window_name = window_name

        # Color palette (BGR)
        self.COLOR_TRACKED = (0, 255, 0)      # Green: Actively tracked MapPoint
        self.COLOR_UNMATCHED = (0, 0, 255)    # Red: Unmatched FAST keypoint
        self.COLOR_FLOW = (0, 255, 255)       # Yellow: Motion vector line
        self.COLOR_TEXT = (255, 255, 255)     # White text
        self.COLOR_PANEL_BG = (20, 20, 20)    # Dark grey HUD background

    def draw_frame(self, image, current_frame, tracking_state_str="OK", atlas=None, last_frame=None):
        """
        Draws 2D tracking features and HUD overlay on top of input camera image.
        
        image: (H, W) or (H, W, 3) numpy array
        current_frame: instance of src.datastructures.frame.Frame
        tracking_state_str: String state ('OK', 'NOT_INITIALIZED', 'LOST')
        atlas: instance of src.datastructures.atlas.Atlas (optional)
        last_frame: instance of Frame for motion tail rendering (optional)
        
        Returns: (H, W, 3) uint8 canvas with rendered features
        """
        if image is None:
            return None

        # Convert grayscale to 3-channel BGR
        if len(image.shape) == 2:
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = np.copy(image)

        if current_frame is None or not hasattr(current_frame, 'kps') or not current_frame.kps:
            self._draw_hud(canvas, 0, 0, tracking_state_str, atlas)
            return canvas

        kps = current_frame.kps
        mps = current_frame.map_points if hasattr(current_frame, 'map_points') else [None] * len(kps)

        n_tracked = 0
        n_total = len(kps)

        # 1. Draw Optical Flow Motion Tails (from last_frame to current_frame)
        if last_frame is not None and hasattr(last_frame, 'map_points'):
            last_mp_to_pt = {}
            for idx, mp in enumerate(last_frame.map_points):
                if mp is not None and not getattr(mp, 'is_bad', False):
                    last_mp_to_pt[mp.id] = last_frame.kps[idx].pt

            for idx, mp in enumerate(mps):
                if mp is not None and not getattr(mp, 'is_bad', False) and mp.id in last_mp_to_pt:
                    p_curr = (int(round(kps[idx].pt[0])), int(round(kps[idx].pt[1])))
                    p_prev = (int(round(last_mp_to_pt[mp.id][0])), int(round(last_mp_to_pt[mp.id][1])))
                    cv2.line(canvas, p_prev, p_curr, self.COLOR_FLOW, 1, cv2.LINE_AA)

        # 2. Draw Keypoints
        for idx, kp in enumerate(kps):
            pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
            mp = mps[idx] if idx < len(mps) else None

            if mp is not None and not getattr(mp, 'is_bad', False):
                # Tracked 3D Landmark -> Green Circle
                n_tracked += 1
                cv2.circle(canvas, pt, radius=3, color=self.COLOR_TRACKED, thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, pt, radius=5, color=(0, 180, 0), thickness=1, lineType=cv2.LINE_AA)
            else:
                # Unmatched FAST feature -> Small Red Dot
                cv2.circle(canvas, pt, radius=2, color=self.COLOR_UNMATCHED, thickness=-1, lineType=cv2.LINE_AA)

        # 3. Draw Telemetry HUD
        self._draw_hud(canvas, n_tracked, n_total, tracking_state_str, atlas)
        return canvas

    def _draw_hud(self, canvas, n_tracked, n_total, tracking_state_str, atlas):
        """Draws telemetry panel at top-left corner."""
        h, w = canvas.shape[:2]
        panel_h, panel_w = 110, 310

        # Semi-transparent background box
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), self.COLOR_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (10, 10), (10 + panel_w, 10 + panel_h), (80, 80, 80), 1)

        # State color badge
        state_color = self.COLOR_TRACKED if tracking_state_str == "OK" else (0, 165, 255) if "INIT" in tracking_state_str else self.COLOR_UNMATCHED

        # Telemetry text lines
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, f"ORB-SLAM3 [Monocular]", (20, 32), font, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
        
        cv2.putText(canvas, "STATE: ", (20, 55), font, 0.45, self.COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{tracking_state_str}", (80, 55), font, 0.5, state_color, 2, cv2.LINE_AA)

        cv2.putText(canvas, f"Tracked Features: {n_tracked} / {n_total}", (20, 78), font, 0.45, self.COLOR_TEXT, 1, cv2.LINE_AA)

        if atlas is not None:
            active_map = atlas.get_active_map()
            active_map_id = active_map.id if active_map else 0
            total_kfs = atlas.count_keyframes()
            total_mps = atlas.count_map_points()
            cv2.putText(canvas, f"Map #{active_map_id} | KFs: {total_kfs} | MPs: {total_mps}", (20, 100), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    def show(self, canvas, wait_ms=1):
        """Displays canvas in an OpenCV window."""
        cv2.imshow(self.window_name, canvas)
        return cv2.waitKey(wait_ms) & 0xFF