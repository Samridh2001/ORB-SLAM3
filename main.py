import argparse
import glob
import os
import sys
import time
import cv2
import numpy as np
import yaml

# Dynamic root path resolution
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.sensors.camera import PinholeCamera
from src.features.orb_extractor import ORBExtractor
from src.datastructures.frame import Frame
from src.datastructures.atlas import Atlas
from src.threads.tracking import Tracking, TrackingState
from src.threads.local_mapping import LocalMapping
from src.threads.loop_closing import LoopClosing
from visualization.image_drawer import ImageDrawer
from visualization.viewer_3d import Viewer3D


class ORBSLAM3:
    """
    Main ORB-SLAM3 System Orchestrator.
    Coordinates feature extraction, tracking, local mapping, loop closing,
    multi-map atlas management, and 2D/3D visualization.
    """

    def __init__(self, camera_config_path, orb_config_path=None, use_viewer=True):
        print("=" * 60)
        print("         ORB-SLAM3 Monocular System Initializing            ")
        print("=" * 60)

        # 1. Load Camera Calibration
        if os.path.exists(camera_config_path):
            self.camera = PinholeCamera.from_yaml(camera_config_path)
            print(f"[SYSTEM] Camera model loaded from '{camera_config_path}' (fx={self.camera.fx:.1f}, fy={self.camera.fy:.1f})")
        else:
            print(f"[SYSTEM WARNING] Config '{camera_config_path}' not found. Using default 640x480 pinhole.")
            self.camera = PinholeCamera(640, 480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)

        # 2. Load ORB Extractor Parameters
        n_features = 1000
        scale_factor = 1.2
        n_levels = 8
        ini_th_fast = 20
        min_th_fast = 7

        if orb_config_path and os.path.exists(orb_config_path):
            with open(orb_config_path, 'r') as f:
                orb_cfg = yaml.safe_load(f).get('ORBExtractor', {})
                n_features = orb_cfg.get('nFeatures', n_features)
                scale_factor = orb_cfg.get('scaleFactor', scale_factor)
                n_levels = orb_cfg.get('nLevels', n_levels)
                ini_th_fast = orb_cfg.get('iniThFAST', ini_th_fast)
                min_th_fast = orb_cfg.get('minThFAST', min_th_fast)

        self.extractor = ORBExtractor(
            n_features=n_features,
            scale_factor=scale_factor,
            n_levels=n_levels,
            ini_th_fast=ini_th_fast,
            min_th_fast=min_th_fast
        )
        print(f"[SYSTEM] ORB Extractor initialized ({n_features} features, {n_levels} pyramid levels).")

        # 3. Initialize Multi-Map Atlas
        self.atlas = Atlas(K=self.camera.K)

        # 4. Initialize Core Threads
        self.tracker = Tracking(K=self.camera.K)
        self.local_mapping = LocalMapping(K=self.camera.K, min_shared_points=15)
        self.loop_closing = LoopClosing(atlas=self.atlas, K=self.camera.K, min_inliers=20)

        # 5. Initialize Visualizers
        self.image_drawer = ImageDrawer(window_name="ORB-SLAM3 2D Tracking")
        self.viewer_3d = Viewer3D(window_name="ORB-SLAM3 3D Map") if use_viewer else None
        if self.viewer_3d:
            self.viewer_3d.start()

        self.last_keyframe_id = -1
        self.last_frame = None
        self.frame_count = 0

    def process_image(self, image_input, timestamp=None):
        """
        Processes a single input camera frame through the SLAM pipeline.
        image_input: Grayscale (H, W) or BGR (H, W, 3) numpy array
        timestamp: float timestamp in seconds
        """
        t_start = time.time()
        self.frame_count += 1
        t_stamp = timestamp if timestamp is not None else float(self.frame_count) / 30.0

        # Convert to grayscale if needed
        if len(image_input.shape) == 3:
            gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_input

        # 1. Feature Extraction (Quadtree FAST + Rotated BRIEF)
        kps_raw, des = self.extractor.extract(gray)

        # 2. Keypoint Undistortion
        kps_undistorted = self.camera.undistort_keypoints(kps_raw)

        # 3. Create Frame Datastructure
        current_frame = Frame(kps_undistorted, des, t_stamp)
        current_frame.id=self.frame_count

        # 4. Process Tracking
        state = self.tracker.process_frame(current_frame)
        state_str = state.name

        # 5. Check for New Keyframe insertion
        if self.tracker.keyframes and self.tracker.keyframes[-1].id != self.last_keyframe_id:
            latest_kf = self.tracker.keyframes[-1]
            self.last_keyframe_id = latest_kf.id

            # Add Keyframe to Atlas Active Map
            self.atlas.get_active_map().add_keyframe(latest_kf)

            # Pass Keyframe to Local Mapping and Loop Closing Queues
            self.local_mapping.add_keyframe(latest_kf)
            self.loop_closing.add_keyframe(latest_kf)

            # Execute Loop Closing Step ONLY when a new keyframe is born
            self.loop_closing.process()

        # 6. Execute Local Mapping Step (Triangulation, Point/Keyframe Culling, Local BA)
        self.local_mapping.process(self.tracker.map_points, self.tracker.keyframes)

        # 7. Update 3D Viewer
        if self.viewer_3d:
            self.viewer_3d.update(
                keyframes=self.tracker.keyframes,
                map_points=self.tracker.map_points,
                current_frame=current_frame
            )

        t_elapsed = time.time() - t_start
        fps = 1.0 / t_elapsed if t_elapsed > 0 else 0.0

        # ---> FIX START: Sync active map points to the Atlas for the HUD viewer <---
        active_map = self.atlas.get_active_map()
        if active_map is not None:
            active_map.map_points = [mp for mp in self.tracker.map_points if mp is not None]
        # ---> FIX END <---

        # 8. Render 2D Tracking HUD Canvas
        canvas = self.image_drawer.draw_frame(
            image=image_input,
            current_frame=current_frame,
            tracking_state_str=f"{state_str} ({fps:.1f} FPS)",
            atlas=self.atlas,
            last_frame=self.last_frame
        )

        self.last_frame = current_frame
        return canvas, state

    def shutdown(self):
        """Clean shutdown for threads and visualization windows."""
        print("[SYSTEM] Shutting down ORB-SLAM3 pipeline...")
        if self.viewer_3d:
            self.viewer_3d.stop()
        cv2.destroyAllWindows()
        
        # ---> FIX START: Count accurate totals directly from Tracker on shutdown <---
        kf_count = len(self.tracker.keyframes)
        mp_count = len([mp for mp in self.tracker.map_points if mp is not None])
        print(f"[SYSTEM] Finished. Total Keyframes: {kf_count}, Total MapPoints: {mp_count}")
        # ---> FIX END <---


# -----------------------------------------------------------------------------
# Input Feed Drivers (Synthetic, Video File, Webcam, Dataset Folder)
# -----------------------------------------------------------------------------
def run_synthetic_demo(system, num_frames=60):
    """Simulates a camera traversing a 3D synthetic point cloud cube."""
    print(f"\n[DEMO] Running Built-in Synthetic Trajectory Demo ({num_frames} frames)...")
    K = system.camera.K

    np.random.seed(42)
    pts3D = np.random.uniform(-1.5, 1.5, (150, 3)).astype(np.float32)
    pts3D[:, 2] += 4.0

    for i in range(num_frames):
        T_cw = np.eye(4, dtype=np.float32)
        T_cw[0, 3] = -np.sin(i * 0.08) * 0.5
        T_cw[1, 3] = -np.cos(i * 0.08) * 0.2

        # Project 3D points to 2D image
        pts_cam = (T_cw[:3, :3] @ pts3D.T).T + T_cw[:3, 3]
        proj = (K @ pts_cam.T).T
        pts2D = proj[:, :2] / proj[:, 2:]

        # Create canvas with textured corner patches
        img = np.zeros((480, 640), dtype=np.uint8)
        img[:] = 40
        for pt in pts2D:
            u, v = int(round(pt[0])), int(round(pt[1]))
            if 10 <= u < 630 and 10 <= v < 470:
                cv2.circle(img, (u, v), 3, 255, -1)
                cv2.rectangle(img, (u - 4, v - 4), (u + 4, v + 4), 180, 1)

        canvas, _ = system.process_image(img, timestamp=float(i) / 30.0)

        # Live Display
        cv2.imshow("ORB-SLAM3 2D Tracking", canvas)
        if cv2.waitKey(20) & 0xFF == 27:  # ESC to quit
            break


def run_video_stream(system, source):
    """Processes live webcam or video file."""
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    print(f"[INPUT] Streaming from video source: {source} (Press ESC to stop)")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        canvas, _ = system.process_image(frame)
        cv2.imshow("ORB-SLAM3 2D Tracking", canvas)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()


def run_image_sequence(system, images_dir):
    """Processes sorted sequence of images from directory (EuRoC/TUM format)."""
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")) +
                         glob.glob(os.path.join(images_dir, "*.jpg")))

    if not image_paths:
        print(f"[ERROR] No images found in directory: {images_dir}")
        return

    print(f"[INPUT] Processing {len(image_paths)} images from '{images_dir}'...")
    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue

        canvas, _ = system.process_image(img, timestamp=float(idx) / 30.0)
        cv2.imshow("ORB-SLAM3 2D Tracking", canvas)

        if cv2.waitKey(5) & 0xFF == 27:
            break


def main():
    parser = argparse.ArgumentParser(description="ORB-SLAM3 Monocular Python Implementation")
    parser.add_argument("--config", type=str, default="config/camera_intrinsics.yaml", help="Path to camera YAML")
    parser.add_argument("--orb_config", type=str, default="config/orb_params.yaml", help="Path to ORB YAML")
    parser.add_argument("--video", type=str, default=None, help="Path to video file or webcam index (e.g. 0)")
    parser.add_argument("--images", type=str, default=None, help="Directory containing image sequence")
    parser.add_argument("--no_viewer", action="store_true", help="Disable Open3D 3D viewer")

    args = parser.parse_args()

    # Instantiate SLAM System
    slam = ORBSLAM3(
        camera_config_path=args.config,
        orb_config_path=args.orb_config,
        use_viewer=not args.no_viewer
    )

    try:
        if args.video is not None:
            run_video_stream(slam, args.video)
        elif args.images is not None:
            run_image_sequence(slam, args.images)
        else:
            # Default fallback: Built-in synthetic demonstration
            run_synthetic_demo(slam, num_frames=80)
    finally:
        slam.shutdown()


if __name__ == "__main__":
    main()