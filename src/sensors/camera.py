import numpy as np
import cv2
import yaml


class PinholeCamera:
    """
    Pinhole Camera Model with Brown-Conrady Radial-Tangential Distortion for ORB-SLAM3.
    Supports 3D-to-2D projection, 2D-to-3D ray lifting, keypoint undistortion,
    and YAML configuration parsing.
    """

    def __init__(self, width, height, fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = int(width)
        self.height = int(height)

        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

        # Distortion coefficients: [k1, k2, p1, p2, k3]
        self.dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

        # 3x3 Intrinsic Matrix K and its inverse
        self.K = np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0]
        ], dtype=np.float64)

        self.K_inv = np.linalg.inv(self.K)

    @classmethod
    def from_yaml(cls, yaml_path):
        """Loads camera intrinsics from a YAML configuration file."""
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        camera_cfg = cfg.get('Camera', cfg)

        width = camera_cfg.get('width', 640)
        height = camera_cfg.get('height', 480)
        fx = camera_cfg['fx']
        fy = camera_cfg['fy']
        cx = camera_cfg['cx']
        cy = camera_cfg['cy']

        k1 = camera_cfg.get('k1', 0.0)
        k2 = camera_cfg.get('k2', 0.0)
        p1 = camera_cfg.get('p1', 0.0)
        p2 = camera_cfg.get('p2', 0.0)
        k3 = camera_cfg.get('k3', 0.0)

        return cls(width, height, fx, fy, cx, cy, k1, k2, p1, p2, k3)

    def project(self, p3D_c):
        """
        Projects a 3D camera-frame point [X, Y, Z] to distorted 2D pixel coordinates [u, v].
        Returns: (u, v) or None if point is behind camera.
        """
        p3D_c = np.asarray(p3D_c, dtype=np.float64)
        if p3D_c[2] <= 0.001:
            return None

        # Normalized coordinates
        x_n = p3D_c[0] / p3D_c[2]
        y_n = p3D_c[1] / p3D_c[2]

        r2 = x_n * x_n + y_n * y_n
        r4 = r2 * r2
        r6 = r4 * r2

        k1, k2, p1, p2, k3 = self.dist_coeffs

        # Radial distortion
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

        # Tangential distortion
        x_dist = x_n * radial + 2.0 * p1 * x_n * y_n + p2 * (r2 + 2.0 * x_n * x_n)
        y_dist = y_n * radial + p1 * (r2 + 2.0 * y_n * y_n) + 2.0 * p2 * x_n * y_n

        # Pixel projection
        u = self.fx * x_dist + self.cx
        v = self.fy * y_dist + self.cy

        return np.array([u, v], dtype=np.float32)

    def unproject_to_ray(self, pt2D):
        """
        Lifts a 2D pixel coordinate to a normalized 3D bearing vector [x_n, y_n, 1.0].
        """
        pt2D = np.asarray(pt2D, dtype=np.float64).reshape(-1, 1, 2)
        undist_pt = cv2.undistortPoints(pt2D, self.K, self.dist_coeffs)
        x_n, y_n = undist_pt[0, 0]
        return np.array([x_n, y_n, 1.0], dtype=np.float32)

    def undistort_keypoints(self, keypoints):
        """
        Undistorts a list of cv2.KeyPoint objects, returning a new list with corrected (u, v).
        """
        if not keypoints:
            return []

        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
        # cv2.undistortPoints with P=self.K outputs pixel coordinates
        undist_pts = cv2.undistortPoints(pts, self.K, self.dist_coeffs, P=self.K)
        undist_pts = undist_pts.reshape(-1, 2)

        undistorted_kps = []
        for i, kp in enumerate(keypoints):
            new_kp = cv2.KeyPoint(
                x=float(undist_pts[i, 0]),
                y=float(undist_pts[i, 1]),
                size=kp.size,
                angle=kp.angle,
                response=kp.response,
                octave=kp.octave,
                class_id=kp.class_id
            )
            undistorted_kps.append(new_kp)

        return undistorted_kps

    def is_in_image(self, pt2D, margin=0):
        """Checks if a 2D pixel point lies inside the camera sensor boundaries."""
        u, v = pt2D[0], pt2D[1]
        return (margin <= u < self.width - margin) and (margin <= v < self.height - margin)