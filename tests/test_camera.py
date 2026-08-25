import os
import sys
import numpy as np
import cv2

# Dynamic path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sensors.camera import PinholeCamera


def test_camera_model():
    # 1. Initialize Pinhole Camera with Barrel Distortion
    cam = PinholeCamera(
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        k1=-0.1,  # Radial barrel distortion
        k2=0.01,
        p1=0.001,
        p2=0.0
    )

    # 2. Test 3D Projection
    p3D = np.array([0.5, 0.2, 2.0], dtype=np.float64)
    uv_distorted = cam.project(p3D)
    assert uv_distorted is not None, "Projection returned None for point in front of camera!"
    assert cam.is_in_image(uv_distorted), "Projected point outside camera bounds!"
    print(f"3D Point: {p3D} -> Distorted 2D Pixel: {uv_distorted}")

    # 3. Test Ray Unprojection (Invert Distortion)
    ray = cam.unproject_to_ray(uv_distorted)
    expected_ray = np.array([p3D[0] / p3D[2], p3D[1] / p3D[2], 1.0], dtype=np.float32)
    ray_error = np.linalg.norm(ray - expected_ray)
    print(f"Recovered Normalized Ray Error: {ray_error:.6f}")
    assert ray_error < 1e-4, "Ray unprojection failed to recover accurate bearing vector!"

    # 4. Test Keypoint Undistortion
    kps = [cv2.KeyPoint(float(uv_distorted[0]), float(uv_distorted[1]), 31.0, octave=2)]
    undist_kps = cam.undistort_keypoints(kps)

    # Ideal pinhole projection without distortion:
    u_ideal = cam.fx * (p3D[0] / p3D[2]) + cam.cx
    v_ideal = cam.fy * (p3D[1] / p3D[2]) + cam.cy

    undist_err = np.linalg.norm(np.array([undist_kps[0].pt[0], undist_kps[0].pt[1]]) - np.array([u_ideal, v_ideal]))
    print(f"Undistorted Keypoint Error vs Ideal Pinhole: {undist_err:.4f} px")
    assert undist_err < 0.1, "Keypoint undistortion did not match ideal pinhole position!"
    assert undist_kps[0].octave == 2, "Keypoint octave was not preserved during undistortion!"

    print("[TEST CAMERA MODEL SUCCESS]")


if __name__ == "__main__":
    test_camera_model()