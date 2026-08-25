import os
import sys
import numpy as np

# Dynamic path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sensors.imu_preintegrator import IMUPreintegrator


def test_imu_preintegration():
    preintegrator = IMUPreintegrator()

    dt = 0.01  # 100 Hz IMU
    n_steps = 100

    # ---------------------------------------------------------
    # Test 1: Pure Rotation around Z-axis (pi/2 radians over 1 sec)
    # ---------------------------------------------------------
    w_z = np.pi / 2.0  # 90 deg/s
    acc_static = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    gyro_z = np.array([0.0, 0.0, w_z], dtype=np.float64)

    for _ in range(n_steps):
        preintegrator.integrate_measurement(acc=acc_static, gyro=gyro_z, dt=dt)

    assert np.isclose(preintegrator.dt_sum, 1.0), "Integrated time mismatch!"

    # Expected rotation: 90 deg around Z
    expected_R = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ], dtype=np.float64)

    rot_err = np.linalg.norm(preintegrator.delta_R - expected_R)
    print(f"Pure Rotation Preintegration Error: {rot_err:.5f}")
    assert rot_err < 1e-2, "Rotation preintegration failed!"

    # ---------------------------------------------------------
    # Test 2: Linear Acceleration along X-axis (1.0 m/s^2 over 1 sec)
    # ---------------------------------------------------------
    preintegrator.reset()
    acc_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    gyro_zero = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    for _ in range(n_steps):
        preintegrator.integrate_measurement(acc=acc_x, gyro=gyro_zero, dt=dt)

    # v = a * t = 1.0 m/s
    # p = 0.5 * a * t^2 = 0.5 m
    assert np.isclose(preintegrator.delta_v[0], 1.0, atol=1e-2), "Velocity preintegration mismatch!"
    assert np.isclose(preintegrator.delta_p[0], 0.5, atol=1e-2), "Position preintegration mismatch!"
    print(f"Integrated Velocity: {preintegrator.delta_v[0]:.4f} m/s (Expected: 1.0000)")
    print(f"Integrated Position: {preintegrator.delta_p[0]:.4f} m (Expected: 0.5000)")

    # ---------------------------------------------------------
    # Test 3: First-order Bias Jacobian Validation
    # ---------------------------------------------------------
    # Introduce small bias delta on accelerometer: dba = [0.1, 0.0, 0.0]
    dba = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    v_corrected = preintegrator.get_delta_velocity(dba=dba)
    # Expected v_corrected ~= 1.0 - 0.1 * 1.0 = 0.9 m/s
    assert np.isclose(v_corrected[0], 0.9, atol=1e-2), "Bias Jacobian correction mismatch!"
    print(f"Bias-Corrected Velocity: {v_corrected[0]:.4f} m/s (Expected: 0.9000)")

    # ---------------------------------------------------------
    # Test 4: Covariance Matrix Validity
    # ---------------------------------------------------------
    assert preintegrator.cov.shape == (15, 15), "Covariance matrix shape mismatch!"
    assert np.all(np.diag(preintegrator.cov) > 0.0), "Covariance diagonal must be strictly positive!"

    print("[TEST IMU PREINTEGRATOR SUCCESS]")


if __name__ == "__main__":
    test_imu_preintegration()