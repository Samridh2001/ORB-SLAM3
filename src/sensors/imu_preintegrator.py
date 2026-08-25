import numpy as np
import cv2


class IMUPreintegrator:
    """
    On-Manifold SO(3) IMU Pre-integration engine for ORB-SLAM3.
    Implements continuous pre-integration of relative rotation, velocity, and position,
    with first-order bias Jacobians and 15x15 noise covariance propagation.
    """

    def __init__(self, bg_init=None, ba_init=None, noise_params=None):
        # Biases at time of integration
        self.bg = np.zeros(3, dtype=np.float64) if bg_init is None else np.array(bg_init, dtype=np.float64)
        self.ba = np.zeros(3, dtype=np.float64) if ba_init is None else np.array(ba_init, dtype=np.float64)

        # Preintegrated measurements
        self.delta_R = np.eye(3, dtype=np.float64)
        self.delta_v = np.zeros(3, dtype=np.float64)
        self.delta_p = np.zeros(3, dtype=np.float64)
        self.dt_sum = 0.0

        # Noise parameters (continuous standard deviations)
        if noise_params is None:
            self.sigma_g = 1e-3     # rad / s / sqrt(Hz)
            self.sigma_a = 1e-2     # m / s^2 / sqrt(Hz)
            self.sigma_gw = 1e-5    # rad / s^2 / sqrt(Hz)
            self.sigma_aw = 1e-4    # m / s^3 / sqrt(Hz)
        else:
            self.sigma_g = float(noise_params.get('sigma_g', 1e-3))
            self.sigma_a = float(noise_params.get('sigma_a', 1e-2))
            self.sigma_gw = float(noise_params.get('sigma_gw', 1e-5))
            self.sigma_aw = float(noise_params.get('sigma_aw', 1e-4))

        # Continuous sensor noise covariance matrix Q (12x12)
        # [n_g (3), n_a (3), n_gw (3), n_aw (3)]
        self.Q = np.diag([
            self.sigma_g**2, self.sigma_g**2, self.sigma_g**2,
            self.sigma_a**2, self.sigma_a**2, self.sigma_a**2,
            self.sigma_gw**2, self.sigma_gw**2, self.sigma_gw**2,
            self.sigma_aw**2, self.sigma_aw**2, self.sigma_aw**2
        ]).astype(np.float64)

        # State Covariance Sigma (15x15)
        # Order: [d_theta (3), d_v (3), d_p (3), d_bg (3), d_ba (3)]
        self.cov = np.zeros((15, 15), dtype=np.float64)

        # First-order Jacobians w.r.t biases
        self.J_R_bg = np.zeros((3, 3), dtype=np.float64)
        self.J_v_bg = np.zeros((3, 3), dtype=np.float64)
        self.J_v_ba = np.zeros((3, 3), dtype=np.float64)
        self.J_p_bg = np.zeros((3, 3), dtype=np.float64)
        self.J_p_ba = np.zeros((3, 3), dtype=np.float64)

        # IMU measurements buffer
        self.measurements = []

    def reset(self, bg_new=None, ba_new=None):
        """Resets the pre-integrator state with updated bias estimates."""
        if bg_new is not None:
            self.bg = np.array(bg_new, dtype=np.float64)
        if ba_new is not None:
            self.ba = np.array(ba_new, dtype=np.float64)

        self.delta_R = np.eye(3, dtype=np.float64)
        self.delta_v = np.zeros(3, dtype=np.float64)
        self.delta_p = np.zeros(3, dtype=np.float64)
        self.dt_sum = 0.0

        self.cov.fill(0.0)

        self.J_R_bg.fill(0.0)
        self.J_v_bg.fill(0.0)
        self.J_v_ba.fill(0.0)
        self.J_p_bg.fill(0.0)
        self.J_p_ba.fill(0.0)

        self.measurements.clear()

    @staticmethod
    def skew(w):
        """Computes 3x3 skew-symmetric matrix from 3D vector."""
        return np.array([
            [0.0, -w[2], w[1]],
            [w[2], 0.0, -w[0]],
            [-w[1], w[0], 0.0]
        ], dtype=np.float64)

    @staticmethod
    def exp_so3(w):
        """Exponential map from so(3) Lie algebra to SO(3) Lie group."""
        theta = np.linalg.norm(w)
        if theta < 1e-7:
            return np.eye(3, dtype=np.float64) + IMUPreintegrator.skew(w)
        R, _ = cv2.Rodrigues(w.astype(np.float64))
        return R

    @staticmethod
    def right_jacobian_so3(w):
        """Computes Right Jacobian Jr(w) for SO(3)."""
        theta = np.linalg.norm(w)
        if theta < 1e-7:
            return np.eye(3, dtype=np.float64) - 0.5 * IMUPreintegrator.skew(w)

        k = w / theta
        K = IMUPreintegrator.skew(k)
        Jr = (np.sin(theta) / theta) * np.eye(3) + \
             (1.0 - np.sin(theta) / theta) * np.outer(k, k) - \
             ((1.0 - np.cos(theta)) / theta) * K
        return Jr

    def integrate_measurement(self, acc, gyro, dt):
        """
        Integrates a single IMU measurement (acc: m/s^2, gyro: rad/s, dt: sec).
        Updates delta_R, delta_v, delta_p, Jacobians, and 15x15 covariance.
        """
        if dt <= 0.0:
            return

        self.measurements.append((np.copy(acc), np.copy(gyro), float(dt)))
        self.dt_sum += dt

        acc = np.array(acc, dtype=np.float64)
        gyro = np.array(gyro, dtype=np.float64)

        # Unbiased measurements
        w_unbiased = gyro - self.bg
        a_unbiased = acc - self.ba

        # Step 1: Incremental rotation matrix delta_R_step = Exp(w * dt)
        step_rot_vec = w_unbiased * dt
        dR_step = self.exp_so3(step_rot_vec)
        Jr = self.right_jacobian_so3(step_rot_vec)

        # Cache previous state for state transition matrix F
        R_prev = np.copy(self.delta_R)
        v_prev = np.copy(self.delta_v)
        dt2 = 0.5 * dt * dt

        # Step 2: Update preintegrated states
        self.delta_p += v_prev * dt + 0.5 * (R_prev @ a_unbiased) * dt2
        self.delta_v += (R_prev @ a_unbiased) * dt
        self.delta_R = R_prev @ dR_step

        # Step 3: Discrete-time State Transition Matrix F (15x15)
        F = np.eye(15, dtype=np.float64)
        F[0:3, 0:3] = dR_step.T
        F[0:3, 9:12] = -Jr * dt
        F[3:6, 0:3] = -R_prev @ self.skew(a_unbiased) * dt
        F[3:6, 12:15] = -R_prev * dt
        F[6:9, 0:3] = -0.5 * R_prev @ self.skew(a_unbiased) * dt2
        F[6:9, 3:6] = np.eye(3) * dt
        F[6:9, 12:15] = -0.5 * R_prev * dt2

        # Step 4: Noise Input Matrix G (15x12)
        G = np.zeros((15, 12), dtype=np.float64)
        G[0:3, 0:3] = Jr * dt
        G[3:6, 3:6] = R_prev * dt
        G[6:9, 3:6] = 0.5 * R_prev * dt2
        G[9:12, 6:9] = np.eye(3) * dt
        G[12:15, 9:12] = np.eye(3) * dt

        # Step 5: Covariance propagation: Cov = F * Cov * F^T + G * (Q/dt) * G^T
        self.cov = F @ self.cov @ F.T + G @ (self.Q / dt) @ G.T

        # Step 6: Bias Jacobians propagation
        self.J_p_bg += self.J_v_bg * dt - 0.5 * (R_prev @ self.skew(a_unbiased) @ self.J_R_bg) * dt2
        self.J_p_ba += self.J_v_ba * dt - 0.5 * R_prev * dt2
        self.J_v_bg += - (R_prev @ self.skew(a_unbiased) @ self.J_R_bg) * dt
        self.J_v_ba += - R_prev * dt
        self.J_R_bg = dR_step.T @ self.J_R_bg - Jr * dt

    def get_delta_rotation(self, dbg=None):
        """Corrected relative rotation with gyro bias correction delta_bg."""
        if dbg is None or np.allclose(dbg, 0):
            return self.delta_R
        return self.delta_R @ self.exp_so3(self.J_R_bg @ np.array(dbg, dtype=np.float64))

    def get_delta_velocity(self, dbg=None, dba=None):
        """Corrected relative velocity with bias corrections."""
        v = np.copy(self.delta_v)
        if dbg is not None:
            v += self.J_v_bg @ np.array(dbg, dtype=np.float64)
        if dba is not None:
            v += self.J_v_ba @ np.array(dba, dtype=np.float64)
        return v

    def get_delta_position(self, dbg=None, dba=None):
        """Corrected relative position with bias corrections."""
        p = np.copy(self.delta_p)
        if dbg is not None:
            p += self.J_p_bg @ np.array(dbg, dtype=np.float64)
        if dba is not None:
            p += self.J_p_ba @ np.array(dba, dtype=np.float64)
        return p