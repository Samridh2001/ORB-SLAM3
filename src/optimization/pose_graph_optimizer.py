import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


class PoseGraphOptimizer:
    """
    7-DoF Sim(3) Pose Graph Optimizer for ORB-SLAM3.
    Optimizes keyframe poses across the Essential Graph upon loop closure
    to correct scale, orientation, and translation drift before Global BA.
    """

    def __init__(self):
        pass

    @staticmethod
    def _sim3_to_params(S):
        """
        Converts a 4x4 Sim(3) matrix to a 7-parameter vector:
        [rvec (3,), tvec (3,), log_scale (1,)]
        """
        det = np.linalg.det(S[:3, :3])
        s = np.cbrt(det) if det > 1e-7 else 1.0
        if s <= 1e-6:
            s = 1.0
        R = S[:3, :3] / s
        rvec, _ = cv2.Rodrigues(R.astype(np.float64))
        tvec = S[:3, 3].astype(np.float64)
        log_s = np.log(s)
        return np.hstack([rvec.ravel(), tvec.ravel(), log_s])

    @staticmethod
    def _params_to_sim3(params):
        """
        Converts a 7-parameter vector back to a 4x4 Sim(3) matrix.
        """
        rvec = params[:3]
        tvec = params[3:6]
        log_s = params[6]
        s = np.exp(log_s)

        R, _ = cv2.Rodrigues(rvec)
        S = np.eye(4, dtype=np.float64)
        S[:3, :3] = s * R
        S[:3, 3] = tvec
        return S

    @staticmethod
    def _invert_sim3(S):
        """Computes inverse of a 4x4 Sim(3) matrix."""
        det = np.linalg.det(S[:3, :3])
        s = np.cbrt(det) if det > 1e-7 else 1.0
        if s <= 1e-6:
            s = 1.0
        R = S[:3, :3] / s
        t = S[:3, 3]

        S_inv = np.eye(4, dtype=np.float64)
        R_inv = R.T
        s_inv = 1.0 / s

        S_inv[:3, :3] = s_inv * R_inv
        S_inv[:3, 3] = -s_inv * (R_inv @ t)
        return S_inv

    def optimize_essential_graph(self, keyframes, loop_edges, max_iterations=40):
        """
        Optimizes poses of `keyframes` over the Essential Graph.
        
        keyframes: list of Keyframe instances
        loop_edges: list of tuples (kf_i, kf_j, S_ji_measured) representing loop closures
        """
        valid_kfs = [kf for kf in keyframes if not getattr(kf, 'is_bad', False)]
        if len(valid_kfs) < 2:
            return

        # Fixed anchor is the initial keyframe (id=0 or first in list)
        anchor_kf = valid_kfs[0]
        opt_kfs = [kf for kf in valid_kfs if kf != anchor_kf]

        kf_to_idx = {kf.id: i for i, kf in enumerate(opt_kfs)}
        n_opt = len(opt_kfs)

        if n_opt == 0:
            return

        # 1. Initialize parameter vector (7 * n_opt)
        params_init = np.zeros(7 * n_opt, dtype=np.float64)
        for kf in opt_kfs:
            i = kf_to_idx[kf.id]
            S_cw = np.copy(kf.T_cw).astype(np.float64)
            params_init[7 * i : 7 * i + 7] = self._sim3_to_params(S_cw)

        # 2. Build graph edges: (kf_parent, kf_child, S_child_parent_relative, weight)
        edges = []

        # (a) Spanning tree / Sequential odometry edges (weight = 1.0)
        for i in range(len(valid_kfs) - 1):
            kf_a = valid_kfs[i]
            kf_b = valid_kfs[i + 1]
            T_aw_inv = self._invert_sim3(kf_a.T_cw.astype(np.float64))
            S_ba = kf_b.T_cw.astype(np.float64) @ T_aw_inv
            edges.append((kf_a, kf_b, S_ba, 1.0))

        # (b) Covisibility high-weight edges (> 50 shared points)
        for kf_a in valid_kfs:
            for kf_b, weight in kf_a.connected_keyframes.items():
                if kf_a.id < kf_b.id and weight >= 50 and not getattr(kf_b, 'is_bad', False):
                    T_aw_inv = self._invert_sim3(kf_a.T_cw.astype(np.float64))
                    S_ba = kf_b.T_cw.astype(np.float64) @ T_aw_inv
                    edges.append((kf_a, kf_b, S_ba, 1.0))

        # (c) Loop closure edges (higher confidence weight = 10.0)
        for kf_i, kf_j, S_ji in loop_edges:
            edges.append((kf_i, kf_j, S_ji.astype(np.float64), 10.0))

        n_edges = len(edges)
        if n_edges == 0:
            return

        # 3. Build Sparse Jacobian Matrix
        A_sparse = lil_matrix((7 * n_edges, 7 * n_opt), dtype=int)
        for edge_idx, (kf_a, kf_b, _, _) in enumerate(edges):
            row = 7 * edge_idx
            if kf_a.id in kf_to_idx:
                idx_a = kf_to_idx[kf_a.id]
                A_sparse[row : row + 7, 7 * idx_a : 7 * idx_a + 7] = 1
            if kf_b.id in kf_to_idx:
                idx_b = kf_to_idx[kf_b.id]
                A_sparse[row : row + 7, 7 * idx_b : 7 * idx_b + 7] = 1

        A_sparse = A_sparse.tocsr()
        anchor_S = anchor_kf.T_cw.astype(np.float64)

        # 4. Residual computation function
        def residuals_sim3(params):
            residuals = np.empty(7 * n_edges, dtype=np.float64)

            sim3_cache = {}
            for kf in opt_kfs:
                i = kf_to_idx[kf.id]
                sim3_cache[kf.id] = self._params_to_sim3(params[7 * i : 7 * i + 7])

            for edge_idx, (kf_a, kf_b, S_ba_measured, weight) in enumerate(edges):
                S_aw = sim3_cache[kf_a.id] if kf_a.id in sim3_cache else anchor_S
                S_bw = sim3_cache[kf_b.id] if kf_b.id in sim3_cache else anchor_S

                S_aw_inv = self._invert_sim3(S_aw)
                S_ba_est = S_bw @ S_aw_inv

                S_ba_meas_inv = self._invert_sim3(S_ba_measured)
                E = S_ba_est @ S_ba_meas_inv

                det_E = np.linalg.det(E[:3, :3])
                s_E = np.cbrt(det_E) if det_E > 1e-7 else 1.0
                if s_E <= 1e-6:
                    s_E = 1.0

                r_err, _ = cv2.Rodrigues(E[:3, :3] / s_E)
                t_err = E[:3, 3]
                log_s_err = np.log(s_E)

                residuals[7 * edge_idx : 7 * edge_idx + 3] = weight * r_err.ravel()
                residuals[7 * edge_idx + 3 : 7 * edge_idx + 6] = weight * t_err.ravel()
                residuals[7 * edge_idx + 6] = weight * log_s_err

            return residuals

        # 5. Run Optimization
        res = least_squares(
            residuals_sim3,
            params_init,
            jac_sparsity=A_sparse,
            method='trf',
            loss='huber',
            f_scale=1.0,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
            max_nfev=max_iterations * 30
        )

        # 6. Unpack optimized poses & propagate to 3D map points
        for kf in opt_kfs:
            i = kf_to_idx[kf.id]
            S_opt = self._params_to_sim3(res.x[7 * i : 7 * i + 7])

            det_opt = np.linalg.det(S_opt[:3, :3])
            s_opt = np.cbrt(det_opt) if det_opt > 1e-7 else 1.0
            if s_opt <= 1e-6:
                s_opt = 1.0

            R_opt = S_opt[:3, :3] / s_opt
            t_opt = S_opt[:3, 3]

            T_cw_old = np.copy(kf.T_cw)

            kf.T_cw[:3, :3] = R_opt.astype(np.float32)
            kf.T_cw[:3, 3] = (t_opt / s_opt).astype(np.float32)

            T_diff = kf.T_cw @ np.linalg.inv(T_cw_old)
            for mp in kf.map_points:
                if mp is not None and not getattr(mp, 'is_bad', False):
                    if mp.observations and next(iter(mp.observations.keys())) == kf:
                        mp.pos = (T_diff[:3, :3] @ mp.pos + T_diff[:3, 3]).astype(np.float32)