import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


class BundleAdjustment:
    """
    CPU-optimized Bundle Adjustment using SciPy Least Squares.
    Handles Motion-Only BA, Local BA, and full Global BA with Huber robust loss.
    """

    def __init__(self, K):
        self.K = np.array(K, dtype=np.float64)
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])

    # -------------------------------------------------------------------------
    # 1. MOTION-ONLY BUNDLE ADJUSTMENT
    # -------------------------------------------------------------------------
    def motion_only_ba(self, frame, max_iterations=15, max_nfev=None):
        """
        Optimizes ONLY the 6-DoF camera pose of current frame.
        All 3D MapPoints are held fixed.
        """
        nfev = max_nfev if max_nfev is not None else max_iterations * 20

        pts_3d = []
        pts_2d = []

        for idx, mp in enumerate(frame.map_points):
            if mp is not None and not getattr(mp, 'is_bad', False):
                pts_3d.append(mp.pos)
                pts_2d.append(frame.kps[idx].pt)

        if len(pts_3d) < 5:
            return False

        pts_3d = np.ascontiguousarray(np.array(pts_3d, dtype=np.float64))
        pts_2d = np.ascontiguousarray(np.array(pts_2d, dtype=np.float64))

        rvec_init, _ = cv2.Rodrigues(frame.T_cw[:3, :3].astype(np.float64))
        tvec_init = frame.T_cw[:3, 3].astype(np.float64)
        params_init = np.hstack([rvec_init.ravel(), tvec_init.ravel()])

        def residuals_motion_only(params):
            rvec = params[:3]
            tvec = params[3:]
            proj, _ = cv2.projectPoints(pts_3d, rvec, tvec, self.K, None)
            proj = proj.reshape(-1, 2)
            return (proj - pts_2d).ravel()

        res = least_squares(
            residuals_motion_only,
            params_init,
            method='trf',
            loss='huber',
            f_scale=1.0,
            ftol=1e-6,
            xtol=1e-6,
            max_nfev=nfev
        )

        if not res.success and res.status < 1:
            return False

        rvec_opt = res.x[:3]
        tvec_opt = res.x[3:]
        R_opt, _ = cv2.Rodrigues(rvec_opt)

        frame.T_cw[:3, :3] = R_opt.astype(np.float32)
        frame.T_cw[:3, 3] = tvec_opt.astype(np.float32)
        return True

    # -------------------------------------------------------------------------
    # 2. LOCAL BUNDLE ADJUSTMENT
    # -------------------------------------------------------------------------
    def local_bundle_adjustment(self, local_keyframes, fixed_keyframes, local_map_points, max_iterations=30, max_nfev=None):
        """
        Jointly optimizes poses of `local_keyframes` and 3D coordinates of `local_map_points`.
        `fixed_keyframes` provide anchoring geometry and are held constant.
        """
        nfev = max_nfev if max_nfev is not None else max_iterations * 20

        if not local_keyframes or not local_map_points:
            return

        # Map active keyframes and map points to unique indices
        kf_to_idx = {kf.id: i for i, kf in enumerate(local_keyframes)}
        n_local_kfs = len(local_keyframes)

        valid_mps = [mp for mp in local_map_points if not getattr(mp, 'is_bad', False) and mp.n_obs >= 2]
        mp_to_idx = {mp.id: j for j, mp in enumerate(valid_mps)}
        n_mps = len(valid_mps)

        if n_mps == 0:
            return

        # Parameter vector: [6 * n_local_kfs (poses)] + [3 * n_mps (3D points)]
        n_params = 6 * n_local_kfs + 3 * n_mps
        params_init = np.zeros(n_params, dtype=np.float64)

        for kf in local_keyframes:
            i = kf_to_idx[kf.id]
            rvec, _ = cv2.Rodrigues(kf.T_cw[:3, :3].astype(np.float64))
            tvec = kf.T_cw[:3, 3].astype(np.float64)
            params_init[6 * i : 6 * i + 3] = rvec.ravel()
            params_init[6 * i + 3 : 6 * i + 6] = tvec.ravel()

        for mp in valid_mps:
            j = mp_to_idx[mp.id]
            params_init[6 * n_local_kfs + 3 * j : 6 * n_local_kfs + 3 * j + 3] = mp.pos

        # Collect observations: (kf_id, is_local, mp_id, measured_2d)
        fixed_kf_poses = {kf.id: kf.T_cw.astype(np.float64) for kf in fixed_keyframes}
        observations = []

        for mp in valid_mps:
            for kf, kp_idx in mp.observations.items():
                if getattr(kf, 'is_bad', False):
                    continue
                if kp_idx >= len(kf.kps):
                    continue

                if kf.id in kf_to_idx:
                    measured_2d = np.array(kf.kps[kp_idx].pt, dtype=np.float64)
                    observations.append((kf.id, True, mp.id, measured_2d))
                elif kf.id in fixed_kf_poses:
                    measured_2d = np.array(kf.kps[kp_idx].pt, dtype=np.float64)
                    observations.append((kf.id, False, mp.id, measured_2d))

        n_obs = len(observations)
        if n_obs < 5:
            return

        # Build Sparse Jacobian Structure
        A_sparse = lil_matrix((2 * n_obs, n_params), dtype=int)
        for obs_idx, (kf_id, is_local, mp_id, _) in enumerate(observations):
            row = 2 * obs_idx
            mp_idx = mp_to_idx[mp_id]

            col_mp = 6 * n_local_kfs + 3 * mp_idx
            A_sparse[row : row + 2, col_mp : col_mp + 3] = 1

            if is_local:
                kf_idx = kf_to_idx[kf_id]
                col_kf = 6 * kf_idx
                A_sparse[row : row + 2, col_kf : col_kf + 6] = 1

        A_sparse = A_sparse.tocsr()

        def residuals_local_ba(params):
            residuals = np.empty(2 * n_obs, dtype=np.float64)
            poses = params[: 6 * n_local_kfs]
            points_3d = params[6 * n_local_kfs :].reshape((n_mps, 3))

            rot_cache = {}
            for i in range(n_local_kfs):
                rvec = poses[6 * i : 6 * i + 3]
                R, _ = cv2.Rodrigues(rvec)
                tvec = poses[6 * i + 3 : 6 * i + 6]
                rot_cache[i] = (R, tvec)

            for obs_idx, (kf_id, is_local, mp_id, measured_2d) in enumerate(observations):
                mp_idx = mp_to_idx[mp_id]
                P_w = points_3d[mp_idx]

                if is_local:
                    k_i = kf_to_idx[kf_id]
                    R, tvec = rot_cache[k_i]
                else:
                    T_fixed = fixed_kf_poses[kf_id]
                    R = T_fixed[:3, :3]
                    tvec = T_fixed[:3, 3]

                P_c = R @ P_w + tvec
                z = P_c[2]

                if z > 0.01:
                    inv_z = 1.0 / z
                    proj_x = self.fx * (P_c[0] * inv_z) + self.cx
                    proj_y = self.fy * (P_c[1] * inv_z) + self.cy
                else:
                    proj_x = -9999.0
                    proj_y = -9999.0

                residuals[2 * obs_idx] = proj_x - measured_2d[0]
                residuals[2 * obs_idx + 1] = proj_y - measured_2d[1]

            return residuals

        res = least_squares(
            residuals_local_ba,
            params_init,
            jac_sparsity=A_sparse,
            method='trf',
            loss='huber',
            f_scale=1.0,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
            max_nfev=nfev
        )

        opt_poses = res.x[: 6 * n_local_kfs]
        opt_points = res.x[6 * n_local_kfs :].reshape((n_mps, 3))

        for kf in local_keyframes:
            i = kf_to_idx[kf.id]
            rvec = opt_poses[6 * i : 6 * i + 3]
            tvec = opt_poses[6 * i + 3 : 6 * i + 6]
            R, _ = cv2.Rodrigues(rvec)
            kf.T_cw[:3, :3] = R.astype(np.float32)
            kf.T_cw[:3, 3] = tvec.astype(np.float32)

        for mp in valid_mps:
            j = mp_to_idx[mp.id]
            mp.pos = opt_points[j].astype(np.float32)

    # -------------------------------------------------------------------------
    # 3. GLOBAL BUNDLE ADJUSTMENT
    # -------------------------------------------------------------------------
    def global_bundle_adjustment(self, keyframes, map_points, max_iterations=30, max_nfev=None):
        """
        Runs full graph optimization across ALL keyframes and map points in the map.
        The origin keyframe (id=0 or first in list) is held fixed as the gauge anchor.
        
        keyframes: list of Keyframe instances
        map_points: list of MapPoint instances
        """
        valid_kfs = [kf for kf in keyframes if not getattr(kf, 'is_bad', False)]
        if len(valid_kfs) < 2:
            return

        # The first keyframe serves as the fixed anchor to fix the gauge freedom (world origin)
        anchor_kf = valid_kfs[0]
        optimizable_kfs = valid_kfs[1:]

        # Run Local BA framework on the entire map with the anchor fixed
        self.local_bundle_adjustment(
            local_keyframes=optimizable_kfs,
            fixed_keyframes=[anchor_kf],
            local_map_points=map_points,
            max_iterations=max_iterations,
            max_nfev=max_nfev
        )