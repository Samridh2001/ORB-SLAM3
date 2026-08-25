import cv2
import numpy as np


def triangulate_point(P1, P2, point1, point2):
    """
    Triangulates a single 3D point from 2D correspondences using Linear DLT (Ax = 0).
    P1, P2: 3x4 Projection Matrices (K @ [R|t])
    point1, point2: 2D pixel points (x, y)
    Returns: 3D point in World coordinates (x, y, z) as float32 array
    """
    A = np.zeros((4, 4), dtype=np.float32)
    A[0] = point1[0] * P1[2] - P1[0]
    A[1] = point1[1] * P1[2] - P1[1]
    A[2] = point2[0] * P2[2] - P2[0]
    A[3] = point2[1] * P2[2] - P2[1]

    # Singular Value Decomposition (SVD) on CPU
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]

    # Convert from Homogeneous (4D) to Euclidean (3D)
    if abs(X[3]) > 1e-7:
        return (X[:3] / X[3]).astype(np.float32)
    return None

def triangulate_points_vectorized(T1, T2, pts1, pts2, K):
    """
    Fast vectorized triangulation for multiple point pairs using OpenCV primitives.
    T1, T2: 4x4 SE(3) pose matrices
    pts1, pts2: (N, 2) arrays of pixel coordinates
    K: (3, 3) Intrinsic matrix
    Returns: (N, 3) array of 3D points
    """
    P1 = K @ T1[:3, :]
    P2 = K @ T2[:3, :]

    # cv2.triangulatePoints accepts (2, N) and returns (4, N)
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float32), pts2.T.astype(np.float32))
    
    # Normalize by scale coordinate w
    mask = np.abs(pts4D[3]) > 1e-7
    pts3D = np.zeros((len(pts1), 3), dtype=np.float32)
    
    valid_indices = np.where(mask)[0]
    pts3D[valid_indices] = (pts4D[:3, valid_indices] / pts4D[3, valid_indices]).T

    return pts3D

def check_cheirality(R, t, pts1_norm, pts2_norm, max_reproj_err=4.0):
    """
    Checks if triangulated 3D points lie IN FRONT of both camera positions (Positive Depth).
    pts1_norm, pts2_norm: Normalized image coordinates (K^-1 @ [x, y, 1]^T)
    """
    P1 = np.eye(3, 4, dtype=np.float32)
    P2 = np.hstack((R, t.reshape(3, 1))).astype(np.float32)

    n_points = len(pts1_norm)
    good_mask = np.zeros(n_points, dtype=bool)
    pts3D_list = np.zeros((n_points, 3), dtype=np.float32)

    for i in range(n_points):
        p3D = triangulate_point(P1, P2, pts1_norm[i], pts2_norm[i])
        if p3D is None:
            continue

        # Depth check relative to Frame 1
        z1 = p3D[2]
        if z1 <= 0:
            continue

        # Depth check relative to Frame 2
        p3D_c2 = R @ p3D + t.squeeze()
        z2 = p3D_c2[2]
        if z2 <= 0:
            continue

        # Reprojection Error in Frame 2 (normalized coordinates)
        proj2 = p3D_c2[:2] / z2
        err2 = np.sum((proj2 - pts2_norm[i]) ** 2)

        if err2 < max_reproj_err:
            good_mask[i] = True
            pts3D_list[i] = p3D

    return good_mask, pts3D_list