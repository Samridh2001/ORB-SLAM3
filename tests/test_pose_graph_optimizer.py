import os
import sys
import numpy as np

# Dynamic path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datastructures.frame import Frame
from src.datastructures.keyframe import Keyframe
from src.optimization.pose_graph_optimizer import PoseGraphOptimizer


def test_pose_graph_optimizer():
    optimizer = PoseGraphOptimizer()

    # 1. Simulate a closed square trajectory of 5 Keyframes (0 -> 1 -> 2 -> 3 -> 4 -> 0)
    gt_positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0]  # Returns to origin
    ]

    keyframes = []
    for i, pos in enumerate(gt_positions):
        f = Frame([], None, timestamp=float(i))
        f.T_cw = np.eye(4, dtype=np.float32)
        f.T_cw[:3, 3] = -np.array(pos, dtype=np.float32)
        kf = Keyframe(f)
        keyframes.append(kf)

    # 2. Introduce +0.20m drift on the closing keyframe (KF4)
    keyframes[4].T_cw[0, 3] -= 0.20

    pre_loop_error = np.linalg.norm(keyframes[4].T_cw[:3, 3] - keyframes[0].T_cw[:3, 3])
    print(f"Pre-Optimization Loop Closure Gap: {pre_loop_error:.4f} m")

    # 3. Define Loop Edge between KF0 and KF4 (measured relative pose is Identity)
    S_04_measured = np.eye(4, dtype=np.float32)
    loop_edges = [(keyframes[4], keyframes[0], S_04_measured)]

    # 4. Run 7-DoF Sim(3) Pose Graph Optimization
    optimizer.optimize_essential_graph(
        keyframes=keyframes,
        loop_edges=loop_edges,
        max_iterations=40
    )

    post_loop_error = np.linalg.norm(keyframes[4].T_cw[:3, 3] - keyframes[0].T_cw[:3, 3])
    print(f"Post-Optimization Loop Closure Gap: {post_loop_error:.4f} m")

    assert post_loop_error < pre_loop_error, "Pose Graph Optimization failed to reduce loop gap!"
    assert post_loop_error < 0.01, "Loop closure residual error is too high!"
    print("[TEST POSE GRAPH OPTIMIZER SUCCESS]")


if __name__ == "__main__":
    test_pose_graph_optimizer()