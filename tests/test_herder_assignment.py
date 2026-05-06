"""牧者–编队槽位最小总路程分配（匈牙利）性质检验。"""
import numpy as np

from envs.sheep_scenario import assign_herders_to_slots


def test_min_cost_total_distance_le_ordered_swap():
    H = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    S = np.array([[10.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    out_ordered = assign_herders_to_slots(H, S, "ordered")
    out_min = assign_herders_to_slots(H, S, "min_cost")
    d_ordered = float(
        sum(
            np.linalg.norm(H[i] - out_ordered[i])
            for i in range(2)
        )
    )
    d_min = float(
        sum(np.linalg.norm(H[i] - out_min[i]) for i in range(2))
    )
    assert d_min <= d_ordered + 1e-5
    assert d_min < 1.0


def test_ordered_is_identity_permutation():
    H = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    S = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    out = assign_herders_to_slots(H, S, "ordered")
    np.testing.assert_allclose(out, S)


def test_three_herders_hand_optimal():
    """3×3：手算可知最优与恒等不同。"""
    H = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    S = np.array([[2.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    out = assign_herders_to_slots(H, S, "min_cost")
    d = float(sum(np.linalg.norm(H[i] - out[i]) for i in range(3)))
    assert d < 1e-5
