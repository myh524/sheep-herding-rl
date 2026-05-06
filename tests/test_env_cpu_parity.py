"""观测快照与向量化 Boids 相对 legacy 的数值一致性。"""
import numpy as np

from envs.defaults import HERDER_PHYSICS_LEGACY_OVERRIDES
from envs.sheep_scenario import SheepScenario


def _reference_observation_full(s: SheepScenario) -> np.ndarray:
    """与 get_observation / _observation_snapshot 等价的逐行实现（用于对照）。"""
    nh = s.num_herders
    dim = 10
    obs = np.zeros(dim, dtype=np.float32)
    target = s.target_position.astype(np.float32)
    r_max = float(max(s.world_radius, 1e-6))

    def _fill_herder_centroid(o: np.ndarray) -> None:
        if nh <= 0:
            o[8] = np.float32(0.0)
            o[9] = np.float32(0.0)
            return
        fc = s.get_flock_center().astype(np.float32)
        hc = np.mean(s.herder_positions.astype(np.float32), axis=0)
        rel = hc - fc
        rn = float(np.linalg.norm(rel))
        o[8] = np.clip(
            np.float32(rn / r_max), np.float32(-10.0), np.float32(10.0)
        )
        if rn > 1e-6:
            ang = float(np.arctan2(float(rel[1]), float(rel[0])) / np.pi)
        else:
            ang = 0.0
        o[9] = np.clip(np.float32(ang), np.float32(-1.0), np.float32(1.0))

    if not s.sheep:
        _fill_herder_centroid(obs)
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

    positions = np.array([sh.position for sh in s.sheep], dtype=np.float32)
    velocities = np.array([sh.velocity for sh in s.sheep], dtype=np.float32)
    flock_center = np.mean(positions, axis=0).astype(np.float32)

    max_sheep_speed = float(s.sheep_config.get("max_speed", 3.0))
    if max_sheep_speed <= 0:
        max_sheep_speed = 1.0

    to_target = target - flock_center
    d_fc = float(np.linalg.norm(to_target))
    obs[0] = np.clip(
        np.float32(d_fc / r_max), np.float32(-10.0), np.float32(10.0)
    )
    if d_fc > 1e-6:
        obs[1] = np.clip(
            np.float32(
                np.arctan2(float(to_target[1]), float(to_target[0])) / np.pi
            ),
            np.float32(-1.0),
            np.float32(1.0),
        )
    else:
        obs[1] = np.float32(0.0)

    mean_velocity = np.mean(velocities, axis=0).astype(np.float32)
    v_sp = float(np.linalg.norm(mean_velocity))
    obs[2] = np.clip(
        np.float32(v_sp / max_sheep_speed), np.float32(-1.0), np.float32(1.0)
    )
    if v_sp > 1e-6:
        obs[3] = np.clip(
            np.float32(
                np.arctan2(float(mean_velocity[1]), float(mean_velocity[0])) / np.pi
            ),
            np.float32(-1.0),
            np.float32(1.0),
        )
    else:
        obs[3] = np.float32(0.0)

    rel = positions.astype(np.float64) - target.astype(np.float64).reshape(1, 2)
    e0 = float(np.max(rel[:, 0]))
    e1 = float(np.max(rel[:, 1]))
    e2 = float(np.max(-rel[:, 0]))
    e3 = float(np.max(-rel[:, 1]))
    obs[4] = np.clip(np.float32(e0 / r_max), np.float32(-10.0), np.float32(10.0))
    obs[5] = np.clip(np.float32(e1 / r_max), np.float32(-10.0), np.float32(10.0))
    obs[6] = np.clip(np.float32(e2 / r_max), np.float32(-10.0), np.float32(10.0))
    obs[7] = np.clip(np.float32(e3 / r_max), np.float32(-10.0), np.float32(10.0))

    _fill_herder_centroid(obs)

    return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)


def test_observation_snapshot_matches_reference():
    np.random.seed(42)
    s = SheepScenario(
        world_size=(60.0, 60.0),
        num_sheep=6,
        num_herders=3,
        random_seed=42,
        use_herder_kinematics=True,
        herder_motion=dict(HERDER_PHYSICS_LEGACY_OVERRIDES),
    )
    s.reset(target_position=None)
    for _ in range(15):
        s.set_herder_targets(
            np.array([[10.0, 8.0], [-12.0, 10.0], [5.0, -14.0]], dtype=np.float32)
        )
        s.update_herders(0.1)
        s.update_sheep(0.1)

    obs_snap, fc_snap, rs = s._observation_snapshot()
    obs_ref = _reference_observation_full(s)
    np.testing.assert_allclose(obs_snap, obs_ref, rtol=1e-5, atol=1e-5)

    r_shared = float(max(s.world_radius, 1e-6))
    flock_center = s.get_flock_center()
    tail = []
    for i in range(s.num_herders):
        rel = (s.herder_positions[i] - flock_center).astype(np.float32)
        rn = float(np.linalg.norm(rel))
        pr = np.clip(np.float32(rn / r_shared), np.float32(-10.0), np.float32(10.0))
        if rn > 1e-6:
            pa = np.clip(
                np.float32(np.arctan2(float(rel[1]), float(rel[0])) / np.pi),
                np.float32(-1.0),
                np.float32(1.0),
            )
        else:
            pa = np.float32(0.0)
        tail.append(np.array([pr, pa], dtype=np.float32))
    shared_ref = np.concatenate([obs_ref] + tail).astype(np.float32)
    shared_new = s.get_shared_observation()
    np.testing.assert_allclose(shared_new, shared_ref, rtol=1e-5, atol=1e-5)

    assert fc_snap.shape == (2,)
    assert rs > 0


def test_vectorized_sheep_matches_legacy():
    np.random.seed(7)

    leg = SheepScenario(
        world_size=(50.0, 50.0),
        num_sheep=8,
        num_herders=3,
        random_seed=7,
        use_herder_kinematics=False,
    )
    leg.reset(target_position=None)
    leg.vectorized_sheep_updates = False

    vec = SheepScenario(
        world_size=(50.0, 50.0),
        num_sheep=8,
        num_herders=3,
        random_seed=7,
        use_herder_kinematics=False,
    )
    vec.reset(target_position=None)
    vec.vectorized_sheep_updates = True

    dt = 0.05
    for step in range(80):
        h = np.array(
            [
                [10.0 + 0.05 * step, 6.0],
                [-8.0, 10.0 + 0.04 * step],
                [4.0, -9.0],
            ],
            dtype=np.float32,
        )
        leg.set_herder_targets(h)
        vec.set_herder_targets(h.copy())
        leg.update_herders(dt)
        vec.update_herders(dt)
        leg.update_sheep(dt)
        vec.update_sheep(dt)

    for sl, sv in zip(leg.sheep, vec.sheep):
        np.testing.assert_allclose(sl.position, sv.position, rtol=1e-4, atol=5e-4)
        np.testing.assert_allclose(sl.velocity, sv.velocity, rtol=1e-4, atol=5e-4)
