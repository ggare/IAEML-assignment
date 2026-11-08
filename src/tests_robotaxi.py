"""Sanity tests for the robotaxi environment (Assignment 1).

Run from the repository ``src/`` directory:

    python tests_robotaxi.py

No pytest dependency: plain functions with asserts, printed results.
"""

import time

import equinox
import jax
import jax.numpy as jnp
import numpy as np

from env.robotaxi import RoboTaxiEnv
from utils.scripted_driver import ScriptedDriver

KEY = jax.random.PRNGKey(0)


def make_env(map_id=0, max_steps=600, **overrides):
    return RoboTaxiEnv.init_params(
        key=KEY, map_id=map_id, max_steps=max_steps, path_length=100,
        perception_radius=5.0, num_ray_sensors=32, overrides=overrides,
    )


def rollout(p, state, action_fn, max_steps=2000):
    """Run an episode; returns (final info, total reward, states list)."""
    obs, s = RoboTaxiEnv.reset(KEY, p, state)
    ret, done, states = 0.0, False, []
    while not done and len(states) < max_steps:
        a = action_fn(obs, s)
        obs, s, r, done, info = RoboTaxiEnv.step(KEY, s, jnp.asarray(a), p)
        ret += float(r)
        states.append(s)
    return info, ret, states


# ---------------------------------------------------------------------------


def test_observation_space():
    p, s0 = make_env()
    obs, _ = RoboTaxiEnv.reset(KEY, p, s0)
    vec = obs.as_vector()
    assert vec.shape == (RoboTaxiEnv.observation_dim(p),), vec.shape
    assert vec.shape == (139,), vec.shape  # 11 + 4 * 32 rays
    assert bool(jnp.all(jnp.isfinite(vec)))
    assert obs.collision_rays.shape == (32, 2)
    assert obs.ray_hit_velocities.shape == (32, 2)
    print("PASS  observation space: dim =", vec.shape[0],
          "| finite | rays (32, 2) | hit velocities (32, 2)")


def test_zero_action_stays():
    p, s0 = make_env()
    info, ret, states = rollout(p, s0, lambda o, s: jnp.zeros(2), max_steps=120)
    s = states[-1]
    assert np.allclose(np.asarray(s.agent_pos), np.asarray(s0.agent_pos), atol=1e-4)
    assert float(jnp.linalg.norm(s.vel_body)) < 1e-3
    assert not bool(info["collision"])
    print("PASS  zero action: the car stays put, no NaNs")


def _physics_sim(p, n_steps, speed_fn, steer_fn, vec0=None):
    """Pure-physics rollout (no obstacles): python-side controllers."""
    vec = jnp.zeros(6) if vec0 is None else vec0
    out = []
    for _ in range(n_steps):
        vx = float(vec[3])
        a, delta = speed_fn(vx), steer_fn(vx)
        vec = RoboTaxiEnv._integrate(vec, a, delta, p)
        out.append(np.asarray(vec))
    return np.stack(out)


def test_terminal_speed():
    p, s0 = make_env()
    # analytic terminal speed: a = c_d v^2 + c_roll  =>  v = sqrt(2.4/0.3)
    v_theory = np.sqrt((p.max_accel - p.roll_coef) / p.drag_coef)
    traj = _physics_sim(p, 1200, lambda vx: p.max_accel, lambda vx: 0.0)  # 20 s
    v_meas = traj[-1, 3]
    assert abs(v_meas - v_theory) < 0.05, (v_meas, v_theory)
    assert traj[-1, 4] == 0.0  # no lateral drift on a straight line
    print(f"PASS  terminal speed: measured {v_meas:.3f} vs analytic {v_theory:.3f}")


def test_circle_radius():
    p, s0 = make_env()
    steer = 0.25
    r_kin = (p.lf + p.lr) / np.tan(steer)  # kinematic bicycle prediction
    # simple speed keeper holding v = 1.6 while steering constantly
    traj = _physics_sim(
        p, 2700,
        lambda vx: 1.5 * (1.6 - vx) + 0.3 * 1.6 ** 2 + p.roll_coef,  # P + drag feedforward
        lambda vx: steer,
    )
    tail = traj[-1400:]  # steady part: more than a full circle
    v, om = tail[:, 3], tail[:, 5]
    assert np.std(om) < 0.02, f"yaw rate did not converge: std={np.std(om):.3f}"
    r_dyn = float(np.mean(v) / np.mean(om))
    # tire slip makes the dynamic radius larger than the kinematic one
    assert r_kin * 0.95 < r_dyn < r_kin * 1.8, (r_dyn, r_kin)
    # the trajectory itself should be a circle (least-squares circle fit)
    xy = tail[:, 0:2]
    A = np.stack([xy[:, 0], xy[:, 1], np.ones(len(xy))], axis=1)
    b = (xy ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r_fit = float(np.sqrt(sol[2] + cx ** 2 + cy ** 2))
    residual = np.abs(np.linalg.norm(xy - np.array([cx, cy]), axis=1) - r_fit)
    assert residual.max() < 0.05, f"not circular: max residual {residual.max():.3f}"
    assert abs(r_fit - r_dyn) < 0.1, (r_fit, r_dyn)
    print(f"PASS  circle test: R_dynamic = {r_dyn:.2f} (fit {r_fit:.2f}) vs "
          f"R_kinematic = {r_kin:.2f}; steady slip angle "
          f"{np.degrees(np.arctan2(tail[:, 4].mean(), tail[:, 3].mean())):.1f} deg")


def test_collision_and_freeze():
    p, s0 = make_env()
    # drive north from the start into the top wall at full throttle
    info, ret, states = rollout(
        p, s0, lambda o, s: jnp.array([1.0, 0.0]), max_steps=600
    )
    assert bool(info["collision"]) and not bool(info["success"])
    assert float(jnp.linalg.norm(states[-1].vel_body)) == 0.0  # frozen after crash
    assert ret < 0.0
    print(f"PASS  collision: crash detected at t={int(info['time'])}, "
          f"velocity frozen, return = {ret:.1f}")


def test_goal_reached():
    p, s0 = make_env()
    # teleport the agent close to the goal, facing it
    goal = jnp.asarray(s0.goal_pos)
    start = goal - jnp.array([2.5, 0.0])
    near = equinox.tree_at(
        lambda s: (s.agent_pos, s.agent_forward_dir, s.heading,
                   s.prev_dist_to_goal),
        s0,
        (start, jnp.array([1.0, 0.0]), jnp.asarray(0.0),
         jnp.linalg.norm(start - goal)),
    )
    info, ret, states = rollout(
        p, near, lambda o, s: jnp.array([0.7, 0.0]), max_steps=400
    )
    assert bool(info["success"]) and not bool(info["collision"])
    assert ret > 20.0
    print(f"PASS  goal: reached in {int(info['time'])} steps, return = {ret:.1f}")


def test_reward_ordering():
    """success > doing nothing > crashing into a wall."""
    p, s0 = make_env(max_steps=1200)

    driver = ScriptedDriver(p)
    _, ret_scripted, _ = rollout(p, s0, lambda o, s: driver.act(o, s))
    _, ret_idle, _ = rollout(p, s0, lambda o, s: jnp.zeros(2))
    _, ret_crash, _ = rollout(p, s0, lambda o, s: jnp.array([1.0, 0.0]))

    assert ret_scripted > ret_idle > ret_crash, (
        ret_scripted, ret_idle, ret_crash)
    print(f"PASS  reward ordering: success {ret_scripted:+.1f} > "
          f"idle {ret_idle:+.1f} > crash {ret_crash:+.1f}")


def test_vmap_and_speed():
    p, s0 = make_env(max_steps=300)
    B = 8
    # different constant actions for every batch element
    actions = jnp.stack(
        [jnp.array([0.8, -0.8 + 1.6 * i / (B - 1)]) for i in range(B)]
    )
    bstates = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (B,) + np.shape(x)), s0
    )
    vstep = jax.jit(
        jax.vmap(RoboTaxiEnv.step, in_axes=(None, 0, 0, None)),
        static_argnames=("env_params",),
    )
    headings = []
    for _ in range(200):
        _, bstates, _, _, _ = vstep(KEY, bstates, actions, p)
    headings = np.asarray(bstates.heading)
    assert headings.shape == (B,)
    assert np.std(headings) > 0.1  # actions actually did something different

    # throughput of a single env (warmup first: params differ from other tests)
    obs, s = RoboTaxiEnv.reset(KEY, p, s0)
    obs, s, r, d, i = RoboTaxiEnv.step(KEY, s, jnp.array([0.5, 0.1]), p)
    jax.block_until_ready(s.agent_pos)
    t0 = time.time()
    for _ in range(300):
        obs, s, r, d, i = RoboTaxiEnv.step(KEY, s, jnp.array([0.5, 0.1]), p)
    jax.block_until_ready(s.agent_pos)
    single = 300 / (time.time() - t0)

    t0 = time.time()
    for _ in range(300):
        _, bstates, _, _, _ = vstep(KEY, bstates, actions, p)
    jax.block_until_ready(bstates.agent_pos)
    batched = 300 * B / (time.time() - t0)

    print(f"PASS  vmap: 8 envs in parallel, headings diverged; "
          f"throughput: {single:.0f} steps/s single, "
          f"{batched:.0f} steps/s batched (x{B})")


if __name__ == "__main__":
    test_observation_space()
    test_zero_action_stays()
    test_terminal_speed()
    test_circle_radius()
    test_collision_and_freeze()
    test_goal_reached()
    test_reward_ordering()
    test_vmap_and_speed()
    print("\nAll tests passed.")
