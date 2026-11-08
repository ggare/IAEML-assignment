"""Robotaxi environment (IAEML, Assignment 1).

Built on top of ``BaseEnv`` from ``env/base.py``:

* **Dynamics** — dynamic bicycle model. Vehicle state:
  ``(x, y, theta, vx, vy, omega)``, where ``(vx, vy)`` is the velocity in the
  body frame. Actions: longitudinal acceleration and steering angle.
  Tire forces use a linear tire model with saturation (friction circle),
  plus aerodynamic drag and rolling resistance. Integrated with RK4 at
  ``dt = 1 / fps``.
* **Observations** — path hints from the DP navigator (distance to path and
  path direction), 360-degree ray perception with the velocity of the
  obstacle each ray hits, and the ego state of the vehicle.
* **Reward** — dense: potential-based progress towards the goal, path
  following, velocity alignment, control cost, proximity penalty and
  terminal bonuses.

``step`` / ``reset`` keep the ``BaseEnv`` interface, are jit-compiled and
are ``jax.vmap``-compatible, which makes the environment ready for parallel
RL training (Assignment 2).
"""

import jax
from jax import numpy as jnp
import equinox
import chex

from functools import partial
from typing import Tuple, Dict, Any, Optional

from env.base import (
    BaseEnv,
    BaseEnvParams,
    BaseEnvState,
    BaseEnvObservation,
    RADIUS,
    NUM_RAY_SENSORS,
    FOV,
    FPS,
)
from env.tasks import BaseTaskSampler
from utils import utils

# ---------------------------------------------------------------------------
# Defaults (units: map cells, seconds; the vehicle mass is normalized to 1)
# ---------------------------------------------------------------------------

MASS = 1.0        # vehicle mass
IZ = 0.2          # yaw moment of inertia
LF = 0.5          # CG -> front axle distance
LR = 0.5          # CG -> rear axle distance
CF = 12.0         # front cornering stiffness [force / rad]
CR = 12.0         # rear cornering stiffness [force / rad]
MU = 0.9          # tire-road friction coefficient
GRAVITY = 10.0    # gravity
DRAG_COEF = 0.3   # quadratic drag: F = c_d * v * |v|
ROLL_COEF = 0.1   # rolling resistance (saturates at low speed)
V_MAX = 3.0       # hard speed limit
V_FADE = 0.25     # low-speed fade of the lateral tire forces

# action space
MAX_ACCEL = 2.5   # max |longitudinal acceleration| [cells / s^2]
MAX_STEER = 0.6   # max |steering angle| [rad]
NORMALIZE_ACTIONS = True  # if True, both action components live in [-1, 1]
ACTION_REPEAT = 1  # physics substeps of dt = 1/fps per env step

# reward weights
W_PROGRESS = 5.0    # potential-based progress towards the goal
W_PATH = 0.1        # penalty for the distance to the planned path
W_ALIGN = 0.2       # velocity alignment with the path direction
W_TIME = 0.03       # per-step time penalty
W_ACTION = 0.03     # quadratic control cost
W_SAFETY = 0.1      # penalty for moving close to obstacles
R_GOAL = 25.0       # terminal bonus for reaching the goal
R_CRASH = -100.0    # terminal penalty for a collision (safety first)
PATH_DIST_CLIP = 2.0   # clip for the path-distance penalty
PATH_FREEBAND = 0.75   # deadband: path is grid-quantized, so ~1 cell off is normal
SAFE_DIST = 2.0        # distance at which the safety penalty saturates


class RoboTaxiParams(BaseEnvParams):
    """Environment parameters (all leaves must stay static / hashable)."""

    # vehicle
    mass: float = MASS
    iz: float = IZ
    lf: float = LF
    lr: float = LR
    cf: float = CF
    cr: float = CR
    mu: float = MU
    gravity: float = GRAVITY
    drag_coef: float = DRAG_COEF
    roll_coef: float = ROLL_COEF
    v_max: float = V_MAX
    v_fade: float = V_FADE
    # action space
    max_accel: float = MAX_ACCEL
    max_steer: float = MAX_STEER
    normalize_actions: bool = NORMALIZE_ACTIONS
    action_repeat: int = ACTION_REPEAT
    # reward
    w_progress: float = W_PROGRESS
    w_path: float = W_PATH
    w_align: float = W_ALIGN
    w_time: float = W_TIME
    w_action: float = W_ACTION
    w_safety: float = W_SAFETY
    r_goal: float = R_GOAL
    r_crash: float = R_CRASH
    path_dist_clip: float = PATH_DIST_CLIP
    path_freeband: float = PATH_FREEBAND
    safe_dist: float = SAFE_DIST


class RoboTaxiState(BaseEnvState):
    """Env state of BaseEnv + the internal state of the vehicle."""

    heading: chex.Scalar      # theta [rad]
    vel_body: chex.Array      # (vx, vy) velocity in the body frame
    yaw_rate: chex.Scalar     # omega [rad/s]
    collided: chex.Scalar     # bool, sticky collision flag
    prev_dist_to_goal: chex.Scalar  # distance to the goal at the previous step


class RoboTaxiObservation(BaseEnvObservation):
    """Observation of the robotaxi agent.

    Components (``as_vector`` size = 11 + 4 * num_ray_sensors):
      distance_to_path (1), direction_of_path (2), collision_rays (2M),
      ray_hit_velocities (2M), speed (1), vel_body (2), yaw_rate (1),
      distance_to_goal (1), direction_to_goal_body (2), heading_error (1).
    """

    # ego state of the vehicle
    speed: chex.Scalar             # |v|
    vel_body: chex.Array           # (vx, vy) in the body frame
    yaw_rate: chex.Scalar          # omega [rad/s]
    # goal
    distance_to_goal: chex.Scalar
    direction_to_goal_body: chex.Array  # (2,) unit vector in the body frame
    # navigation
    heading_error: chex.Scalar     # signed angle: heading -> path tangent
    # perception extras
    ray_hit_velocities: chex.Array  # (M, 2) velocity of the obstacle each ray hit

    def as_vector(self) -> chex.Array:
        """Concatenate all components into a single flat observation vector.

        Batch-safe: works for a single observation and for a batch produced
        by ``jax.vmap`` (the batch dimension, if present, is preserved).
        """
        # (M, 2) or (B, M, 2) ray blocks -> (2M,) or (B, 2M)
        rays = self.collision_rays
        ray_vels = self.ray_hit_velocities
        rays_flat = rays.reshape(rays.shape[:-2] + (-1,))
        ray_vels_flat = ray_vels.reshape(ray_vels.shape[:-2] + (-1,))
        return jnp.concatenate(
            [
                self.distance_to_path[..., None],
                self.direction_of_path,
                rays_flat,
                ray_vels_flat,
                self.speed[..., None],
                self.vel_body,
                self.yaw_rate[..., None],
                self.distance_to_goal[..., None],
                self.direction_to_goal_body,
                self.heading_error[..., None],
            ],
            axis=-1,
        )


class RoboTaxiEnv(BaseEnv):
    """BaseEnv + dynamic bicycle vehicle, richer observations, dense reward."""

    # ------------------------------------------------------------------
    # API (same signatures and semantics as BaseEnv)
    # ------------------------------------------------------------------

    def init_params(
        key: chex.PRNGKey,
        map_id: int,
        max_steps: int,
        path_length: int,
        discretization_scale: int = 1,
        perception_radius: float = 5,
        num_ray_sensors: int = NUM_RAY_SENSORS,
        fov: float = FOV,
        fps: float = FPS,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[RoboTaxiParams, RoboTaxiState]:
        """Build env params and the initial state for a given map.

        ``overrides`` may replace any ``RoboTaxiParams`` field (vehicle or
        reward settings), e.g. ``overrides={"max_steer": 0.5}``.
        """
        if discretization_scale != 1:
            raise Exception("discretization_scale != 1 is not implemented!")

        task = BaseTaskSampler(map_id)

        kw = dict(
            max_steps_in_episode=max_steps,
            fps=fps,
            step_size=1 / fps,
            map_height_width=(task.height, task.width),
            nav_grid_shape_dtype=(
                (
                    task.height * discretization_scale,
                    task.width * discretization_scale,
                ),
                jnp.float32,
            ),
            path_array_shape_dtype=((path_length, 2), jnp.float32),
            discretization_scale=discretization_scale,
            perception_radius=perception_radius,
            agent_FOV=fov,
            num_ray_sensors=num_ray_sensors,
        )
        if overrides:
            kw.update(overrides)
        env_params = RoboTaxiParams(**kw)

        shape, dtype = env_params.path_array_shape_dtype

        convert_to_world_f = partial(
            utils.convert_to_world_view, map_shape=env_params.map_height_width
        )
        convert_to_world_vmap = jax.vmap(convert_to_world_f, in_axes=(0))

        agent_pos = convert_to_world_f(task.agent_pos).astype(jnp.float32)
        goal_pos = convert_to_world_f(task.goal_pos).astype(jnp.float32)
        forward = task.agent_forward_dir.astype(jnp.float32)
        # initial heading from the default forward direction [0, 1] -> pi/2
        heading = jnp.arctan2(forward[1], forward[0]).astype(jnp.float32)

        init_state = RoboTaxiState(
            time=jnp.asarray(0),
            goal_pos=goal_pos,
            agent_pos=agent_pos,
            agent_forward_dir=forward,
            static_obstacles=convert_to_world_vmap(task.static_obstacles).astype(
                jnp.float32
            ),
            kinematic_obstacles=convert_to_world_vmap(task.kinematic_obstacles).astype(
                jnp.float32
            ),
            kinematic_obst_velocities=task.kinematic_obst_velocities.astype(
                jnp.float32
            ),
            path_array=jnp.zeros(shape=shape, dtype=dtype),
            heading=heading,
            vel_body=jnp.zeros((2,), dtype=jnp.float32),
            yaw_rate=jnp.asarray(0.0),
            collided=jnp.asarray(False),
            prev_dist_to_goal=jnp.linalg.norm(agent_pos - goal_pos),
        )
        obstacles = jnp.concatenate(
            [init_state.static_obstacles, init_state.kinematic_obstacles], axis=0
        )
        path_array = BaseEnv._find_path(
            agent_pos, goal_pos, obstacles, env_params
        )
        init_state = equinox.tree_at(lambda t: t.path_array, init_state, path_array)

        return env_params, init_state

    @partial(jax.jit, static_argnames=("env_params",))
    def reset(
        key: chex.PRNGKey, env_params: RoboTaxiParams, init_state: RoboTaxiState
    ) -> Tuple[RoboTaxiObservation, RoboTaxiState]:
        """Reset the env to ``init_state`` and return the first observation."""
        if init_state is None:
            raise Exception("init_state is None is Not implemented!")
        state = init_state
        # NOTE: BaseEnv.reset hard-codes BaseEnv.get_observation, so reset is
        # overridden here to use the robotaxi observation instead.
        obs = RoboTaxiEnv.get_observation(state, env_params)
        return obs, state

    @partial(jax.jit, static_argnames=("env_params",))
    def step(
        key: chex.PRNGKey,
        env_state: RoboTaxiState,
        action: chex.Array,
        env_params: RoboTaxiParams,
    ) -> Tuple[RoboTaxiObservation, RoboTaxiState, chex.Scalar, chex.Array, Dict]:
        """One env step: action -> physics -> (obs, state, reward, done, info).

        The action is ``(acceleration, steering)``; with the default
        ``normalize_actions=True`` both components are in ``[-1, 1]`` and are
        scaled by ``max_accel`` / ``max_steer``.
        """
        accel, steer = RoboTaxiEnv._prepare_action(action, env_params)

        # kinematic obstacles move once per env step (same speed as in BaseEnv)
        kinematic_obstacles = RoboTaxiEnv._move_kinematic_obstacles(
            env_state, env_params
        )
        obstacles = jnp.concatenate(
            [env_state.static_obstacles, kinematic_obstacles], axis=0
        )

        # --- physics: `action_repeat` RK4 substeps of dt = 1/fps ---
        def substep(carry, _):
            vec, collided = carry
            vec = RoboTaxiEnv._integrate(vec, accel, steer, env_params)
            hit = BaseEnv._check_collisions(vec[0:2], obstacles)
            collided = jnp.logical_or(collided, hit)
            # a crash freezes the vehicle in place
            vec = jnp.where(
                hit, vec.at[3].set(0.0).at[4].set(0.0).at[5].set(0.0), vec
            )
            return (vec, collided), None

        vec0 = jnp.stack(
            [
                env_state.agent_pos[0],
                env_state.agent_pos[1],
                env_state.heading,
                env_state.vel_body[0],
                env_state.vel_body[1],
                env_state.yaw_rate,
            ]
        ).astype(jnp.float32)
        (vec, collided), _ = jax.lax.scan(
            substep, (vec0, env_state.collided), xs=None,
            length=env_params.action_repeat,
        )

        new_pos = vec[0:2]
        new_heading = vec[2]
        new_vel = vec[3:5]
        new_omega = vec[5]
        new_time = env_state.time + 1

        # replan the path every `fps` steps (as in BaseEnv)
        pred = jnp.mod(new_time, env_params.fps)
        path_array = jax.lax.cond(
            pred,
            lambda _: BaseEnv._find_path(
                new_pos, env_state.goal_pos, obstacles, env_params
            ),
            lambda _: env_state.path_array,
            None,
        )

        dist_to_goal = jnp.linalg.norm(new_pos - env_state.goal_pos)

        new_state = RoboTaxiState(
            time=new_time,
            goal_pos=env_state.goal_pos,
            agent_pos=new_pos,
            agent_forward_dir=jnp.stack(
                [jnp.cos(new_heading), jnp.sin(new_heading)]
            ),
            static_obstacles=env_state.static_obstacles,
            kinematic_obstacles=kinematic_obstacles,
            kinematic_obst_velocities=env_state.kinematic_obst_velocities,
            path_array=path_array,
            heading=new_heading,
            vel_body=new_vel,
            yaw_rate=new_omega,
            collided=collided,
            prev_dist_to_goal=dist_to_goal,
        )

        obs = RoboTaxiEnv.get_observation(new_state, env_params)

        # termination is evaluated on the *new* state (gymnax-style)
        goal_done = BaseEnv._check_goal(new_pos, env_state.goal_pos)
        collision_done = collided
        time_done = new_time >= env_params.max_steps_in_episode
        done = jnp.logical_or(goal_done, jnp.logical_or(collision_done, time_done))

        reward = RoboTaxiEnv._reward(
            env_state, new_state, obs, accel, steer, dist_to_goal,
            goal_done, collision_done, env_params,
        )

        info = {
            "time": new_time,
            "distance_to_path": obs.distance_to_path,
            "direction_of_path": obs.direction_of_path,
            "distance_to_goal": dist_to_goal,
            "speed": obs.speed,
            "success": goal_done,
            "collision": collision_done,
        }
        return obs, new_state, reward, done, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(
        env_state: RoboTaxiState, env_params: RoboTaxiParams
    ) -> RoboTaxiObservation:
        pos = env_state.agent_pos
        path = env_state.path_array
        n = path.shape[0]

        # --- path hints: closest waypoint + local path tangent ---
        dists = jnp.linalg.norm(path - pos, axis=1)
        closest = jnp.minimum(jnp.argmin(dists), n - 2)
        anchor = path[closest]
        path_vec = path[closest + 1] - anchor
        path_dir = path_vec / (jnp.linalg.norm(path_vec) + 1e-8)

        # direction to the goal (world frame); also a fallback for a
        # degenerate path tangent (e.g. duplicated waypoints at the goal)
        to_goal = env_state.goal_pos - pos
        goal_dir = to_goal / (jnp.linalg.norm(to_goal) + 1e-8)
        path_dir = jnp.where(
            jnp.linalg.norm(path_vec) > 1e-6, path_dir, goal_dir
        )

        rel = pos - anchor
        proj = anchor + jnp.dot(rel, path_dir) * path_dir
        distance_to_path = jnp.linalg.norm(pos - proj)

        # --- perception: rays + velocity of the obstacle each ray hits ---
        obstacles = jnp.concatenate(
            [env_state.static_obstacles, env_state.kinematic_obstacles], axis=0
        )
        obstacle_velocities = jnp.concatenate(
            [jnp.zeros_like(env_state.static_obstacles),
             env_state.kinematic_obst_velocities],
            axis=0,
        )
        rays, hit_vel = RoboTaxiEnv._perceive(
            pos, env_state.agent_forward_dir, obstacles, obstacle_velocities,
            env_params,
        )

        # --- ego state ---
        speed = jnp.linalg.norm(env_state.vel_body)
        c, s = jnp.cos(env_state.heading), jnp.sin(env_state.heading)
        goal_dir_body = jnp.stack(
            [c * goal_dir[0] + s * goal_dir[1], -s * goal_dir[0] + c * goal_dir[1]]
        )
        fwd = env_state.agent_forward_dir
        heading_error = jnp.arctan2(
            fwd[0] * path_dir[1] - fwd[1] * path_dir[0], jnp.dot(fwd, path_dir)
        )

        return RoboTaxiObservation(
            distance_to_path=distance_to_path,
            direction_of_path=path_dir,
            collision_rays=rays,
            speed=speed,
            vel_body=env_state.vel_body,
            yaw_rate=env_state.yaw_rate,
            distance_to_goal=jnp.linalg.norm(to_goal),
            direction_to_goal_body=goal_dir_body,
            heading_error=heading_error,
            ray_hit_velocities=hit_vel,
        )

    def _perceive(
        agent_pos, forward_dir, obstacles, obstacle_velocities, env_params
    ):
        """Ray casting against all obstacles; also returns, for every ray,
        the velocity of the obstacle it hit (zeros if no hit)."""
        rays = BaseEnv._generate_rays(
            forward_dir, env_params.num_ray_sensors, env_params.agent_FOV
        )
        t_min, hit_idx = RoboTaxiEnv._ray_circle_intersections_idx(
            agent_pos, rays, obstacles, RADIUS
        )
        hit = jnp.isfinite(t_min)
        dist = jnp.minimum(t_min, env_params.perception_radius)
        ray_vectors = rays * dist[:, None]
        hit_vel = jnp.where(
            hit[:, None], obstacle_velocities[hit_idx],
            jnp.zeros_like(obstacle_velocities[0]),
        )
        return ray_vectors, hit_vel

    @partial(jax.jit, static_argnames=("circle_radius",))
    def _ray_circle_intersections_idx(
        origins, dirs, centers, circle_radius=RADIUS
    ):
        """Same as BaseEnv._ray_circle_intersections, but also returns the
        index of the closest hit obstacle for every ray."""
        if origins.ndim == 1:
            origins = jnp.broadcast_to(origins, dirs.shape)
        o = origins[:, None, :]  # (N, 1, 2)
        d = dirs[:, None, :]     # (N, 1, 2)
        c = centers[None, :, :]  # (1, M, 2)

        m = o - c
        b = jnp.sum(m * d, axis=-1)
        c_val = jnp.sum(m * m, axis=-1) - circle_radius**2
        disc = b**2 - c_val

        valid = disc >= 0.0
        sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
        t1 = -b - sqrt_disc
        t2 = -b + sqrt_disc
        t_candidates = jnp.where(t1 > 0, t1, jnp.where(t2 > 0, t2, jnp.inf))
        t_candidates = jnp.where(valid, t_candidates, jnp.inf)

        idx = jnp.argmin(t_candidates, axis=1)
        t_min = jnp.min(t_candidates, axis=1)
        return t_min, idx

    # ------------------------------------------------------------------
    # Vehicle dynamics (dynamic bicycle model)
    # ------------------------------------------------------------------

    def _prepare_action(action, env_params) -> Tuple[chex.Scalar, chex.Scalar]:
        """Map the raw action to (acceleration, steering angle)."""
        action = jnp.asarray(action)
        if env_params.normalize_actions:
            accel = jnp.clip(action[0], -1.0, 1.0) * env_params.max_accel
            steer = jnp.clip(action[1], -1.0, 1.0) * env_params.max_steer
        else:
            accel = jnp.clip(
                action[0], -env_params.max_accel, env_params.max_accel
            )
            steer = jnp.clip(
                action[1], -env_params.max_steer, env_params.max_steer
            )
        return accel, steer

    def _tire_forces(vx, vy, omega, steer, env_params):
        """Lateral tire forces: linear tire model + saturation.

        The denominator of the slip angles is clamped from below, so at
        (near) standstill and in reverse the slip saturates instead of
        dividing by zero. A fade factor smoothly switches the lateral
        forces off at very low speed.
        """
        p = env_params
        vx_safe = jnp.maximum(vx, 1e-3)
        alpha_f = jnp.arctan((vy + p.lf * omega) / vx_safe) - steer
        alpha_r = jnp.arctan((vy - p.lr * omega) / vx_safe)
        fade = jnp.abs(vx) / (jnp.abs(vx) + p.v_fade)

        load_f = p.mass * p.gravity * p.lr / (p.lf + p.lr)
        load_r = p.mass * p.gravity * p.lf / (p.lf + p.lr)
        fyf = jnp.clip(-p.cf * alpha_f * fade, -p.mu * load_f, p.mu * load_f)
        fyr = jnp.clip(-p.cr * alpha_r * fade, -p.mu * load_r, p.mu * load_r)
        return fyf, fyr

    def _deriv(vec, controls, env_params) -> chex.Array:
        """Time derivative of the vehicle state (dynamic bicycle model)."""
        p = env_params
        theta, vx, vy, omega = vec[2], vec[3], vec[4], vec[5]
        accel, steer = controls[0], controls[1]

        fyf, fyr = RoboTaxiEnv._tire_forces(vx, vy, omega, steer, p)

        # longitudinal force on the rear axle: command - drag - rolling
        fx = (
            p.mass * accel
            - p.drag_coef * vx * jnp.abs(vx)
            - p.roll_coef * jnp.tanh(3.0 * vx)
        )

        # friction ellipse on the rear axle: combined (fx, fyr) is capped
        load_r = p.mass * p.gravity * p.lf / (p.lf + p.lr)
        f_max = p.mu * load_r
        scale = f_max / jnp.maximum(jnp.sqrt(fx**2 + fyr**2), f_max)
        fx = fx * scale
        fyr = fyr * scale

        cd, sd = jnp.cos(steer), jnp.sin(steer)

        dx = vx * jnp.cos(theta) - vy * jnp.sin(theta)
        dy = vx * jnp.sin(theta) + vy * jnp.cos(theta)
        dtheta = omega
        # body-frame equations of motion
        dvx = fx / p.mass - fyf * sd / p.mass + omega * vy
        dvy = fyf * cd / p.mass + fyr / p.mass - omega * vx
        domega = (p.lf * fyf * cd - p.lr * fyr) / p.iz
        return jnp.stack([dx, dy, dtheta, dvx, dvy, domega])

    def _integrate(vec, accel, steer, env_params) -> chex.Array:
        """One RK4 step of dt = step_size, plus a hard speed cap."""
        p = env_params
        dt = p.step_size
        u = (accel, steer)
        k1 = RoboTaxiEnv._deriv(vec, u, p)
        k2 = RoboTaxiEnv._deriv(vec + 0.5 * dt * k1, u, p)
        k3 = RoboTaxiEnv._deriv(vec + 0.5 * dt * k2, u, p)
        k4 = RoboTaxiEnv._deriv(vec + dt * k3, u, p)
        out = vec + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # speed limiter (also active during slides)
        v = out[3:5]
        speed = jnp.linalg.norm(v)
        v = jnp.where(speed > p.v_max, v * (p.v_max / (speed + 1e-8)), v)
        return out.at[3:5].set(v)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _reward(
        old_state, new_state, obs, accel, steer, dist_to_goal,
        goal_done, collision_done, env_params,
    ):
        """Dense reward (see the report for the motivation of each term)."""
        p = env_params
        # potential-based progress towards the goal
        r_progress = p.w_progress * (old_state.prev_dist_to_goal - dist_to_goal)
        # stay close to the planned path (deadband compensates the grid
        # quantization of the DP path: it is built from cell centers)
        r_path = -p.w_path * jnp.clip(
            jnp.minimum(obs.distance_to_path, p.path_dist_clip) - p.path_freeband,
            0.0, None,
        )
        # move along the path (not sideways / backwards)
        r_align = (
            p.w_align
            * jnp.dot(new_state.vel_body, obs.direction_of_path)
            / p.v_max
        )
        # cost of time and control effort
        r_time = -p.w_time
        r_action = -p.w_action * (
            (accel / p.max_accel) ** 2 + (steer / p.max_steer) ** 2
        )
        # proximity penalty: distance to the closest obstacle seen by rays.
        # Only active while the vehicle moves: a parked taxi touching a curb
        # is safe, a moving one is not.
        d_min = jnp.min(jnp.linalg.norm(obs.collision_rays, axis=1))
        speed = jnp.linalg.norm(new_state.vel_body)
        r_safety = (
            -p.w_safety
            * jnp.minimum(speed / 0.3, 1.0)
            * jnp.maximum(0.0, 1.0 - d_min / p.safe_dist)
        )
        # terminal rewards
        r_terminal = jnp.where(
            goal_done, p.r_goal, jnp.where(collision_done, p.r_crash, 0.0)
        )
        return (
            r_progress + r_path + r_align + r_time + r_action + r_safety
            + r_terminal
        )

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnames=("env_params",))
    def _move_kinematic_obstacles(env_state, env_params):
        """Same as BaseEnv, but scaled with action_repeat (wall-clock time)."""
        positions = env_state.kinematic_obstacles
        velocities = env_state.kinematic_obst_velocities
        dt = env_params.step_size * env_params.action_repeat
        new_positions = positions + dt * velocities
        new_positions = jax.vmap(lambda x, y: jnp.mod(x, y), in_axes=(0, None))(
            new_positions,
            jnp.flip(jnp.asarray(env_params.map_height_width, dtype=jnp.int32)),
        )
        return new_positions

    @staticmethod
    def action_dim() -> int:
        """Dimensionality of the action vector: (acceleration, steering)."""
        return 2

    @staticmethod
    def observation_dim(env_params) -> int:
        """Dimensionality of ``RoboTaxiObservation.as_vector()``."""
        return 11 + 4 * env_params.num_ray_sensors
