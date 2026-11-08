"""A simple scripted driver for the robotaxi environment.

Not a learned policy -- just a hand-crafted controller used for demos and
sanity checks of the environment:

* **Path following**: classic pure pursuit towards a lookahead waypoint of
  the DP path that the environment provides in the observation.
* **Obstacle avoidance**: reactive steering away from near ray hits. Static
  obstacles (walls) only get a gentle repulsion (the DP path already routes
  around them), while moving obstacles get a strong one -- their velocity is
  taken from ``obs.ray_hit_velocities``.
* **Speed control**: the target speed is limited by the upcoming path
  curvature, by the free space straight ahead and by *approaching* moving
  obstacles (approach rate is estimated from the hit velocity of each ray).
* **Unstuck maneuver**: if the car has not moved for a while, it backs up
  for a moment (demo-only heuristic).

Ray distances are floored because a ray that grazes a circle the agent is
touching can report a distance of ~0 (see the report).
"""

import numpy as np

from jax import numpy as jnp


class ScriptedDriver:
    """Pure-pursuit path follower with reactive obstacle avoidance."""

    STUCK_WINDOW = 100   # steps to decide the car is stuck
    STUCK_DIST = 0.25    # ...if it moved less than this
    UNSTUCK_STEPS = 45   # reverse duration

    def __init__(
        self,
        env_params,
        v_nominal: float = 2.2,    # nominal cruise speed [cells / s]
        lookahead: float = 3.5,    # pure-pursuit lookahead distance [cells]
        k_accel: float = 2.5,      # speed-controller gain
        # static walls: gentle repulsion, small radius
        r_static: float = 1.7,
        k_static: float = 0.35,
        # moving obstacles: strong repulsion, larger radius
        r_dynamic: float = 2.6,
        k_dynamic: float = 1.1,
        a_lat_max: float = 3.0,    # comfortable lateral acceleration limit
        d_floor: float = 0.3,      # floor for grazing ray distances
    ):
        self.p = env_params
        self.v_nom = float(min(v_nominal, env_params.v_max))
        self.lookahead = lookahead
        self.k_accel = k_accel
        self.r_static = r_static
        self.k_static = k_static
        self.r_dynamic = r_dynamic
        self.k_dynamic = k_dynamic
        self.a_lat_max = a_lat_max
        self.d_floor = d_floor

        # ray angles, must match BaseEnv._generate_rays
        n = env_params.num_ray_sensors
        fov = float(env_params.agent_FOV)
        self.ray_angles = np.linspace(-fov, fov, n, endpoint=False)

        # unstuck bookkeeping
        self._hist = []        # recent positions
        self._unstuck_left = 0
        self._unstuck_steer = 0.0

    # ------------------------------------------------------------------

    def act(self, obs, state) -> np.ndarray:
        """Return the (acceleration, steering) action, normalized to [-1, 1]."""
        p = self.p
        pos = np.asarray(state.agent_pos, dtype=float)
        heading = float(state.heading)
        vel = np.asarray(state.vel_body, dtype=float)
        vx = vel[0]
        path = np.asarray(state.path_array, dtype=float)
        ang = self.ray_angles

        # ------------------------- unstuck heuristic -------------------------
        self._hist.append(pos.copy())
        if len(self._hist) > self.STUCK_WINDOW:
            self._hist.pop(0)
        if self._unstuck_left == 0 and len(self._hist) == self.STUCK_WINDOW:
            if np.linalg.norm(pos - self._hist[0]) < self.STUCK_DIST:
                self._unstuck_left = self.UNSTUCK_STEPS
                # reverse while steering away from the closer side
                d = np.maximum(
                    np.linalg.norm(np.asarray(obs.collision_rays), axis=1),
                    self.d_floor,
                )
                left = (ang > np.deg2rad(30)) & (ang < np.deg2rad(120))
                right = (ang < -np.deg2rad(30)) & (ang > -np.deg2rad(120))
                d_left = d[left].min() if left.any() else 1e9
                d_right = d[right].min() if right.any() else 1e9
                # steer towards the blocked side while reversing (nose swings
                # away from the free side)
                self._unstuck_steer = -0.7 if d_left >= d_right else 0.7
        if self._unstuck_left > 0:
            self._unstuck_left -= 1
            return np.array([-0.55, self._unstuck_steer], dtype=np.float32)

        # ------------------------- goal docking -------------------------
        # Near the goal the DP path degenerates (duplicated waypoints) and the
        # goal circle usually touches a wall, so wall repulsion would keep the
        # car out. Switch to a simple docking maneuver instead.
        goal = np.asarray(state.goal_pos, dtype=float)
        d_goal = float(np.linalg.norm(goal - pos))
        docking = d_goal < 2.5

        # ------------------------- path following -------------------------
        dists = np.linalg.norm(path - pos, axis=1)
        i0 = int(np.argmin(dists))
        ahead = np.where(dists >= self.lookahead)[0]  # far-enough waypoints
        ahead = ahead[ahead >= i0]
        j = int(ahead[0]) if ahead.size else len(path) - 1
        target = path[j]

        to_target = target - pos
        ld = max(float(np.linalg.norm(to_target)), 0.6)
        alpha = np.arctan2(to_target[1], to_target[0]) - heading
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        wheelbase = p.lf + p.lr
        delta_pp = np.arctan2(2.0 * wheelbase * np.sin(alpha), ld)

        # ------------------------- obstacle avoidance -------------------------
        rays = np.asarray(obs.collision_rays, dtype=float)        # (M, 2)
        hit_vel = np.asarray(obs.ray_hit_velocities, dtype=float)  # (M, 2)
        ray_len = np.linalg.norm(rays, axis=1)
        d = np.maximum(ray_len, self.d_floor)
        moving = np.linalg.norm(hit_vel, axis=1) > 0.1
        side = np.abs(ang) < np.deg2rad(100)

        # agent velocity in the world frame
        c, s = np.cos(heading), np.sin(heading)
        v_world = np.array([c * vel[0] - s * vel[1], s * vel[0] + c * vel[1]])

        # per-ray approach rate: is the hit obstacle closing in?
        ray_dir = rays / np.maximum(ray_len, self.d_floor)[:, None]
        approach = np.sum((hit_vel - v_world[None, :]) * ray_dir, axis=1)
        closing = moving & (approach < -0.05)

        w_static = np.where(
            side, np.clip(1.0 - d / self.r_static, 0.0, 1.0) ** 2, 0.0
        )
        if docking:
            w_static = np.zeros_like(w_static)
        w_dynamic = np.where(
            side & moving,
            np.clip(1.0 - d / self.r_dynamic, 0.0, 1.0) ** 2
            * np.where(closing, 1.0, 0.45),
            0.0,
        )
        # steer away from the weighted lateral centroid of near hits
        delta_av = -(
            self.k_static * np.sum(w_static * np.sin(ang))
            + self.k_dynamic * np.sum(w_dynamic * np.sin(ang))
        ) / (np.sum(w_static) + np.sum(w_dynamic) + 0.5)

        # ------------------------- speed control -------------------------
        # free space straight ahead (narrow cone; walls we drive along must
        # not freeze the car -- hence the speed floor below)
        front12 = np.abs(ang) < np.deg2rad(12)
        d_ahead = float(d[front12].min())
        v_obs = 2.0 * max(d_ahead - 0.7, 0.0)
        # approaching moving obstacles in the front sector
        front35_closing = (np.abs(ang) < np.deg2rad(35)) & closing
        if front35_closing.any():
            d_dyn = float(d[front35_closing].min())
            v_obs = min(v_obs, 2.0 * max(d_dyn - 1.0, 0.0))
        # hard override: narrow front really blocked -> steer to the freer side
        if d_ahead < 0.9:
            left = (ang > np.deg2rad(20)) & (ang < np.deg2rad(90))
            right = (ang < -np.deg2rad(20)) & (ang > -np.deg2rad(90))
            d_left = float(d[left].min()) if left.any() else 1e9
            d_right = float(d[right].min()) if right.any() else 1e9
            delta_av += 0.5 * (1.0 if d_left >= d_right else -1.0)

        kappa = abs(np.tan(delta_pp)) / wheelbase + 1e-6
        v_curve = np.sqrt(self.a_lat_max / kappa)
        v_target = min(self.v_nom, v_curve, v_obs, p.v_max)
        if docking:
            # steer straight at the goal and crawl in; only moving obstacles
            # and a really blocked front still matter
            alpha_g = np.arctan2(goal[1] - pos[1], goal[0] - pos[0]) - heading
            alpha_g = (alpha_g + np.pi) % (2 * np.pi) - np.pi
            delta_dock = float(np.clip(1.5 * alpha_g, -p.max_steer, p.max_steer))
            delta = delta_dock + delta_av
            v_target = min(1.2, 0.6 * d_goal + 0.25, v_obs)
        else:
            delta = delta_pp + delta_av
            # never fully freeze because of wall-grazing rays (unless very close)
            v_target = max(v_target, 0.55 if d_ahead > 0.5 else 0.0)
        accel = float(np.clip(self.k_accel * (v_target - vx), -p.max_accel, p.max_accel))

        delta = float(np.clip(delta, -p.max_steer, p.max_steer))
        action = np.array(
            [accel / p.max_accel, delta / p.max_steer], dtype=np.float32
        )
        return np.clip(action, -1.0, 1.0)
