"""Record demo videos / run headless evaluation of the robotaxi environment.

Examples (run from the repository ``src/`` directory):

    python demo_robotaxi.py --map 0                     # video of map 0
    python demo_robotaxi.py --map 1 --out ../videos/robotaxi_map1.mp4
    python demo_robotaxi.py --all                       # eval all maps, no video

The car is driven by the hand-crafted controller from
``utils/scripted_driver.py`` (pure pursuit + reactive avoidance) -- this
demo shows the *environment*, not a learned policy.
"""

import argparse
import os

import jax
import numpy as np
from jax import numpy as jnp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle

from env.robotaxi import RoboTaxiEnv
from utils.scripted_driver import ScriptedDriver

KEY = jax.random.PRNGKey(0)

# colors
C_STATIC = "#5b5f66"
C_DYNAMIC = "#d64545"
C_GOAL = "#3fa34d"
C_PATH = "#3b6fd4"
C_RAY = (1.0, 0.35, 0.35, 0.30)
C_AGENT = "#2d7dd2"


def run_episode(map_id, max_steps=1200, video_path=None, stills_dir=None,
                render_every=2):
    """Run one episode; optionally record an mp4 and save still frames."""
    p, s0 = RoboTaxiEnv.init_params(
        key=KEY, map_id=map_id, max_steps=max_steps, path_length=100,
        perception_radius=5.0, num_ray_sensors=32,
    )
    driver = ScriptedDriver(p)
    obs, state = RoboTaxiEnv.reset(KEY, p, s0)

    frames = []  # (state, obs, reward) per step
    ret, done = 0.0, False
    while not done:
        action = jnp.asarray(driver.act(obs, state))
        obs, state, reward, done, info = RoboTaxiEnv.step(KEY, state, action, p)
        ret += float(reward)
        frames.append((state, obs, float(reward)))

    stats = dict(
        map=map_id, steps=len(frames), success=bool(info["success"]),
        crash=bool(info["collision"]), ret=ret,
        d_goal=float(info["distance_to_goal"]),
    )

    if video_path or stills_dir:
        os.makedirs(os.path.dirname(os.path.abspath(video_path or stills_dir)),
                    exist_ok=True)
        if video_path:
            _record_video(p, frames, video_path, render_every)
        if stills_dir:
            _save_stills(p, frames, stills_dir)
    return stats


# ---------------------------------------------------------------------------


def _draw_frame(ax, p, state, obs, ret=0.0):
    ax.clear()
    h, w = p.map_height_width
    ax.set_xlim(-1.4, w + 1.4)
    ax.set_ylim(-1.4, h + 1.4)
    ax.set_aspect("equal")
    ax.axis("off")

    pos = np.asarray(state.agent_pos, dtype=float)
    goal = np.asarray(state.goal_pos, dtype=float)
    path = np.asarray(state.path_array, dtype=float)

    # planned path
    ax.plot(path[:, 0], path[:, 1], "-", color=C_PATH, lw=2, alpha=0.75,
            zorder=1)

    # static obstacles
    for o in np.asarray(state.static_obstacles):
        ax.add_patch(Circle(o, 1.0, facecolor=C_STATIC, edgecolor="none",
                            zorder=2))
    # moving obstacles
    for o in np.asarray(state.kinematic_obstacles):
        ax.add_patch(Circle(o, 1.0, facecolor=C_DYNAMIC, edgecolor="none",
                            zorder=2))

    # goal
    ax.add_patch(Circle(goal, 1.0, facecolor=C_GOAL, alpha=0.55, zorder=2))
    ax.add_patch(Circle(goal, 0.25, facecolor=C_GOAL, zorder=3))

    # perception rays
    rays = np.asarray(obs.collision_rays, dtype=float)
    for r in rays:
        ax.plot([pos[0], pos[0] + r[0]], [pos[1], pos[1] + r[1]],
                color=C_RAY, lw=1.0, zorder=3)

    # agent + heading + velocity
    ax.add_patch(Circle(pos, 0.45, facecolor=C_AGENT,
                        edgecolor="white", lw=1.2, zorder=4))
    fwd = np.asarray(state.agent_forward_dir, dtype=float)
    ax.annotate("", xy=tuple(pos + 1.15 * fwd), xytext=tuple(pos),
                arrowprops=dict(arrowstyle="-|>", color="#f2b705", lw=2),
                zorder=5)
    vel_world = np.asarray(state.vel_body, dtype=float)
    if np.linalg.norm(vel_world) > 0.05:
        c, s = np.cos(float(state.heading)), np.sin(float(state.heading))
        vw = np.array([c * vel_world[0] - s * vel_world[1],
                       s * vel_world[0] + c * vel_world[1]])
        ax.annotate("", xy=tuple(pos + 0.7 * vw), xytext=tuple(pos),
                    arrowprops=dict(arrowstyle="-|>", color="#2e8b57", lw=1.6),
                    zorder=5)

    ax.set_title(
        f"t = {float(state.time) / p.fps:5.1f} с    "
        f"v = {float(obs.speed):4.2f}    R = {ret:+7.1f}",
        fontsize=11,
    )


def _record_video(p, frames, path, render_every):
    fig = plt.figure(figsize=(12.0, 6.4), dpi=100)
    ax = fig.add_subplot(111)
    writer = FFMpegWriter(
        fps=30,
        metadata={"title": "robotaxi demo", "artist": "IAEML Assignment 1"},
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23"],
    )
    ret = 0.0
    with writer.saving(fig, path, dpi=100):
        for i, (state, obs, reward) in enumerate(frames):
            if i % render_every != 0:
                continue
            ret += reward * render_every  # approximate skipped frames
            _draw_frame(ax, p, state, obs, ret)
            writer.grab_frame()
    plt.close(fig)
    print(f"video written: {path}")


def _save_stills(p, frames, stills_dir):
    os.makedirs(stills_dir, exist_ok=True)
    # pick: first frame, closest encounter while moving, last frame
    dists = [
        float(jax.numpy.min(jax.numpy.linalg.norm(obs.collision_rays, axis=1)))
        if float(obs.speed) > 0.5 else 1e9
        for _, obs, _ in frames
    ]
    idx_enc = int(np.argmin(dists))
    picks = {"start": 0, "encounter": idx_enc, "final": len(frames) - 1}
    for name, idx in picks.items():
        fig = plt.figure(figsize=(12.0, 6.4), dpi=160)
        ax = fig.add_subplot(111)
        _draw_frame(ax, p, frames[idx][0], frames[idx][1], ret=0.0)
        fig.savefig(os.path.join(stills_dir, f"still_{name}.png"),
                    bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    print(f"stills written: {stills_dir} ({list(picks)})")


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=int, default=0)
    parser.add_argument("--out", type=str, default=None,
                        help="output mp4 path (default ../videos/robotaxi_mapN.mp4)")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--all", action="store_true",
                        help="evaluate maps 0..2 without recording")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--stills-dir", type=str, default=None)
    args = parser.parse_args()

    if args.all:
        rows = []
        for m in (0, 1, 2):
            rows.append(run_episode(m, max_steps=args.max_steps))
        print(f"{'map':>4} {'steps':>6} {'success':>8} {'crash':>6} {'return':>8}")
        for r in rows:
            print(f"{r['map']:>4} {r['steps']:>6} {str(r['success']):>8} "
                  f"{str(r['crash']):>6} {r['ret']:>8.1f}")
    else:
        out = args.out or os.path.join("..", "videos", f"robotaxi_map{args.map}.mp4")
        stats = run_episode(
            args.map, max_steps=args.max_steps,
            video_path=None if args.no_video else out,
            stills_dir=args.stills_dir,
        )
        print(stats)
