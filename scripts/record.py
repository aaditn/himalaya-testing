"""Render a trained policy to MP4 on the pod.

    python scripts/record.py runs/ice_120000/policy --out videos/ice.mp4
    python scripts/record.py runs/ice_120000/policy --friction 0.08 --seconds 12

Renders offscreen on the GPU -- no display needed -- so it works while
training is running. Pull the result with scripts/pull_videos.sh.
"""
import argparse
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", help="path to a saved brax params file")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--friction", type=float, default=0.1,
                    help="floor friction to render at (0.1 = ice, 0.8 = dry)")
    ap.add_argument("--vx", type=float, default=0.8, help="commanded forward vel")
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0, help="commanded yaw rate")
    ap.add_argument("--rough", action="store_true")
    ap.add_argument("--camera", default="side",
                    help="'side'/'front'/'chase' = fixed world-up cameras; "
                         "'track' = Playground's body-mounted camera, which "
                         "ROLLS WITH THE TORSO and makes an upright robot look "
                         "upside down (mode=2, mjCAMLIGHT_TRACKCOM)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=640)
    args = ap.parse_args()

    import jax
    import jax.numpy as jp
    import mediapy
    import mujoco
    import numpy as np
    from brax.io import model as brax_model
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import registry

    from himalaya.ice import ice_env

    task = "G1JoystickRoughTerrain" if args.rough else "G1JoystickFlatTerrain"
    cfg = ice_env.ice_config(registry.get_default_config(task))
    # Must match training: the policy was trained with strict termination
    # (height + tilt). Rendering under Playground's stock termination changes
    # where episodes end and inflates the fall count.
    env = ice_env.patch_termination(registry.load(task, config=cfg))

    # Rebuild the same network shape the trainer used, then load the weights.
    net = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
    )
    params = brax_model.load_params(args.policy)
    inference = ppo_networks.make_inference_fn(net)(params, deterministic=True)
    inference = jax.jit(inference)

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    state = reset(jax.random.PRNGKey(0))
    # hold a fixed velocity command so the clip shows deliberate walking
    state.info["command"] = jp.array([args.vx, args.vy, args.wz])

    n = int(args.seconds / env.dt)
    rollout, slips = [], 0
    rng = jax.random.PRNGKey(1)
    for _ in range(n):
        rng, key = jax.random.split(rng)
        act, _ = inference(state.obs, key)
        state = step(state, act)
        state.info["command"] = jp.array([args.vx, args.vy, args.wz])
        rollout.append(state)
        if float(state.done):
            state = reset(key)
            state.info["command"] = jp.array([args.vx, args.vy, args.wz])
            slips += 1

    # Set the floor friction we want to SHOW (rendering only).
    mj_model = env.mj_model
    mj_model.pair_friction[0:2, 0:2] = args.friction

    if args.camera in ("side", "front", "chase"):
        # Free camera with world up-axis. The only camera in the model is
        # "track", which is body-mounted: it inherits the torso's orientation,
        # so a rolling robot produces a rolling image. The robot was upright
        # in every physics check (min gravity_z +0.63); the picture was lying.
        import mujoco as _mj
        cam = _mj.MjvCamera()
        cam.type = _mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, 0.0, 0.7]
        cam.distance = {"side": 3.5, "front": 3.0, "chase": 4.0}[args.camera]
        cam.azimuth = {"side": 90.0, "front": 180.0, "chase": 135.0}[args.camera]
        cam.elevation = -12.0
        frames = _render_free(env, rollout, cam, args.width, args.height)
    else:
        frames = env.render(
            rollout, camera=args.camera, width=args.width, height=args.height
        )
    out = Path(args.out or f"videos/{Path(args.policy).parent.name}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(out), frames, fps=1.0 / env.dt)

    print(f"wrote {out}  ({len(frames)} frames, {args.seconds}s)")
    print(f"  friction={args.friction}  command=({args.vx},{args.vy},{args.wz})")
    print(f"  falls during clip: {slips}")


def _render_free(env, rollout, cam, width, height):
    """Render with an explicit free camera that keeps world-up."""
    import mujoco
    import numpy as np

    m = env.mj_model
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=height, width=width)
    frames = []
    for st in rollout:
        d.qpos[:] = np.array(st.data.qpos)
        d.qvel[:] = np.array(st.data.qvel)
        mujoco.mj_forward(m, d)
        # follow the robot in x/y but never tilt with it
        cam.lookat[0] = float(d.qpos[0])
        cam.lookat[1] = float(d.qpos[1])
        renderer.update_scene(d, camera=cam)
        frames.append(renderer.render())
    renderer.close()
    return frames


if __name__ == "__main__":
    main()
