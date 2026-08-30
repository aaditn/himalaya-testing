"""Render a trained policy to MP4 on the pod.

    python scripts/record.py runs/g1_120000/policy --out videos/walk.mp4
    python scripts/record.py runs/g1_120000/policy --seconds 12

Renders offscreen on the GPU -- no display needed -- so it works while
training is running. Pull the result with scripts/pod/pull.sh.
"""
import argparse
import functools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The pod is headless: without this, mujoco.Renderer raises "an OpenGL platform
# library has not been loaded into this process" -- but only AFTER the rollout
# has finished, so a full simulation is thrown away at the last step. Set here
# rather than relied on from the shell, which is how it got lost. Must precede
# the mujoco import.
os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", help="path to a saved brax params file")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--friction", type=float, default=0.8,
                    help="floor friction to render at")
    ap.add_argument("--vx", type=float, default=0.8, help="commanded forward vel")
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0, help="commanded yaw rate")
    ap.add_argument("--rough", action="store_true")
    ap.add_argument("--slope-deg", type=float, default=None,
                    help="tilt the floor by N degrees (uses the slope_terrain "
                         "scene, which is the rough heightfield on a tilt)")
    ap.add_argument("--climb", type=float, default=None, metavar="DEG",
                    help="render the CLIMBING task at DEG degrees. Mirrors "
                         "train.py --climb exactly: mountain_terrain scene and "
                         "climb_config. Using --slope-deg for a climb policy "
                         "renders the wrong scene (slope_terrain) with the "
                         "wrong 24-term reward config, so the clip would not "
                         "show what was trained.")
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
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from himalaya.env import scene as sc
    from himalaya.env import Joystick, climb_config, default_config

    # Same env as training -- himalaya/env/ is the single definition, so a
    # reward or termination change cannot drift between train and record.
    NJMAX, NACONMAX = sc.NJMAX, sc.NACONMAX
    # Names the vendored env uses directly; Playground's registry was
    # translating its public "G1Joystick*Terrain" ids onto these.
    if args.climb is not None:
        # Must match train.py --climb: same scene, same reward config. See the
        # --climb help text for why the slope_terrain path is not a substitute.
        task = "mountain_terrain"
        cfg = climb_config(args.climb)
    elif args.slope_deg is not None:
        task = "slope_terrain"
        cfg = default_config()
    else:
        task = "rough_terrain" if args.rough else "flat_terrain"
        cfg = default_config()
    cfg.njmax = NJMAX
    cfg.naconmax = NACONMAX
    # base.py reads this at load time and writes the floor geom's quat, so the
    # tilt lands on the model that is actually stepped AND the one that is
    # rendered -- both come from the same MjModel here.
    if args.slope_deg is not None:
        cfg.slope_deg = args.slope_deg
    env = Joystick(task=task, config=cfg)

    # Rebuild the same network shape the trainer used, then load the weights.
    # preprocess_observations_fn is NOT optional. brax defaults it to identity,
    # but training ran with normalize_observations=True, so the policy learned
    # on normalized observations. Omitting it here loads the normalizer params
    # (params[0], a RunningStatisticsState) and then ignores them, feeding the
    # policy raw inputs. It fails silently: no error, just a policy that looks
    # bad. Measured on the same checkpoint: mean episode length 70 without
    # this line, 905 with it.
    net = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        policy_hidden_layer_sizes=sc.HIDDEN_LAYERS,
        value_hidden_layer_sizes=sc.HIDDEN_LAYERS,
        policy_obs_key="state",
        value_obs_key="privileged_state",
        preprocess_observations_fn=running_statistics.normalize,
    )
    params = brax_model.load_params(args.policy)
    inference = ppo_networks.make_inference_fn(net)(params, deterministic=True)
    inference = jax.jit(inference)

    # Friction must be set on the model that gets STEPPED, before the rollout.
    # Setting it on env.mj_model afterwards only changes what is drawn, so the
    # clip would show a "slippery" floor the physics never used.
    # [0:4], not [0:2]: the model has FOUR floor pairs -- left/right foot then
    # left/right hand (verified with mj_id2name; MuJoCo reorders <pair> elements,
    # so re-check after any XML pair change). Pairs 4-6 are self-collision
    # (hand-thigh, foot-foot) and must keep their own friction, so do not widen
    # this slice further. With [0:2] the hands rendered at the XML default while
    # the feet used the swept value.
    env._mjx_model = env._mjx_model.tree_replace(
        {"pair_friction": env._mjx_model.pair_friction.at[sc.FLOOR_PAIRS, 0:2].set(args.friction)}
    )

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    state = reset(jax.random.PRNGKey(0))
    # hold a fixed velocity command so the clip shows deliberate walking
    state.info["command"] = jp.array([args.vx, args.vy, args.wz])

    n = int(args.seconds / env.dt)
    rollout, slips = [], 0
    rng = jax.random.PRNGKey(1)

    # Climb diagnostics. Whether the hands are actually being used, and whether
    # the robot ends up higher than it started, are the two questions a reward
    # curve cannot answer -- and both prior climb runs reported hand contact
    # (9%, 3%) that turned out to be incidental brushing, not weight-bearing.
    from mujoco_playground._src import mjx_env as _mjx_env
    def _sensor(st, name):
        return float(_mjx_env.get_sensor_data(env.mj_model, st.data, name)[0])
    hand_steps, z_start, z_last = 0, None, None

    for _ in range(n):
        rng, key = jax.random.split(rng)
        act, _ = inference(state.obs, key)
        state = step(state, act)
        state.info["command"] = jp.array([args.vx, args.vy, args.wz])
        rollout.append(state)
        try:
            if (_sensor(state, "left_hand_floor_found") > 0
                    or _sensor(state, "right_hand_floor_found") > 0):
                hand_steps += 1
        except (KeyError, ValueError, IndexError):
            pass  # scene has no hand-floor sensors (flat/rough walking tasks)
        z = float(state.data.qpos[2])
        if z_start is None:
            z_start = z
        z_last = z
        if float(state.done):
            state = reset(key)
            state.info["command"] = jp.array([args.vx, args.vy, args.wz])
            slips += 1
            # A reset teleports the robot, so height gained is only meaningful
            # within one episode. Re-baseline instead of counting the jump.
            z_start = float(state.data.qpos[2])

    # Match the rendered model to the simulated one.
    mj_model = env.mj_model
    mj_model.pair_friction[sc.FLOOR_PAIRS, 0:2] = args.friction

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
    print(f"  hand-floor contact: {hand_steps}/{n} steps "
          f"({100.0 * hand_steps / max(n, 1):.1f}%)")
    if z_start is not None:
        print(f"  height: start {z_start:+.2f} m  end {z_last:+.2f} m  "
              f"net {z_last - z_start:+.2f} m (last episode only)")


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
