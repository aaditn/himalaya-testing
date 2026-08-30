"""Render a trained policy to MP4 on the pod.

    python scripts/record.py runs/g1_120000/policy --out videos/walk.mp4
    python scripts/record.py runs/g1_120000/policy --seconds 12

Renders offscreen on the GPU -- no display needed -- so it works while
training is running. Pull the result with scripts/pod/pull.sh.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", help="path to a saved brax params file")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--friction", type=float, default=0.95,
                    help="hand microspike friction to render at")
    ap.add_argument("--foot-friction", type=float, default=1.90,
                    help="foot microspike friction to render at")
    ap.add_argument("--vx", type=float, default=0.8, help="commanded forward vel")
    ap.add_argument("--action-scale", type=float, default=0.35,
                    help="residual action scale used during training")
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0, help="commanded yaw rate")
    ap.add_argument("--rough", action="store_true")
    ap.add_argument("--climb", action="store_true")
    ap.add_argument("--slope", type=float, default=12.0)
    ap.add_argument("--roughness", type=float, default=0.060)
    ap.add_argument("--no-boulders", action="store_true")
    ap.add_argument("--zero-policy", action="store_true",
                    help="evaluate only the phase-conditioned residual reference")
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
    os.environ.setdefault(
        "MPLCONFIGDIR", str((Path(".tmp") / "matplotlib").resolve())
    )
    import imageio_ffmpeg
    import mediapy
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    from brax.io import model as brax_model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    from himalaya.env import Joystick, default_config

    # Same env as training -- himalaya/env/ is the single definition, so a
    # reward or termination change cannot drift between train and record.
    NJMAX, NACONMAX = 256, 131072
    # Names the vendored env uses directly; Playground's registry was
    # translating its public "G1Joystick*Terrain" ids onto these.
    task = (
        "climb_terrain" if args.climb
        else "rough_terrain" if args.rough
        else "flat_terrain"
    )
    cfg = default_config()
    cfg.njmax = NJMAX
    cfg.naconmax = NACONMAX
    if args.climb:
        cfg.impl = "jax"
        cfg.climb.slope_degrees = args.slope
        cfg.climb.roughness_m = args.roughness
        cfg.climb.boulders_enabled = not args.no_boulders
        cfg.climb.spike_friction = args.friction
        cfg.climb.foot_spike_friction = args.foot_friction
        cfg.climb.residual_action_scale = args.action_scale
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
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
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
    floor_pair_count = env.mj_model.npair - 3 if args.climb else 2
    foot_pair_ids = [
        pair_id for pair_id in range(floor_pair_count)
        if env.mj_model.pair(pair_id).name.startswith(
            ("left_foot_", "right_foot_")
        )
    ]
    hand_pair_ids = [
        pair_id for pair_id in range(floor_pair_count)
        if env.mj_model.pair(pair_id).name.startswith(
            ("left_hand_", "right_hand_")
        )
    ]
    pair_friction = env._mjx_model.pair_friction
    pair_friction = pair_friction.at[jp.array(foot_pair_ids), 0:2].set(
        args.foot_friction
    )
    if hand_pair_ids:
        pair_friction = pair_friction.at[jp.array(hand_pair_ids), 0:2].set(
            args.friction
        )
    env._mjx_model = env._mjx_model.tree_replace({
        "pair_friction": pair_friction
    })

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    state = reset(jax.random.PRNGKey(0))
    # hold a fixed velocity command so the clip shows deliberate walking
    state.info["command"] = jp.array([args.vx, args.vy, args.wz])

    n = int(args.seconds / env.dt)
    rollout, actions, slips = [], [], 0
    rng = jax.random.PRNGKey(1)
    for _ in range(n):
        rng, key = jax.random.split(rng)
        if args.zero_policy:
            act = jp.zeros(env.action_size)
        else:
            act, _ = inference(state.obs, key)
        actions.append(act)
        state = step(state, act)
        state.info["command"] = jp.array([args.vx, args.vy, args.wz])
        rollout.append(state)
        if float(state.done):
            state = reset(key)
            state.info["command"] = jp.array([args.vx, args.vy, args.wz])
            slips += 1

    # Match the rendered model to the simulated one.
    mj_model = env.mj_model
    mj_model.pair_friction[foot_pair_ids, 0:2] = args.foot_friction
    if hand_pair_ids:
        mj_model.pair_friction[hand_pair_ids, 0:2] = args.friction

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
    print(
        f"  hand_friction={args.friction} foot_friction={args.foot_friction}  "
        f"command=({args.vx},{args.vy},{args.wz})"
    )
    print(f"  falls during clip: {slips}")
    if args.climb:
        import numpy as np

        qpos = np.stack([np.asarray(item.data.qpos) for item in rollout])
        action_array = np.stack([np.asarray(item) for item in actions])
        feet = np.stack([
            np.asarray(item.info["last_contact"]) for item in rollout
        ])
        hands = np.stack([
            np.asarray(item.info["last_hand_contact"]) for item in rollout
        ])
        uphill_axis = np.asarray(env._slope_tangent)
        episode_start = np.asarray([
            item.info["start_uphill_position"] for item in rollout
        ])
        uphill = qpos[:, :3] @ uphill_axis - episode_start
        foot_z = np.stack([
            np.asarray(item.data.site_xpos[env._feet_site_id, 2])
            for item in rollout
        ])
        foot_reach = np.stack([
            np.asarray(item.data.site_xpos[env._feet_site_id]) @ uphill_axis
            for item in rollout
        ])
        hand_reach = np.stack([
            np.asarray(item.data.site_xpos[env._hands_site_id]) @ uphill_axis
            for item in rollout
        ])
        slope_normal = np.asarray(env._slope_normal)
        hand_clearance = np.stack([
            np.asarray(item.data.site_xpos[env._hands_site_id]) @ slope_normal
            - env._terrain_plane_offset
            for item in rollout
        ])
        diagonal_sync = np.mean((~hands) == (~feet[:, ::-1]))
        motor_targets = np.stack([
            np.asarray(item.info["motor_targets"]) for item in rollout
        ])
        phase = np.stack([
            np.asarray(item.info["phase"]) for item in rollout
        ])
        # Central two thirds of the half-wave excludes expected contact at
        # liftoff/touchdown boundaries and measures failure to unload.
        desired_foot_swing = np.cos(phase) > 0.5
        swing_contact = np.sum(feet & desired_foot_swing, axis=0) / np.maximum(
            np.sum(desired_foot_swing, axis=0), 1
        )
        leg_indices = [0, 3, 6, 9]
        arm_indices = [15, 18, 22, 25]
        pelvis_clearance = (
            qpos[:, :3] @ slope_normal - env._terrain_plane_offset
        )
        uphill_velocity = np.asarray([
            item.metrics["climb/uphill_velocity"] for item in rollout
        ])
        velocity_reward = np.asarray([
            item.metrics["reward/uphill_progress"] for item in rollout
        ])
        hand_lift_metric = np.asarray([
            item.metrics["climb/hand_lift_height"] for item in rollout
        ])
        hand_lift_events = np.asarray([
            item.metrics["climb/hand_lift_target"] for item in rollout
        ])
        hand_lift_reward = np.asarray([
            item.metrics["reward/hand_lift_height"] for item in rollout
        ])
        knee_clearance = np.asarray([
            item.metrics["climb/knee_clearance_min"] for item in rollout
        ])
        knee_contact = np.asarray([
            item.metrics["climb/knee_contact_fraction"] for item in rollout
        ])
        knee_clearance_reward = np.asarray([
            item.metrics["reward/knee_clearance"] for item in rollout
        ])
        large_foot_step_bonus = np.asarray([
            item.metrics["climb/large_foot_step_bonus"] for item in rollout
        ])
        foot_swing_contact_metric = np.asarray([
            item.metrics["climb/foot_swing_contact_fraction"]
            for item in rollout
        ])
        hand_swing_contact_metric = np.asarray([
            item.metrics["climb/hand_swing_contact_fraction"]
            for item in rollout
        ])
        support_exchange_metric = np.asarray([
            item.metrics["climb/support_exchange"] for item in rollout
        ])
        overspeed_ratio = np.asarray([
            item.metrics["climb/overspeed_ratio"] for item in rollout
        ])
        print("  rollout diagnostics:")
        print(
            f"    uphill final/max/min={uphill[-1]:+.3f}/"
            f"{uphill.max():+.3f}/{uphill.min():+.3f} m"
        )
        print(
            f"    action rms={np.sqrt(np.mean(action_array**2)):.3f}; "
            f"joint peak-to-peak mean={np.ptp(qpos[:, 7:], axis=0).mean():.3f} rad"
        )
        print(
            f"    foot contact={feet.mean(axis=0)}; transitions="
            f"{np.count_nonzero(np.diff(feet.astype(int), axis=0), axis=0)}"
        )
        print(
            f"    hand contact={hands.mean(axis=0)}; transitions="
            f"{np.count_nonzero(np.diff(hands.astype(int), axis=0), axis=0)}"
        )
        print(
            f"    foot vertical excursion={np.ptp(foot_z, axis=0)} m; "
            f"foot uphill excursion={np.ptp(foot_reach, axis=0)} m; "
            f"hand uphill excursion={np.ptp(hand_reach, axis=0)} m"
        )
        print(
            f"    hand terrain-normal excursion="
            f"{np.ptp(hand_clearance, axis=0)} m; "
            f"diagonal swing sync={diagonal_sync:.3f}"
        )
        print(
            f"    foot contact during scheduled swing={swing_contact}; "
            f"leg target p2p="
            f"{np.ptp(motor_targets[:, leg_indices], axis=0)}; "
            f"actual p2p="
            f"{np.ptp(qpos[:, 7 + np.asarray(leg_indices)], axis=0)}"
        )
        print(
            f"    phase contact mean: foot={foot_swing_contact_metric.mean():.3f} "
            f"hand={hand_swing_contact_metric.mean():.3f}; "
            f"support exchange={support_exchange_metric.mean():.3f}; "
            f"overspeed ratio max={overspeed_ratio.max():.2f}x"
        )
        print(
            f"    arm target p2p={np.ptp(motor_targets[:, arm_indices], axis=0)}; "
            f"actual p2p={np.ptp(qpos[:, 7 + np.asarray(arm_indices)], axis=0)}"
        )
        print(
            f"    pelvis terrain-normal min/max="
            f"{pelvis_clearance.min():.3f}/{pelvis_clearance.max():.3f} m"
        )
        print(
            f"    uphill velocity mean/max={uphill_velocity.mean():+.3f}/"
            f"{uphill_velocity.max():+.3f} m/s; "
            f"scaled velocity reward mean={velocity_reward.mean():.3f}"
        )
        print(
            f"    hand lift max={hand_lift_metric.max():.3f} m / "
            f"target={env._config.reward_config.max_hand_height:.5f} m; "
            f"target events={int(hand_lift_events.sum())}; "
            f"scaled lift reward mean={hand_lift_reward.mean():.3f}"
        )
        print(
            f"    knee surface clearance min/mean="
            f"{knee_clearance.min():.3f}/{knee_clearance.mean():.3f} m; "
            f"contact-frame fraction={np.mean(knee_contact > 0):.3f}; "
            f"scaled clearance reward mean="
            f"{knee_clearance_reward.mean():.3f}"
        )
        print(
            f"    large-foot-step events="
            f"{np.count_nonzero(large_foot_step_bonus > 0)}; "
            f"bonus sum/max={large_foot_step_bonus.sum():.3f}/"
            f"{large_foot_step_bonus.max():.3f}"
        )


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
