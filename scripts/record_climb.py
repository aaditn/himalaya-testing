"""Render a trained climbing policy to MP4.

    python scripts/record_climb.py runs/<name>/policy --slope 37.5 --friction 1.0

Every change to the climb env, its rewards or its scene ships with one of these.
A reward curve answers a different question: training eval averages over slope
angles AND friction draws, including the low-mu draws at the steep end that are
physically unclimbable (mu < tan(theta)). A clip at a stated slope and friction
is the only thing that shows whether it climbs.
"""
import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--slope", type=float, default=37.5, help="degrees")
    ap.add_argument("--friction", type=float, default=1.0)
    ap.add_argument("--flat", action="store_true", help="flat control scene")
    ap.add_argument("--camera", default="side", help="side/front/chase")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    # Training runs MJX at 3 solver iterations / 5 line-search -- the speed
    # budget -- and under a 45 deg load that visibly sinks feet into the
    # plane. Recording is offline, so it steps at full solver quality. The
    # policy transfers fine; the clip reports min foot clearance so the fix
    # is verifiable, not cosmetic.
    ap.add_argument("--iterations", type=int, default=20)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import jax
    import jax.numpy as jp
    import mediapy
    import numpy as np
    from brax.io import model as brax_model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from himalaya.env.climb import Climb, default_config
    from himalaya.env.randomize_climb import ground_pair_ids

    cfg = default_config()
    cfg.njmax, cfg.naconmax = 300, 16 * 8192
    # The slope is COMPILED into the scene, not written at runtime. The warp
    # backend bakes static geom poses at put_model, so the old runtime
    # geom_quat write tilted the spawn and the reward frame while the
    # collision plane stayed at the scene's 37.5 deg -- which is what "the
    # robot keeps going through the floor at 45 deg" actually was.
    if not args.flat:
        cfg.slope_compile_deg = float(args.slope)
    env = Climb(task="flat" if args.flat else "incline", config=cfg)

    # preprocess_observations_fn is NOT optional: training ran with
    # normalize_observations=True, so the policy learned on normalised inputs.
    # Omitting it loads the normaliser params and then ignores them, feeding raw
    # observations. It fails silently -- measured on the walking task, mean
    # episode length 70 without it and 905 with it.
    net = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
        preprocess_observations_fn=running_statistics.normalize,
    )
    params = brax_model.load_params(args.policy)
    inference = jax.jit(ppo_networks.make_inference_fn(net)(params, deterministic=True))

    # Friction and solver settings go onto the MJ MODEL and the mjx model is
    # RE-PUT from it. Runtime tree_replace writes are impl-dependent (warp
    # bakes some of them); mj-level-then-put is honoured by every backend.
    # Pair indices resolved BY NAME: the climb scene's table puts the feet at
    # 2 and 5, so record.py's hardcoded [0:2] would hit thigh and shin instead.
    theta = np.deg2rad(args.slope if not args.flat else 0.0)
    fid = int(env.mj_model.geom("floor").id)
    pids = np.array(ground_pair_ids(env.mj_model))
    mj0 = env.mj_model
    mj0.pair_friction[pids, 0:2] = args.friction
    # Stiffer contact for the camera: the training solref (20 ms time
    # constant) lets a loaded foot corner sink ~17 mm at push-off, which
    # reads on video as clipping through the floor. 10 ms (the 2*dt floor)
    # halves the sag; the clip prints min clearance so it stays verifiable.
    mj0.pair_solref[:, 0] = 0.01
    mj0.pair_solref[:, 1] = 1.0
    mj0.opt.iterations = args.iterations
    mj0.opt.ls_iterations = args.iterations
    from mujoco import mjx as _mjx
    env._mjx_model = _mjx.put_model(mj0, impl=env._mjx_model.impl.value)

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(0))

    n = int(args.seconds / env.dt)
    rollout, falls = [], 0
    rng = jax.random.PRNGKey(1)
    start_h = None
    climbed = []
    uphill = np.array([np.cos(theta), 0.0, np.sin(theta)])
    for _ in range(n):
        rng, key = jax.random.split(rng)
        act, _ = inference(state.obs, key)
        state = step(state, act)
        rollout.append(state)
        pos = np.array(state.data.qpos[0:3])
        if start_h is None:
            start_h = float(pos @ uphill)
        climbed.append(float(pos @ uphill) - start_h)
        if float(state.done):
            state = reset(key)
            falls += 1
            start_h = None

    # The rendered model IS the simulated one now: the slope is compiled in
    # and friction/solver were set on mj_model before put_model.
    mj = env.mj_model

    import mujoco as _mj

    # The bare plane renders as a near-black featureless field from the side
    # camera, which makes gait unreadable -- a skating policy LOOKED like a
    # crawl until the contact logs said otherwise. Brighter light plus 1 m
    # stripes across the slope give motion, scale, and progress a reference.
    mj.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    mj.vis.headlight.diffuse[:] = [0.9, 0.9, 0.9]

    # Presentation ground: the rock texture rendered near-black and its
    # specular 0.5 made it glossy. Matte warm grey, no texture, no shine.
    mat = mj.geom_matid[fid]
    mj.mat_texid[mat, :] = -1
    mj.mat_rgba[mat] = [0.62, 0.58, 0.52, 1.0]
    mj.mat_specular[mat] = 0.0
    mj.mat_reflectance[mat] = 0.0

    uphill3 = np.array([np.cos(theta), 0.0, np.sin(theta)])
    normal3 = np.array([-np.sin(theta), 0.0, np.cos(theta)])
    stripe_mat = np.stack([uphill3, np.array([0.0, 1.0, 0.0]), normal3],
                          axis=1).flatten()

    # Pebble field: fixed scatter of small matte stones. They give the eye
    # parallax between stripes, so motion reads even in a still-ish shot.
    peb_rng = np.random.default_rng(7)
    pebbles = []
    for _ in range(140):
        along = peb_rng.uniform(-3.0, 16.0)
        across = peb_rng.uniform(-2.2, 2.2)
        r = peb_rng.uniform(0.015, 0.05)
        shade = peb_rng.uniform(0.30, 0.48)
        pebbles.append((along * uphill3 + across * np.array([0.0, 1.0, 0.0])
                        + r * 0.4 * normal3, r,
                        [shade, shade * 0.95, shade * 0.88, 1.0]))

    def add_stripes(scene):
        for k in range(-3, 17):
            if scene.ngeom >= scene.maxgeom:
                break
            g = scene.geoms[scene.ngeom]
            rgba = [0.85, 0.30, 0.22, 1.0] if k % 5 == 0 else [0.92, 0.92, 0.90, 0.9]
            _mj.mjv_initGeom(
                g, _mj.mjtGeom.mjGEOM_BOX,
                np.array([0.02, 1.8, 0.004]),
                (k * uphill3 + 0.004 * normal3).astype(np.float64),
                stripe_mat.astype(np.float64),
                np.array(rgba, dtype=np.float32))
            scene.ngeom += 1
        for pos, r, rgba in pebbles:
            if scene.ngeom >= scene.maxgeom:
                break
            g = scene.geoms[scene.ngeom]
            _mj.mjv_initGeom(
                g, _mj.mjtGeom.mjGEOM_ELLIPSOID,
                np.array([r, r * 0.8, r * 0.4]),
                pos.astype(np.float64),
                stripe_mat.astype(np.float64),
                np.array(rgba, dtype=np.float32))
            scene.ngeom += 1

    cam = _mj.MjvCamera()
    cam.type = _mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.3]
    cam.distance = {"side": 3.5, "front": 3.0, "chase": 4.0}.get(args.camera, 3.5)
    cam.azimuth = {"side": 90.0, "front": 180.0, "chase": 135.0}.get(args.camera, 90.0)
    cam.elevation = -12.0

    d = _mj.MjData(mj)
    renderer = _mj.Renderer(mj, height=args.height, width=args.width)
    frames = []
    foot_gids = [mj.geom("left_foot").id, mj.geom("right_foot").id]
    min_clear = np.inf
    for st in rollout:
        d.qpos[:] = np.array(st.data.qpos)
        d.qvel[:] = np.array(st.data.qvel)
        _mj.mj_forward(mj, d)
        for g in foot_gids:
            R = d.geom_xmat[g].reshape(3, 3)
            c = float(d.geom_xpos[g] @ normal3) - float(
                np.abs(R.T @ normal3) @ mj.geom_size[g])
            min_clear = min(min_clear, c)
        cam.lookat[0] = float(d.qpos[0])
        cam.lookat[1] = float(d.qpos[1])
        # Track z as well: on a slope the robot GAINS height, and a camera
        # pinned at z=0.3 loses it out of the top of the frame within metres.
        cam.lookat[2] = float(d.qpos[2])
        renderer.update_scene(d, camera=cam)
        add_stripes(renderer.scene)
        frames.append(renderer.render())
    renderer.close()

    out = Path(args.out or f"videos/climb_{Path(args.policy).parent.name}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(out), frames, fps=1.0 / env.dt)
    print(f"wrote {out}  ({len(frames)} frames, {args.seconds}s)")
    print(f"  slope={args.slope} deg  friction={args.friction}  (mu needed {np.tan(theta):.2f})")
    print(f"  falls during clip: {falls}")
    print(f"  net distance up the slope: {climbed[-1]:+.2f} m   max {max(climbed):+.2f} m")
    print(f"  min foot clearance: {min_clear*1000:+.0f} mm  "
          f"(negative = sunk into the plane; solver iterations {args.iterations})")


if __name__ == "__main__":
    main()
