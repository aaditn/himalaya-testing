"""Train the G1 to climb a 30-45 degree slope on all four limbs.

    python scripts/train_climb.py --timesteps 60_000_000
    python scripts/train_climb.py --flat          # control: same task, no slope

Writes checkpoints and a metrics log to runs/<name>/.

The task lives in himalaya/env/climb.py and the scenes in
himalaya/env/xmls/scene_mjx_climb_*.xml. Both are ours to edit.
"""
import argparse
import functools
import json
import time
from datetime import datetime
from pathlib import Path

import jax

# Four limbs on a slope make far more simultaneous contacts than a biped on
# flat ground: 8 limb-vs-floor pairs instead of 2. The walking task already
# needed 160 (stock 90 overflowed 2,580 times); the climb config raises it
# again. Overridable here for the same reason it is in train.py.
NJMAX = 300
NACONMAX = 16 * 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=100_000_000)
    ap.add_argument("--envs", type=int, default=8192)
    ap.add_argument("--name", default=None)
    ap.add_argument("--flat", action="store_true",
                    help="flat control scene instead of the slope")
    ap.add_argument("--slope-min", type=float, default=30.0)
    ap.add_argument("--slope-max", type=float, default=45.0)
    # Ideal-conditions grip: 2x tan(45 deg) at the bottom of the band, so no
    # environment is ever friction-limited. See randomize_climb.py.
    ap.add_argument("--mu-min", type=float, default=2.0)
    ap.add_argument("--mu-max", type=float, default=3.0)
    ap.add_argument("--scripted-gait", action="store_true",
                    help="scripted trot baseline + RL residual (climb.py)")
    ap.add_argument("--gait-freq-min", type=float, default=None)
    ap.add_argument("--gait-freq-max", type=float, default=None)
    # Compile the slope into the scene and train at that ONE angle. This is
    # currently the only way a non-default angle physically exists under the
    # warp backend: randomize_climb's geom_quat writes tilt the reward frame
    # and spawn but the collision plane stays at the scene's compile angle
    # (verified via data.geom_xmat). With this flag the runtime band is forced
    # to the same angle so frames and physics agree.
    ap.add_argument("--slope-compile", type=float, default=None)
    ap.add_argument("--no-randomization", action="store_true",
                    help="fixed slope and friction (control only)")
    args = ap.parse_args()

    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from himalaya.env.climb import Climb, default_config
    from himalaya.env import randomize_climb

    task = "flat" if args.flat else "incline"
    cfg = default_config()
    cfg.njmax = NJMAX
    cfg.naconmax = NACONMAX
    if args.scripted_gait:
        cfg.scripted_gait = True
    if args.gait_freq_min is not None:
        cfg.gait_freq_range = (args.gait_freq_min, args.gait_freq_max)
    if args.slope_compile is not None:
        cfg.slope_compile_deg = args.slope_compile
        args.slope_min = args.slope_max = args.slope_compile

    name = args.name or f"climb_{datetime.now():%H%M%S}"
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)

    env = Climb(task=task, config=cfg)
    eval_env = Climb(task=task, config=cfg)

    # Slope angle AND slipperiness are randomised per environment. The pair
    # indices are resolved BY NAME and bound here -- see randomize_climb.py for
    # why a hardcoded slice silently randomises the wrong geoms in this scene.
    randomizer = None
    if not args.no_randomization:
        # The flat control keeps friction/mass/armature randomisation but pins
        # the slope at 0: tilting the control's floor would re-create the very
        # condition it exists to isolate.
        slope = (0.0, 0.0) if args.flat else (args.slope_min, args.slope_max)
        mu = (args.mu_min, args.mu_max)
        randomizer = functools.partial(
            randomize_climb.domain_randomize,
            floor_geom_id=int(env.mj_model.geom("floor").id),
            ground_pair_ids=randomize_climb.ground_pair_ids(env.mj_model),
            slope_deg=slope,
            friction=mu,
        )

    print(f"run={name}  task={task}")
    dr = ('OFF (control)' if randomizer is None
          else f'ON (slope {slope[0]}-{slope[1]} deg, mu {mu[0]}-{mu[1]})')
    print(f"  domain randomization: {dr}")
    print(f"  envs={args.envs}  timesteps={args.timesteps:,}  device={jax.devices()[0]}")

    history = []
    t0 = time.time()

    def progress(step, metrics):
        row = {
            "step": int(step),
            "elapsed_s": round(time.time() - t0, 1),
            "reward": float(metrics.get("eval/episode_reward", 0.0)),
            "episode_len": float(metrics.get("eval/avg_episode_length", 0.0)),
        }
        for k, v in metrics.items():
            if k.startswith("eval/episode_reward/"):
                row[k.split("/")[-1]] = round(float(v), 4)
        history.append(row)
        # Best-effort: runs/ lives on an HF bucket FUSE mount whose mkdir is
        # eventually consistent -- a transient ENOENT here killed two full
        # training runs (headup-11, real37b). A lost metrics snapshot costs
        # nothing; a lost run costs a GPU-hour.
        try:
            out.mkdir(parents=True, exist_ok=True)
            (out / "metrics.json").write_text(json.dumps(history, indent=1))
        except OSError as e:
            print(f"  metrics write failed ({e}); continuing", flush=True)
        print(f"  {row['step']:>11,}  reward={row['reward']:8.2f}  "
              f"len={row['episode_len']:7.1f}  ({row['elapsed_s']:.0f}s)", flush=True)

    train = functools.partial(
        ppo.train,
        num_timesteps=args.timesteps,
        num_evals=15,
        episode_length=cfg.episode_length,
        num_envs=args.envs,
        batch_size=256,
        num_minibatches=32,
        unroll_length=20,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=0.005,
        clipping_epsilon=0.2,
        num_resets_per_eval=1,
        action_repeat=1,
        max_grad_norm=1.0,
        normalize_observations=True,
        reward_scaling=1.0,
        network_factory=functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
        randomization_fn=randomizer,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress,
        seed=0,
    )

    make_inference_fn, params, _ = train(environment=env, eval_env=eval_env)

    from brax.io import model as brax_model
    # The bucket mount can hiccup; the policy is the only unrecoverable
    # artifact, so retry rather than die at the finish line.
    for attempt in range(5):
        try:
            out.mkdir(parents=True, exist_ok=True)
            brax_model.save_params(str(out / "policy"), params)
            break
        except OSError as e:
            print(f"  policy save failed (attempt {attempt+1}: {e})", flush=True)
            time.sleep(5)
    print(f"\nsaved -> {out/'policy'}   ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
