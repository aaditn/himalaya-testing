"""Train the G1 to walk.

    python scripts/train.py --timesteps 60_000_000
    python scripts/train.py --rough              # rough terrain instead of flat

Writes checkpoints and a metrics log to runs/<name>/.

Uses the vendored env in himalaya/env/, not Playground's registry. Rewards,
observations, and termination all live in himalaya/env/joystick.py, so change
them there directly.
"""
import argparse
import functools
import json
import time
from datetime import datetime
from pathlib import Path

import jax

# Playground ships njmax=90 (max simultaneous constraints), which is too small
# for the G1: training logged 2,580 "nefc overflow - please increase njmax"
# warnings on ordinary ground. When the solver runs out of constraint slots it
# DROPS contacts, and a dropped foot contact means nothing holds the robot up
# that step -- the likely source of floor penetration.
NJMAX = 160
NACONMAX = 131072


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000_000)
    ap.add_argument("--envs", type=int, default=8192)
    ap.add_argument("--name", default=None)
    ap.add_argument("--rough", action="store_true",
                    help="rough terrain instead of flat")
    args = ap.parse_args()

    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from himalaya.env import Joystick, default_config

    # Names the vendored env uses directly; Playground's registry was
    # translating its public "G1Joystick*Terrain" ids onto these.
    task = "rough_terrain" if args.rough else "flat_terrain"
    cfg = default_config()
    cfg.njmax = NJMAX
    cfg.naconmax = NACONMAX

    name = args.name or f"g1_{datetime.now():%H%M%S}"
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)

    env = Joystick(task=task, config=cfg)
    eval_env = Joystick(task=task, config=cfg)

    print(f"run={name}  task={task}")
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
        (out / "metrics.json").write_text(json.dumps(history, indent=1))
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
        # Playground's tuned G1 values (config/locomotion_params.py). 1e-2 is
        # the generic locomotion default and leaves too much exploration noise
        # for this robot.
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
            # The critic reads privileged_state (216 dims: contact forces,
            # true velocities, friction) while the actor reads state (103).
            # Without these keys brax defaults both to "state", so the value
            # function trains on the actor's partial view and its estimates
            # are much worse than they need to be -- this is asymmetric
            # actor-critic, and omitting it costs real sample efficiency.
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress,
        seed=0,
    )

    make_inference_fn, params, _ = train(environment=env, eval_env=eval_env)

    from brax.io import model as brax_model
    brax_model.save_params(str(out / "policy"), params)
    print(f"\nsaved -> {out/'policy'}   ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
