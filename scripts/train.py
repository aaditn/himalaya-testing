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
from datetime import UTC, datetime
from pathlib import Path

import jax

# Playground ships njmax=90 (max simultaneous constraints), which is too small
# for the G1: training logged 2,580 "nefc overflow - please increase njmax"
# warnings on ordinary ground. When the solver runs out of constraint slots it
# DROPS contacts, and a dropped foot contact means nothing holds the robot up
# that step -- the likely source of floor penetration.
NJMAX = 256
NACONMAX = 131072


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000_000)
    ap.add_argument("--envs", type=int, default=8192)
    ap.add_argument("--name", default=None)
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="root directory for checkpoints, metrics, and the final policy",
    )
    ap.add_argument("--rough", action="store_true",
                    help="rough terrain instead of flat")
    ap.add_argument("--climb", action="store_true",
                    help="continuous four-limb ascent task")
    ap.add_argument("--slope", type=float, default=12.0,
                    help="climb grade in degrees")
    ap.add_argument("--roughness", type=float, default=0.060,
                    help="height-field relief in metres")
    ap.add_argument("--spike-friction", type=float, default=0.95)
    ap.add_argument("--foot-friction", type=float, default=1.90,
                    help="foot microspike friction; defaults to twice the hands")
    ap.add_argument("--no-boulders", action="store_true",
                    help="move boulders below the terrain for crawl bootstrap")
    ap.add_argument("--hand-load", type=float, default=0.28,
                    help="target fraction of limb load carried by hands")
    ap.add_argument("--speed", type=float, default=0.30,
                    help="target uphill speed in m/s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restore", default=None,
                    help="Orbax checkpoint or checkpoint directory to resume")
    ap.add_argument("--num-evals", type=int, default=None)
    ap.add_argument("--eval-envs", type=int, default=None)
    ap.add_argument("--no-randomization", action="store_true",
                    help="train on fixed physics (the old behaviour). Useful "
                         "only as a control -- see the note by randomize below.")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from himalaya.utils.jax_compat import install_brax_compatibility

    install_brax_compatibility()

    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    from himalaya.env import Joystick, default_config
    from himalaya.env import randomize as g1_randomize

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
        cfg.climb.spike_friction = args.spike_friction
        cfg.climb.foot_spike_friction = args.foot_friction
        cfg.climb.boulders_enabled = not args.no_boulders
        cfg.climb.target_hand_load_share = args.hand_load
        cfg.climb.target_uphill_speed = args.speed

    name = args.name or f"g1_{datetime.now(UTC):%H%M%S}"
    out = (args.runs_dir / name).resolve()
    out.mkdir(parents=True, exist_ok=True)

    env = Joystick(task=task, config=cfg)
    eval_env = Joystick(task=task, config=cfg)

    # Domain randomization. The notebook (docs/playground/notebooks/
    # locomotion.ipynb, cell 46) passes this to ppo.train, and omitting it
    # trains a policy against one fixed set of physics -- friction, masses and
    # armature never vary, so nothing forces a gait that survives a different
    # floor. Six parameters are perturbed per environment; floor/foot friction
    # U(0.4, 1.0) is the one that changes the gait most.
    #
    # Expect a LOWER reward curve than a fixed-physics run. The task is
    # genuinely harder, and the comparison that matters is robustness across a
    # friction sweep, not the training reward.
    randomizer = None if args.no_randomization else g1_randomize.domain_randomize

    print(f"run={name}  task={task}")
    print(f"  domain randomization: {'OFF (control)' if randomizer is None else 'ON'}")
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

    num_minibatches = 32
    batch_size = 256
    if args.envs < 512:
        # MODIFIED: permit CPU/API smoke runs without changing production PPO.
        divisors = [value for value in range(1, 9) if args.envs % value == 0]
        num_minibatches = max(divisors)
        batch_size = (args.envs // num_minibatches) * 4
    num_evals = args.num_evals or (1 if args.timesteps < 1_000_000 else 15)
    num_eval_envs = args.eval_envs or min(128, max(8, args.envs // 16))

    train = functools.partial(
        ppo.train,
        num_timesteps=args.timesteps,
        num_evals=num_evals,
        num_eval_envs=num_eval_envs,
        episode_length=cfg.episode_length,
        num_envs=args.envs,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
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
        randomization_fn=randomizer,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress,
        seed=args.seed,
        save_checkpoint_path=str(out / "checkpoints"),
        restore_checkpoint_path=args.restore,
    )

    _make_inference_fn, params, _ = train(environment=env, eval_env=eval_env)

    from brax.io import model as brax_model
    brax_model.save_params(str(out / "policy"), params)
    print(f"\nsaved -> {out/'policy'}   ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
