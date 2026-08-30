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

import sys

import jax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# numpy-only, so this costs nothing at import time and keeps the physics
# limits, network shape and contact-pair layout in a single place.
from himalaya.env import scene as sc

# Playground ships njmax=90 (max simultaneous constraints), which is too small
# for the G1: training logged 2,580 "nefc overflow - please increase njmax"
# warnings on ordinary ground. When the solver runs out of constraint slots it
# DROPS contacts, and a dropped foot contact means nothing holds the robot up
# that step -- the likely source of floor penetration.
NJMAX = sc.NJMAX
NACONMAX = sc.NACONMAX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000_000)
    ap.add_argument("--envs", type=int, default=8192)
    ap.add_argument("--name", default=None)
    ap.add_argument("--rough", action="store_true",
                    help="rough terrain instead of flat")
    ap.add_argument("--walk-slope", type=float, default=None, metavar="DEG",
                    help="Run A, the null test: the WALKING reward on a tilted "
                         "floor, no climb term. Checks that the slope-frame "
                         "retargeting works before a long run assumes it does.")
    ap.add_argument("--climb-walk", type=float, default=None, metavar="DEG",
                    help="Run B: walk UP a slope of DEG degrees. Run A's "
                         "walking reward plus progress_uphill at 1.5.")
    ap.add_argument("--climb", type=float, default=None, metavar="DEG",
                    help="train the climbing task on a rough slope of DEG "
                         "degrees (strips the reward terms that forbid a "
                         "hands-down posture; see himalaya/env/climb_config)")
    ap.add_argument("--load", default=None, metavar="PATH",
                    help="warm-start from a saved policy (e.g. "
                         "runs/walk4_rough/policy). Restores the policy and the "
                         "observation normalizer but NOT the critic -- see "
                         "--restore-value.")
    ap.add_argument("--restore-value", action="store_true",
                    help="also restore the value function. Off by default: when "
                         "the reward changes, the old critic predicts returns "
                         "from a different MDP, and confidently wrong values "
                         "produce confidently wrong advantages. PPO's clipping "
                         "bounds the policy step, not the baseline error. "
                         "Re-learning a value head costs ~2%% of a 200M budget.")
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate. Defaults to 3e-4 cold, 1e-4 when "
                         "--load is given: a warm-started policy needs a small "
                         "step to survive its freshly-initialised critic.")
    ap.add_argument("--no-randomization", action="store_true",
                    help="train on fixed physics (the old behaviour). Useful "
                         "only as a control -- see the note by randomize below.")
    args = ap.parse_args()

    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from himalaya.env import Joystick, climb_config, default_config
    from himalaya.env import walk_on_slope_config, climb_walk_config
    from himalaya.env import randomize as g1_randomize

    # Names the vendored env uses directly; Playground's registry was
    # translating its public "G1Joystick*Terrain" ids onto these.
    if args.climb_walk is not None:
        task = "mountain_terrain"
        cfg = climb_walk_config(args.climb_walk)
    elif args.walk_slope is not None:
        task = "mountain_terrain"
        cfg = walk_on_slope_config(args.walk_slope)
    elif args.climb is not None:
        # mountain_terrain, not slope_terrain: the stock rough hfield's 5 cm
        # bumps are smaller than the hand capsule, so a palm can press on them
        # but never hook one, and climbing cannot emerge.
        task = "mountain_terrain"
        cfg = climb_config(args.climb)
    else:
        task = "rough_terrain" if args.rough else "flat_terrain"
        cfg = default_config()
    cfg.njmax = NJMAX
    cfg.naconmax = NACONMAX

    name = args.name or f"g1_{datetime.now():%H%M%S}"
    out = Path("runs") / name
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

    # Warm start, loaded before anything else so a bad path fails fast rather
    # than after the environment has compiled.
    warm_params = None
    if args.load:
        from brax.io import model as brax_model
        warm_params = brax_model.load_params(args.load)
        print(f"  warm start: {args.load}")
        print(f"    restoring policy + observation normalizer"
              f"{' + value function' if args.restore_value else ''}")
        print(f"    learning rate {args.lr if args.lr is not None else 1e-4:g}"
              f" (cold default is 3e-4)")

    print(f"run={name}  task={task}")
    print(f"  domain randomization: {'OFF (control)' if randomizer is None else 'ON'}")
    print(f"  envs={args.envs}  timesteps={args.timesteps:,}  device={jax.devices()[0]}")

    history = []
    t0 = time.time()

    def save_checkpoint(step, make_policy, params):
        del step, make_policy  # only the params are needed
        from brax.io import model as brax_model
        brax_model.save_params(str(out / "policy"), params)

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
        learning_rate=(args.lr if args.lr is not None
                       else (1e-4 if args.load else 3e-4)),
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
            policy_hidden_layer_sizes=sc.HIDDEN_LAYERS,
            value_hidden_layer_sizes=sc.HIDDEN_LAYERS,
            # The critic reads privileged_state (216 dims: contact forces,
            # true velocities, friction) while the actor reads state (103).
            # Without these keys brax defaults both to "state", so the value
            # function trains on the actor's partial view and its estimates
            # are much worse than they need to be -- this is asymmetric
            # actor-critic, and omitting it costs real sample efficiency.
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
        # Warm start. brax restores the observation normalizer along with the
        # policy and gives no flag to separate them -- that is what makes the
        # restored policy work at all, though the running statistics are stale
        # for the new terrain and take a few thousand steps to re-converge.
        restore_params=warm_params,
        restore_value_fn=args.restore_value,
        randomization_fn=randomizer,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        # Save at every eval, not just at the end. Without this a run is
        # all-or-nothing: brax hands back the params only on completion, so
        # killing a 300M-step job early leaves metrics.json and no policy,
        # and there is nothing to render. Overwrites one file rather than
        # keeping every eval -- the latest is what gets watched, and the
        # checkpoints are 2 MB each.
        policy_params_fn=save_checkpoint,
        progress_fn=progress,
        seed=0,
    )

    make_inference_fn, params, _ = train(environment=env, eval_env=eval_env)

    from brax.io import model as brax_model
    brax_model.save_params(str(out / "policy"), params)
    print(f"\nsaved -> {out/'policy'}   ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
