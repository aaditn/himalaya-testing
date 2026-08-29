"""Train G1 to walk on ice.

    python scripts/train_ice.py --timesteps 60_000_000
    python scripts/train_ice.py --baseline          # normal friction, for comparison
    python scripts/train_ice.py --mixed             # ice-to-dry, patchy surface

Writes checkpoints + a metrics log to runs/<name>/. The baseline run matters
for the demo: "here is a normal policy on ice" vs "here is ours" is the
clearest way to show the result.
"""
import argparse
import functools
import json
import time
from datetime import datetime
from pathlib import Path

import jax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=60_000_000)
    ap.add_argument("--envs", type=int, default=8192)
    ap.add_argument("--name", default=None)
    ap.add_argument("--baseline", action="store_true",
                    help="stock friction U(0.4,1.0) -- the control condition")
    ap.add_argument("--mixed", action="store_true",
                    help="friction U(0.05,0.9) -- patchy ice-to-dry ground")
    ap.add_argument("--no-arm-swing", action="store_true",
                    help="keep the stock pose penalty, pinning the arms")
    ap.add_argument("--video-every", type=int, default=3,
                    help="record an MP4 every N evals (0 = off). Demo is 20%% "
                         "of the hackathon score, so footage accumulates as "
                         "training runs rather than only at the end.")
    args = ap.parse_args()

    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import registry, wrapper

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from himalaya.ice import ice_env

    task = "G1JoystickFlatTerrain"
    base_cfg = registry.get_default_config(task)

    if args.baseline:
        # Baseline still needs the njmax fix -- it overflowed 2,580 times too.
        cfg = base_cfg.copy_and_resolve_references()
        cfg.njmax = ice_env.NJMAX
        cfg.naconmax = ice_env.NACONMAX
        randomizer, tag = None, "baseline"
    elif args.mixed:
        cfg = ice_env.ice_config(base_cfg, arm_swing=not args.no_arm_swing)
        randomizer, tag = ice_env.mixed_randomize, "mixed"
    else:
        cfg = ice_env.ice_config(base_cfg, arm_swing=not args.no_arm_swing)
        randomizer, tag = ice_env.ice_randomize, "ice"

    name = args.name or f"{tag}_{datetime.now():%H%M%S}"
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)

    env = ice_env.patch_termination(registry.load(task, config=cfg))
    eval_env = ice_env.patch_termination(registry.load(task, config=cfg))

    print(f"run={name}  task={task}  condition={tag}")
    print(f"  friction: {'stock U(0.4,1.0)' if args.baseline else ice_env.ICE_ONLY if tag=='ice' else ice_env.MIXED_TERRAIN}")
    print(f"  feet_slip={cfg.reward_config.scales.feet_slip}  "
          f"ang_vel_xy={cfg.reward_config.scales.ang_vel_xy}  "
          f"pose={cfg.reward_config.scales.pose}")
    print(f"  envs={args.envs}  timesteps={args.timesteps:,}  device={jax.devices()[0]}")

    history = []
    t0 = time.time()
    eval_n = [0]
    policy_params_holder = [None]

    def snapshot_video(params):
        """Render a short clip from the current policy, mid-training."""
        try:
            import jax as _jax
            import jax.numpy as _jp
            import mediapy
            from brax.training.agents.ppo import networks as _nets

            net = _nets.make_ppo_networks(
                env.observation_size, env.action_size,
                policy_hidden_layer_sizes=(512, 256, 128),
                value_hidden_layer_sizes=(512, 256, 128),
            )
            inf = _jax.jit(_nets.make_inference_fn(net)(params, deterministic=True))
            st = _jax.jit(eval_env.reset)(_jax.random.PRNGKey(0))
            st.info["command"] = _jp.array([0.8, 0.0, 0.0])
            stepfn = _jax.jit(eval_env.step)
            roll = []
            rng = _jax.random.PRNGKey(1)
            for _ in range(int(6.0 / eval_env.dt)):
                rng, k = _jax.random.split(rng)
                a, _ = inf(st.obs, k)
                st = stepfn(st, a)
                st.info["command"] = _jp.array([0.8, 0.0, 0.0])
                roll.append(st)
            frames = eval_env.render(roll, camera="track", width=640, height=440)
            vp = Path("videos") / f"{name}_step{history[-1]['step']}.mp4"
            vp.parent.mkdir(parents=True, exist_ok=True)
            mediapy.write_video(str(vp), frames, fps=1.0 / eval_env.dt)
            print(f"      video -> {vp}")
        except Exception as e:  # never let recording kill a training run
            print(f"      (video skipped: {type(e).__name__}: {e})")

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
        eval_n[0] += 1
        if args.video_every and eval_n[0] % args.video_every == 0:
            snapshot_video(policy_params_holder[0]) if policy_params_holder[0] else None

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
        entropy_cost=1e-2,
        clipping_epsilon=0.2,
        action_repeat=1,
        max_grad_norm=1.0,
        normalize_observations=True,
        reward_scaling=1.0,
        network_factory=functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
        ),
        policy_params_fn=lambda step, make_fn, params: policy_params_holder.__setitem__(0, params),
        randomization_fn=randomizer,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress,
        seed=0,
    )

    make_inference_fn, params, _ = train(environment=env, eval_env=eval_env)
    snapshot_video(params)

    from brax.io import model as brax_model
    brax_model.save_params(str(out / "policy"), params)
    print(f"\nsaved -> {out/'policy'}   ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
