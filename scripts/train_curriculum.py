"""Train G1 to walk on ice via a friction curriculum.

    python scripts/train_curriculum.py --timesteps 400_000_000

Training directly at ice friction fails: the policy converges to ~8-step
episodes because nearly every action ends in a fall, so there is no gradient
toward walking. This starts on dry ground and steps friction down through
FRICTION_STAGES, each stage warm-starting from the previous stage's weights.

Brax fixes the randomization function when it builds the training loop, so
the curriculum runs as sequential ppo.train calls rather than one call with a
changing randomizer. Each stage keeps the network parameters; only the floor
changes underneath.
"""
import argparse
import functools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=400_000_000,
                    help="total across all stages")
    ap.add_argument("--envs", type=int, default=8192)
    ap.add_argument("--name", default=None)
    ap.add_argument("--stages", type=int, default=None,
                    help="how many curriculum stages to run (default: all)")
    ap.add_argument("--video-every-stage", action="store_true", default=True)
    args = ap.parse_args()

    import jax
    from brax.io import model as brax_model
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import registry, wrapper

    from himalaya.ice import ice_env

    task = "G1JoystickFlatTerrain"
    base_cfg = registry.get_default_config(task)
    cfg = ice_env.ice_config(base_cfg)

    stages = ice_env.FRICTION_STAGES[: args.stages] if args.stages else ice_env.FRICTION_STAGES
    # Front-load: earlier stages teach the gait, later ones only adapt it.
    weights = [0.25, 0.15, 0.15, 0.15, 0.15, 0.15][: len(stages)]
    weights = [w / sum(weights) for w in weights]
    budgets = [int(args.timesteps * w) for w in weights]

    name = args.name or f"curr_{datetime.now():%H%M%S}"
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"run={name}  total={args.timesteps:,}  device={jax.devices()[0]}")
    for i, (rng_, b) in enumerate(zip(stages, budgets)):
        print(f"  stage {i}: friction={rng_}  budget={b:,}")

    history = []
    params = None
    t0 = time.time()

    net_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
    )

    for stage_i, (frict, budget) in enumerate(zip(stages, budgets)):
        env = ice_env.patch_termination(registry.load(task, config=cfg))
        eval_env = ice_env.patch_termination(registry.load(task, config=cfg))
        randomizer = ice_env.curriculum_randomizer(stage_i)

        print(f"\n=== stage {stage_i}  friction={frict}  "
              f"{budget:,} steps  ({time.time()-t0:.0f}s elapsed) ===", flush=True)

        # Save on every eval, not just at the end. A run killed mid-training
        # otherwise loses everything: base_v3 reached episode length 346 and
        # left no checkpoint behind because only the final save existed.
        latest = {"params": None}

        def _keep(step, make_fn, params, _s=stage_i):
            latest["params"] = params
            brax_model.save_params(str(out / f"latest_stage{_s}"), params)

        def progress(step, metrics, _s=stage_i, _f=frict):
            row = {
                "stage": _s,
                "friction": list(_f),
                "step": int(step),
                "elapsed_s": round(time.time() - t0, 1),
                "reward": float(metrics.get("eval/episode_reward", 0.0)),
                "episode_len": float(metrics.get("eval/avg_episode_length", 0.0)),
            }
            history.append(row)
            (out / "metrics.json").write_text(json.dumps(history, indent=1))
            print(f"  s{_s} mu={_f[0]:.2f}-{_f[1]:.2f}  {row['step']:>11,}  "
                  f"reward={row['reward']:8.2f}  len={row['episode_len']:7.1f}  "
                  f"({row['elapsed_s']:.0f}s)", flush=True)

        train_fn = functools.partial(
            ppo.train,
            num_timesteps=budget,
            num_evals=6,
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
            network_factory=net_factory,
            policy_params_fn=_keep,
            randomization_fn=randomizer,
            wrap_env_fn=wrapper.wrap_for_brax_training,
            progress_fn=progress,
            seed=0,
        )
        # Warm-start every stage after the first from the previous weights.
        if params is not None:
            train_fn = functools.partial(train_fn, restore_params=params)

        _, params, _ = train_fn(environment=env, eval_env=eval_env)

        brax_model.save_params(str(out / f"policy_stage{stage_i}"), params)
        _snapshot(eval_env, params, out, name, stage_i, frict)

    brax_model.save_params(str(out / "policy"), params)
    print(f"\ndone in {time.time()-t0:.0f}s -> {out/'policy'}")


def _snapshot(env, params, out, name, stage_i, frict):
    """Record a clip at the end of each stage."""
    try:
        import jax
        import jax.numpy as jp
        import mediapy
        from brax.training.agents.ppo import networks as nets

        net = nets.make_ppo_networks(
            env.observation_size, env.action_size,
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
        )
        inf = jax.jit(nets.make_inference_fn(net)(params, deterministic=True))
        step = jax.jit(env.step)
        st = jax.jit(env.reset)(jax.random.PRNGKey(0))
        st.info["command"] = jp.array([0.8, 0.0, 0.0])
        roll = []
        rng = jax.random.PRNGKey(1)
        for _ in range(int(8.0 / env.dt)):
            rng, k = jax.random.split(rng)
            a, _ = inf(st.obs, k)
            st = step(st, a)
            st.info["command"] = jp.array([0.8, 0.0, 0.0])
            roll.append(st)
        frames = env.render(roll, camera="track", width=640, height=440)
        vp = Path("videos") / f"{name}_stage{stage_i}_mu{frict[0]:.2f}.mp4"
        vp.parent.mkdir(parents=True, exist_ok=True)
        mediapy.write_video(str(vp), frames, fps=1.0 / env.dt)
        print(f"      video -> {vp}", flush=True)
    except Exception as e:
        print(f"      (video skipped: {type(e).__name__}: {e})", flush=True)


if __name__ == "__main__":
    main()
