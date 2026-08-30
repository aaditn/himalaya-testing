---
library_name: brax
tags:
- mujoco
- mjx
- reinforcement-learning
- unitree-g1
- robotics
license: apache-2.0
---

# Unitree G1 continuous four-limb climbing policy

This is a Brax PPO policy trained with the `aaditn/himalaya-testing` platform
and its vendored MuJoCo Playground G1 environment. The task rewards continuous
palm support, diagonal hand-foot support, uphill progress under hand support,
signed mountain progress, uphill foot drive, target hand load share, and low
hand slip on rough inclined terrain.

Forward/ascent rewards have a combined scale of 15.0, versus 2.8 for hand
contact, diagonal support, and load-share shaping. Forward velocity tracking is
gated by positive uphill speed, and downhill displacement receives a negative
signed-progress reward. Hand support is therefore a means of climbing rather
than a substitute for forward motion.

Episodes start from a suspended four-point crouch with palms and feet planted
and the knees, shins, pelvis, and torso clear of the terrain.

Both tangential microspike coefficients are randomized together in [0.9, 1.0].
The terrain curriculum uses 6-12 cm height-field relief and ten physical
6-inch-diameter boulders with explicit palm/foot contacts.

## Curriculum

```json
[
  {
    "name": "crawl-foundation-12deg",
    "slope_degrees": 12,
    "roughness_m": 0.06,
    "spike_friction": 0.95,
    "target_uphill_speed": 0.4,
    "target_hand_load_share": 0.2,
    "num_timesteps": 30000000,
    "num_evals": 6,
    "terrain_seed": 11
  },
  {
    "name": "four-limb-20deg",
    "slope_degrees": 20,
    "roughness_m": 0.08,
    "spike_friction": 0.95,
    "target_uphill_speed": 0.35,
    "target_hand_load_share": 0.25,
    "num_timesteps": 50000000,
    "num_evals": 8,
    "terrain_seed": 22
  },
  {
    "name": "diagonal-crawl-28deg",
    "slope_degrees": 28,
    "roughness_m": 0.1,
    "spike_friction": 0.95,
    "target_uphill_speed": 0.3,
    "target_hand_load_share": 0.3,
    "num_timesteps": 80000000,
    "num_evals": 10,
    "terrain_seed": 33
  },
  {
    "name": "steep-crawl-35deg",
    "slope_degrees": 35,
    "roughness_m": 0.12,
    "spike_friction": 0.95,
    "target_uphill_speed": 0.25,
    "target_hand_load_share": 0.35,
    "num_timesteps": 120000000,
    "num_evals": 12,
    "terrain_seed": 44
  }
]
```

## Reproduction

```bash
python scripts/train_climb_curriculum.py --curriculum configs/curriculum.json
python scripts/record.py runs/<stage>/policy --climb --out videos/eval.mp4
```

The microspikes are a macro contact-friction model, not resolved spike teeth.
Short smoke runs validate the pipeline only; they do not demonstrate Class 2
climbing competence. This policy is a simulation research artifact and must not be deployed to
hardware without system identification, wrist/actuator load limits, fall
arrest, supervised emergency stop, and progressive physical validation.
