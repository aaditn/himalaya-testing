"""Hugging Face packaging for native Himalaya Brax policies."""

import json
from pathlib import Path


def write_model_card(folder: Path, curriculum: list[dict]) -> Path:
    card = f"""---
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

Signed potential progress has scale 10.0 and is measured from pelvis
displacement along the slope. It receives full credit with hand support and 30%
during hand exchange, while regression is penalized 1.75x. New-high progress
has scale 3.0 and each new 0.25 m waypoint receives a one-time 2.0 bonus.
Supported forward velocity has scale 8.0 and pays linearly up to the commanded
speed, then saturates; stationary, backward, unsupported, and collapsed motion
receive zero velocity credit.
The climb-only fall event is -20 and a failed episode returns 20% of its ascent
potential instead of cancelling it completely. This keeps falling undesirable
without erasing the early forward trials PPO needs to discover a stable crawl.
Each scheduled palm swing targets 0.24384 m (0.8 ft) terrain-normal lift from
its last plant height. Dense lift credit saturates at the target and a one-time
crossing bonus resets only after replant; the opposite palm and at least one
foot must remain planted. Opposed 60% hand duty cycles retain 20% two-hand
overlap. Generic hand/foot phase-height rewards are disabled because they
requested clearance during scheduled support; completed foot air time and the
explicitly supported palm-lift objective remain active.
Contact, load-share, diagonal-support, and foot-drive shaping is gated by
positive displacement. Episodes terminate after three seconds without a new 5
mm progress checkpoint or after losing 0.35 m from their highest point. Hand
support is therefore a means of climbing rather than a substitute for motion.

Knee grounding uses dense terrain-relative shaping. Clearance is measured from
each knee origin to the conservative height-field envelope and nearest boulder,
after subtracting a 6 cm knee-housing radius. Positive credit saturates at 5 cm
surface clearance and is weighted toward forward progress; crossing zero
clearance receives a separate contact cost. The simplified knee geometry does
not add terrain collision pairs, so this remains a differentiable policy
objective without changing the actor/critic observation dimensions.

The crawl curriculum starts at 5 degrees with 5 mm relief and boulders disabled,
then transfers through smooth and rocky 12-degree stages before 20, 28, 35,
and a terminal 42 degrees. The last stage uses 15 cm relief and 40% target hand
loading. Positive progress is gated by terrain-normal pelvis clearance and
torso alignment, and failed episodes surrender accumulated ascent credit.
Touchdown credit requires at least 80 ms airborne and a new episode-best uphill
plant location. Do not initialize this curriculum from obsolete smoke policies
that learned a forward-collapse strategy.

Large foot placements also receive a timestep-corrected quadratic event bonus.
Advance is normalized by the 20 cm target: qualifying 10, 20, and 30 cm steps
earn 0.25, 1.0, and 2.25 respectively. Only a new episode-best foot plant after
at least 80 ms airborne qualifies, preventing sliding or stamping from farming
the term. Hand touchdowns retain their separate linear shaping.

Motor targets include a bounded diagonal crawl reference with PPO actions as
residuals: right hand moves with left foot, then left hand with right foot.
The moving arm uses shoulder pitch, shoulder roll, and elbow to lift and reach,
while the planted diagonal extends to unload it. The crawl clock is randomized
in [0.70, 0.95] Hz. Each hip uses a continuous +/-0.30 rad fore-aft sweep so
the swing leg reaches while the stance leg retracts to propel the pelvis. Knee
lift is a 0.38 rad half-wave and touchdown advance uses a 20 cm target. Reset
begins at the all-planted phase boundary. Static pose/hip/knee deviation costs
are disabled for climbing; safety termination, collision cost, smoothness,
energy, and joint-limit costs remain. This prevents coordination from being a
sparse discovery problem while leaving balance, foothold adaptation, and
propulsion to the learned policy.

Episodes start from a forward-biased suspended four-point crouch. Relative to
the preceding keyframe, the torso is exactly 10 cm farther uphill while the
same palms and feet remain planted; knee-joint clearance is 18 cm and wrists
remain straight. Policies trained from the older crouch are incompatible and
the curriculum must restart from stage one.

Each stock rubber hand is replaced by a visible, rigid 5 cm-radius sphere
centered on the existing palm site. These spherical hand end-effectors remain
attached to the wrists; the palm force sensors, contact IDs, rewards, and
boulder collision pairs are unchanged.
All three wrist axes are centered at zero and limited to +/-0.08 rad, with a
position gain of 20 and +/-25 Nm actuator limits. This preserves slight
compliance while preventing the wrist from folding under support load.

Hand tangential microspike friction is randomized in [0.9, 1.0], and foot
friction is exactly twice the sampled hand value in [1.8, 2.0].
Rocky curriculum stages use 6-15 cm height-field relief and ten physical
10-inch-diameter boulders with explicit palm/foot contacts. Boulder placement
uses each compiled radius, and the four-point reset is rigidly aligned with the
configured ramp about its floor origin so steeper stages do not invalidate the
initial contacts.

## Curriculum

```json
{json.dumps(curriculum, indent=2)}
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
"""
    path = folder / "README.md"
    path.write_text(card, encoding="utf-8")
    return path


def upload(folder: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=folder,
        commit_message="Upload Himalaya G1 four-limb climb policy",
    )
