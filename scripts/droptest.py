"""Does the 23-DOF robot stand on FLAT ground with zero action?

Runs 3/4/5 all failed with the robot falling. Before tuning any more rewards,
establish whether the asset itself is sane: spawn it on flat ground, command
nothing, and see if it stays up. If it falls here, the problem is the URDF
conversion / spawn height / collision setup, not the reward function.
"""
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import torch  # noqa: E402
from isaaclab.sim import SimulationContext, SimulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.himalaya_env_cfg import (  # noqa: E402
    G1_23DOF_CFG_V2,
)


@configclass
class FlatSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg())
    robot = G1_23DOF_CFG_V2.replace(prim_path="{ENV_REGEX_NS}/Robot")


sim = SimulationContext(SimulationCfg(dt=1 / 200, device="cuda:0"))
scene = InteractiveScene(FlatSceneCfg(num_envs=4, env_spacing=3.0))
sim.reset()

robot = scene["robot"]
print("DOF_COUNT:", robot.num_joints)
print("JOINT_NAMES:", robot.joint_names)
print("BODY_COUNT:", robot.num_bodies)

hold = robot.data.default_joint_pos.clone()
for step in range(400):  # 2 s at 200 Hz
    robot.set_joint_position_target(hold)
    scene.write_data_to_sim()
    sim.step()
    scene.update(1 / 200)
    if step % 100 == 0:
        h = robot.data.root_pos_w[:, 2]
        print(f"STEP {step:3d}  height_mean={h.mean().item():.3f} min={h.min().item():.3f}")

h = robot.data.root_pos_w[:, 2]
print(f"FINAL_HEIGHT: {h.mean().item():.3f}  (spawned at 0.79)")
print("VERDICT:", "STANDS" if h.mean().item() > 0.6 else "FELL")
app.close()
