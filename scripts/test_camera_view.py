"""把相机画面存成 PNG, 人工确认障碍物到底有没有被渲染出来.

用法: python test_camera_view.py --num_envs 2
输出: /tmp/cam_A_obstacle.png (障碍物在原位) 和 /tmp/cam_B_moved.png (障碍物搬到 100m 外)
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import cv2
import numpy as np
import torch

from scout_mini.tasks.direct.scout_mini.scout_mini_env_cfg import ScoutMiniEnvCfg
from scout_mini.tasks.direct.scout_mini.scout_mini_env import ScoutMiniEnv


def save_rgb(env, path, tag):
    rgb = env._camera.data.output["rgb"][0].cpu().numpy()  # (H,W,4) rgba? or (H,W,3)
    rgb = rgb[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
    # 统计"绿色"像素(障碍物材质 diffuse=(0,1,0))
    g = rgb[:, :, 1].astype(np.int32)
    r = rgb[:, :, 0].astype(np.int32)
    b = rgb[:, :, 2].astype(np.int32)
    green_mask = (g > 100) & (g > r + 40) & (g > b + 40)
    print(f"[{tag}] 画面已存 {path} | 绿色像素={int(green_mask.sum())} | "
          f"RGB均值=({rgb[:,:,0].mean():.0f},{rgb[:,:,1].mean():.0f},{rgb[:,:,2].mean():.0f})")


def main():
    cfg = ScoutMiniEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    env = ScoutMiniEnv(cfg=cfg)
    env.reset()

    for i, obs in enumerate(env.obstacles):
        print(f"obstacle {i}: pos={[round(v,3) for v in obs.data.root_pos_w[0,:3].tolist()]}")

    def settle(robot_xy, heading, n=30):
        quat = torch.tensor([np.cos(heading / 2), 0.0, 0.0, np.sin(heading / 2)], device=env.device)
        pose = torch.zeros(1, 7, device=env.device)
        pose[0, :3] = torch.tensor([robot_xy[0], robot_xy[1], 0.178], device=env.device)
        pose[0, 3:] = quat
        env._robot.write_root_pose_to_sim(pose)
        env._robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
        for k in range(n):
            env.sim.step(render=False)
            if k % 2 == 0:
                env.sim.render()
            env.scene.update(dt=env.physics_dt)

    if len(env.obstacles) == 0:
        print("没有障碍物")
        return
    obs0 = env.obstacles[0]
    opos = obs0.data.root_pos_w[0, :2].cpu().numpy()
    settle((opos[0] - 2.0, opos[1]), 0.0)
    save_rgb(env, "/tmp/cam_A_obstacle.png", "A 障碍物在原位")

    # 搬到 100m 外
    pose = torch.zeros(1, 7, device=env.device)
    pose[0, :3] = torch.tensor([100.0, 100.0, 0.75], device=env.device)
    pose[0, 3] = 1.0
    obs0.write_root_pose_to_sim(pose)
    obs0.write_data_to_sim()
    obs0.update(env.physics_dt)
    print(f"搬走后 obstacle0 报告位置 = {obs0.data.root_pos_w[0,:3].tolist()}")
    settle((opos[0] - 2.0, opos[1]), 0.0)
    save_rgb(env, "/tmp/cam_B_moved.png", "B 障碍物搬到100m外")
    print()


if __name__ == "__main__":
    main()
    simulation_app.close()
