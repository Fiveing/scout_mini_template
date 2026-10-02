"""把两个评估环境的相机画面存成 PNG, 确认墙/障碍物真的被渲染出来(策略靠深度图避障).

用法: python check_scene_visual.py --tier 4
输出: /tmp/scene_{generated,lobby}.png
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--tier", type=int, default=4)
parser.add_argument("--which", type=str, default="both", choices=["both", "generated", "lobby"])
parser.add_argument("--pose_idx", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=1)
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

from scout_mini.tasks.direct.scout_mini.scout_mini_generated_env_cfg import ScoutMiniGeneratedEnvCfg
from scout_mini.tasks.direct.scout_mini.scout_mini_generated_env import ScoutMiniGeneratedEnv
from scout_mini.tasks.direct.scout_mini.scout_mini_lobby_env_cfg import ScoutMiniLobbyEnvCfg
from scout_mini.tasks.direct.scout_mini.scout_mini_lobby_env import ScoutMiniLobbyEnv


def run(name, cfg, cls, out_path):
    cfg.dynamic_obstacle_tier = args_cli.tier
    cfg.scene.num_envs = args_cli.num_envs
    env = cls(cfg=cfg)
    env.reset()
    env.eval = True
    for _ in range(args_cli.pose_idx):
        env._reset_idx(None)
    # 放到第 0 组起点, 朝目标方向(便于看清障碍物)
    for _ in range(30):
        env.sim.step(render=False)
        env.sim.render()
        env.scene.update(dt=env.physics_dt)
    rgb = env._camera.data.output["rgb"][0, :, :, :3].clone().cpu().numpy()
    depth = env._camera.data.output["distance_to_image_plane"][0, :, :, 0].clone().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    h, w = rgb.shape[:2]
    center = depth[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    finite = np.isfinite(center)
    print(f"[{name}] 车={env._robot.data.root_pos_w[0,:2].tolist()} 目标={env._desired_pos_w[0,:2].tolist()}")
    print(f"[{name}] 深度图中央区: 有效像素 {int(finite.sum())}/{center.size}, 最近={center[finite].min():.2f}m, "
          f"中位={np.median(center[finite]):.2f}m")
    cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"[{name}] 画面已存 {out_path}")
    env.close()


def main():
    if args_cli.which in ("both", "generated"):
        run("generated", ScoutMiniGeneratedEnvCfg(), ScoutMiniGeneratedEnv, "/tmp/scene_generated.png")
    if args_cli.which in ("both", "lobby"):
        run("lobby", ScoutMiniLobbyEnvCfg(), ScoutMiniLobbyEnv, "/tmp/scene_lobby.png")


if __name__ == "__main__":
    main()
    simulation_app.close()
