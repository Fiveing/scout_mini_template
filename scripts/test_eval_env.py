"""验证迁移过来的两个评估环境: 场景是否搭对、动态障碍物是否在动、碰撞判定是否生效.

用法:
  python test_eval_env.py --env generated --tier 4 --num_envs 2
  python test_eval_env.py --env lobby     --tier 4 --num_envs 2
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="generated", choices=["generated", "lobby"])
parser.add_argument("--tier", type=int, default=4, choices=[4, 6, 8, 12])
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import numpy as np
import torch

from scout_mini.tasks.direct.scout_mini.scout_mini_generated_env_cfg import ScoutMiniGeneratedEnvCfg
from scout_mini.tasks.direct.scout_mini.scout_mini_generated_env import ScoutMiniGeneratedEnv
from scout_mini.tasks.direct.scout_mini.scout_mini_lobby_env_cfg import ScoutMiniLobbyEnvCfg
from scout_mini.tasks.direct.scout_mini.scout_mini_lobby_env import ScoutMiniLobbyEnv


def main():
    if args_cli.env == "generated":
        cfg = ScoutMiniGeneratedEnvCfg()
        env_cls = ScoutMiniGeneratedEnv
    else:
        cfg = ScoutMiniLobbyEnvCfg()
        env_cls = ScoutMiniLobbyEnv
    cfg.dynamic_obstacle_tier = args_cli.tier
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    env = env_cls(cfg=cfg)
    obs, _ = env.reset()

    print(f"\n===== {args_cli.env} tier={args_cli.tier} =====")
    print(f"静态: AABB {env._extra_static_aabb[0].shape[0]} 个, 圆 {env._extra_static_circle[0].shape[0]} 个")
    print(f"动态障碍物: {len(env.dyn_obs_list)} 个 | 固定起终点 {env._pair_count} 组")
    print(f"起终点前 3 组: {[tuple(round(v,2) for v in env._fixed_start_positions[i,:2].tolist()) for i in range(3)]}"
          f" -> {[tuple(round(v,2) for v in env._fixed_target_positions[i,:2].tolist()) for i in range(3)]}")
    d0 = torch.linalg.norm(env._robot.data.root_pos_w[:, :2] - env._desired_pos_w[:, :2], dim=1)
    print(f"reset 后: 车 {env._robot.data.root_pos_w[0,:2].tolist()} 目标 {env._desired_pos_w[0,:2].tolist()} 距离 {d0[0]:.2f}m")

    # 动态障碍物是否在动
    p0 = env.dyn_obs_pos[:, :2].clone()
    for _ in range(50):
        env.step(torch.zeros((env.num_envs, 2), device=env.device))
    p1 = env.dyn_obs_pos[:, :2].clone()
    moved = torch.linalg.norm(p1 - p0, dim=1)
    print(f"动态障碍物 50 步位移(应≈速度*1s): {[round(v,3) for v in moved.tolist()]}")

    # 碰撞判定: 把车瞬移到第一个障碍物/墙上, 看 collided 是否触发
    targets = env.dyn_obs_pos[:, :2] * 0 + env.dyn_obs_pos[:, :2]
    probe = env.dyn_obs_pos[0, :2].clone()
    pose = torch.zeros(env.num_envs, 7, device=env.device)
    pose[:, :3] = torch.tensor([probe[0].item(), probe[1].item(), 0.178], device=env.device)
    pose[:, 3] = 1.0
    env._robot.write_root_pose_to_sim(pose)
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    print(f"瞬移到动态障碍物中心 -> collided={env._compute_collided_privileged()[0].item()} (应为 True)")
    coll = env._compute_collided_privileged()
    print(f"  全部 env collided = {coll.tolist()}")

    # 跑一段脚本专家(朝目标转+前进), 看成功率
    print("\n--- 脚本专家跑 300 步 ---")
    reached_cnt = 0
    for step in range(300):
        pos_b = env._robot.data.root_pos_w - env._desired_pos_w
        # 用世界的相对位置算朝目标的方向(与体坐标系差一个 yaw)
        yaw = env._robot.data.heading_w
        dx, dy = pos_b[:, 0], pos_b[:, 1]
        goal_dir = torch.atan2(dy, dx) - yaw
        goal_dir = torch.atan2(torch.sin(goal_dir), torch.cos(goal_dir))
        ang = torch.clamp(goal_dir * 1.5, -1.0, 1.0)
        lin = torch.where(goal_dir.abs() < 0.5, torch.full_like(goal_dir, 1.5), torch.full_like(goal_dir, 0.6))
        act = torch.stack([lin - 1.0, ang], dim=-1)
        _, _, terminated, truncated, info = env.step(act)
        reached_cnt += info["log"]["Episode_Termination/reached"]
    print(f"专家 300 步内到达次数 = {reached_cnt} (仅供参考: 未考虑避障, 会撞障碍物)")
    print(f"collided 累计 = {info['log']['Episode_Termination/collided']}, time_out = {info['log']['Episode_Termination/time_out']}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
