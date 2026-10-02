"""检查 RayCaster 到底能不能打到静态障碍物.

做法: 把小车瞬移到障碍物正前方 1.0/2.0/3.0/4.5 m 处并朝向它, 步进若干步后读 ray_hits_w,
比较"打到障碍物的距离"和"打不到(等于量程)"两种情况.

用法: python test_lidar.py --num_envs 2
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

import numpy as np
import torch

from scout_mini.tasks.direct.scout_mini.scout_mini_env_cfg import ScoutMiniEnvCfg
from scout_mini.tasks.direct.scout_mini.scout_mini_env import ScoutMiniEnv


def main():
    cfg = ScoutMiniEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    env = ScoutMiniEnv(cfg=cfg)
    env.reset()

    print("\n================ 传感器配置 ================")
    print(f"mesh_prim_paths       : {cfg.ray_caster.mesh_prim_paths}")
    print(f"pattern               : {cfg.ray_caster.pattern_cfg.__class__.__name__} "
          f"res={cfg.ray_caster.pattern_cfg.horizontal_res} fov={cfg.ray_caster.pattern_cfg.horizontal_fov_range}")
    print(f"ray_caster_range      : {cfg.ray_caster_range}")
    print(f"robot_radius(碰撞阈值) : {cfg.robot_radius}")

    print("\n================ 障碍物 ================")
    if len(env.obstacles) == 0:
        print("本次运行场景里没有障碍物 (cfg.num_obstacles = 0), 请把 difficulty 调成 easy/medium/hard")
    for i, (obs, w) in enumerate(zip(env.obstacles, env.obstacles_width)):
        pos = obs.data.root_pos_w[0, :3].tolist()
        print(f"obstacle {i}: prim=/World/Obstacles/Box{i} pos={[round(v, 3) for v in pos]} width={w:.3f}")

    lidar = env._lidar_sensor
    print(f"\nlidar prim 数 / 射线起始偏移 z = {cfg.ray_caster.offset.pos}")

    def probe(robot_xy, heading_rad, tag, n_steps=6):
        """把 env0 的小车放到指定位置/朝向后, 读雷达命中距离"""
        quat = torch.tensor([np.cos(heading_rad / 2), 0.0, 0.0, np.sin(heading_rad / 2)], device=env.device)
        pose = torch.zeros(1, 7, device=env.device)
        pose[0, :3] = torch.tensor([robot_xy[0], robot_xy[1], 0.178], device=env.device)
        pose[0, 3:] = quat
        env._robot.write_root_pose_to_sim(pose)
        env._robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
        for _ in range(n_steps):
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        hits = lidar.data.ray_hits_w[0]                        # (N_rays, 3)
        origin = env._robot.data.root_pos_w[0, :3]
        dist = torch.linalg.norm(hits - origin, dim=-1)
        n_hit_real = int((dist < cfg.ray_caster_range - 1e-3).sum().item())
        # 同时看深度相机(策略真正用的传感器)能不能看到障碍物
        depth = env._camera.data.output["distance_to_image_plane"][0, :, :, 0]  # (H, W) 单位 m
        h, w = depth.shape
        center = depth[h // 4: 3 * h // 4, w // 3: 2 * w // 3]
        n_close = int((center < cfg.depth_image_range).sum().item())
        print(f"[{tag}] | 雷达: 命中射线={n_hit_real}/{len(dist)} 最近={dist.min().item():.1f}m "
              f"中位={dist.median().item():.0f}m collided={bool(env.collided[0].item())} "
              f"|| 深度图中央区: 最近={center.min().item():.2f}m 中位={center.median().item():.2f}m "
              f"近距(<5m)像素={n_close}/{center.numel()}")
        return dist

    print("\n================ 正前方打障碍物测试 ================")
    for i, (obs, w) in enumerate(zip(env.obstacles, env.obstacles_width)):
        opos = obs.data.root_pos_w[0, :2].cpu().numpy()
        for d in (1.0, 2.0, 3.0, 4.5):
            # 从障碍物出发沿 -x 方向退 d 米, 朝 +x (heading=0)
            probe((opos[0] - d, opos[1]), 0.0, f"障碍{i}正前方{d}m")

    print("\n================ 对照: 空地上(远离障碍物) ================")
    probe((0.0, -3.9), 0.0, "空地")
    print()


if __name__ == "__main__":
    main()
    simulation_app.close()
