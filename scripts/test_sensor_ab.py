"""A/B 对照测试: 雷达和深度相机到底能不能感知到静态障碍物.

做法: 把小车瞬移到障碍物正前方, 步进足够多步让相机刷新,
然后 (A) 记录雷达命中/深度图, (B) 把障碍物搬到 100m 外再记录一次.
A/B 的差异 = 传感器是否真的看到了障碍物.

用法: python test_sensor_ab.py --num_envs 2
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

FAR = torch.tensor([100.0, 100.0, 0.75])


def main():
    cfg = ScoutMiniEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    env = ScoutMiniEnv(cfg=cfg)
    env.reset()

    print("\n========== 传感器配置 ==========")
    print(f"ray_caster.mesh_prim_paths = {cfg.ray_caster.mesh_prim_paths}")
    print(f"ray_caster_range={cfg.ray_caster_range}  robot_radius={cfg.robot_radius}  "
          f"相机 update_period={cfg.camera.update_period}s")
    for i, (obs, w) in enumerate(zip(env.obstacles, env.obstacles_width)):
        print(f"obstacle {i}: {obs.cfg.prim_path} pos={[round(v,3) for v in obs.data.root_pos_w[0,:3].tolist()]} width={w:.3f}")
    if len(env.obstacles) == 0:
        print("!! 场景里没有障碍物, 请确认 difficulty 设置")

    def settle(robot_xy, heading, n=40):
        quat = torch.tensor([np.cos(heading / 2), 0.0, 0.0, np.sin(heading / 2)], device=env.device)
        pose = torch.zeros(1, 7, device=env.device)
        pose[0, :3] = torch.tensor([robot_xy[0], robot_xy[1], 0.178], device=env.device)
        pose[0, 3:] = quat
        env._robot.write_root_pose_to_sim(pose)
        env._robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
        for k in range(n):
            env.sim.step(render=False)
            # 必须真的跑渲染管线, 否则相机/深度图不会刷新(训练时 render_interval=decimation 会触发)
            if k % 2 == 0:
                env.sim.render()
            env.scene.update(dt=env.physics_dt)

    def read_sensors():
        # 注意: sensor.data.output 返回的是内部 buffer 的视图, 必须 clone 才能拿到快照,
        # 否则 A/B 两次读到的其实是同一个张量(差恒为 0).
        hits = env._lidar_sensor.data.ray_hits_w[0].clone()
        origin = env._robot.data.root_pos_w[0, :3].clone()
        dist = torch.linalg.norm(hits - origin, dim=-1)
        depth = env._camera.data.output["distance_to_image_plane"][0, :, :, 0].clone()
        return dist, depth

    def move_obstacle(idx, pos):
        for j, obs in enumerate(env.obstacles):
            p = pos.clone() if j == idx else obs.data.root_pos_w[0, :3].clone()
            pose = torch.zeros(1, 7, device=env.device)
            pose[0, :3] = p
            pose[0, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
            obs.write_root_pose_to_sim(pose)
            obs.write_data_to_sim()
            obs.update(env.physics_dt)

    for i, obs in enumerate(env.obstacles):
        opos = obs.data.root_pos_w[0, :2].cpu().numpy()
        for d in (1.5, 3.0):
            test_xy = (opos[0] - d, opos[1])
            # ---- A: 障碍物在原位 ----
            settle(test_xy, 0.0)
            dist_a, depth_a = read_sensors()
            # ---- B: 把该障碍物搬到 100m 外 ----
            move_obstacle(i, FAR.to(env.device))
            settle(test_xy, 0.0)
            dist_b, depth_b = read_sensors()
            # 搬回来
            move_obstacle(i, torch.tensor([opos[0], opos[1], 0.75], device=env.device))

            lidar_delta = (dist_a - dist_b).abs().max().item()
            d_diff = (depth_a - depth_b).abs()
            finite = torch.isfinite(depth_a) & torch.isfinite(depth_b)
            n_changed = int((d_diff[finite] > 0.1).sum().item())
            print(f"\n[障碍{i} 正前方 {d}m] 车在 ({test_xy[0]:+.2f},{test_xy[1]:+.2f})")
            print(f"  雷达: 最近命中 A={dist_a.min().item():.1f}m B={dist_b.min().item():.1f}m "
                  f"最大差={lidar_delta:.3f}m -> {'能看到' if lidar_delta > 0.05 else '看不到(无变化)'}")
            print(f"  深度图: 有障碍时 中央区最近={depth_a[240-60:240+60, 320-80:320+80].min().item():.2f}m "
                  f"| A/B 差异>0.1m 的像素数={n_changed} (总有效像素 {int(finite.sum().item())}) "
                  f"-> {'能看到' if n_changed > 50 else '看不到(无变化)'}")
    print()


if __name__ == "__main__":
    main()
    simulation_app.close()
