"""实测碰撞判定: 让小车以固定速度撞向障碍物, 同时记录
  (1) 接触传感器 与障碍物 的接触力 (force_matrix_w)
  (2) 接触传感器方式判定的 collided
  (3) 特权信息方式判定的 collided
并跑一段"空地上直行"作为对照(噪声本底).

用法: python test_collision.py --num_envs 2
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

    print("\n========== 配置 ==========")
    print(f"collision_check_mode   = {cfg.collision_check_mode}")
    print(f"contact_force_threshold= {cfg.contact_force_threshold} N")
    print(f"contact_sensor prim    = {cfg.contact_sensor.prim_path}")
    print(f"contact_sensor filter  = {cfg.contact_sensor.filter_prim_paths_expr}")
    print(f"robot_footprint_half   = {cfg.robot_footprint_half}")
    fm = env._contact_sensor.data.force_matrix_w
    print(f"force_matrix_w shape   = {None if fm is None else tuple(fm.shape)}")

    def drive_into(start_xy, heading, target_xy, n_steps=400, v_cmd=2.0):
        """把车放到 start_xy 朝 heading, 以固定速度直行, 记录碰撞相关量"""
        quat = torch.tensor([np.cos(heading / 2), 0.0, 0.0, np.sin(heading / 2)], device=env.device)
        pose = torch.zeros(env.num_envs, 7, device=env.device)
        pose[:, :3] = torch.tensor([start_xy[0], start_xy[1], 0.178], device=env.device)
        pose[:, 3:] = quat
        env._robot.write_root_pose_to_sim(pose)
        env._robot.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))
        for _ in range(3):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)

        # 直接写根速度前进(等价于 action 常量), 绕开策略
        max_force_hist, contact_flag_hist, priv_flag_hist, d_hist, net_hist = [], [], [], [], []
        for k in range(n_steps):
            cmd = torch.zeros(env.num_envs, 6, device=env.device)
            cmd[:, 0] = v_cmd
            env._robot.write_root_velocity_to_sim(cmd)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)

            fm_now = env._contact_sensor.data.force_matrix_w
            nf_now = env._contact_sensor.data.net_forces_w
            nf_max = float(torch.norm(nf_now, dim=-1).max().item()) if nf_now is not None else -1.0
            net_hist.append(nf_max)
            fmax = float(torch.norm(fm_now, dim=-1).max().item()) if fm_now is not None else -1.0
            contact_hit = bool((torch.norm(fm_now, dim=-1).max() > cfg.contact_force_threshold).item()) if fm_now is not None else False
            priv_hit = bool(env._compute_collided_privileged()[0].item())
            max_force_hist.append(fmax)
            contact_flag_hist.append(contact_hit)
            priv_flag_hist.append(priv_hit)
            d = float(torch.linalg.norm(env._robot.data.root_pos_w[0, :2] - torch.tensor(target_xy, device=env.device)).item())
            d_hist.append(d)

        return max_force_hist, contact_flag_hist, priv_flag_hist, d_hist, net_hist

    if len(env.obstacles) == 0:
        print("没有障碍物, 请检查 difficulty")
        return

    obs_list = [(ob.data.root_pos_w[0, :2].cpu().numpy(), w) for ob, w in zip(env.obstacles, env.obstacles_width)]
    print("\n场上障碍物: " + " | ".join(f"({p[0]:+.2f},{p[1]:+.2f}) w={w:.2f}" for p, w in obs_list))

    for i, (opos, w) in enumerate(obs_list):
        for approach in (0.0, np.pi / 2):   # 正面撞 / 侧面擦碰
            heading = approach
            # 从障碍物外侧 3m 处沿 heading 方向朝障碍物开
            sx = opos[0] - 3.0 * np.cos(heading)
            sy = opos[1] - 3.0 * np.sin(heading)
            mf, cf, pf, dd, nf = drive_into((sx, sy), heading, opos)
            print(f"\n[障碍{i} w={w:.2f} 从{np.degrees(heading):.0f}度方向撞 起点({sx:+.2f},{sy:+.2f})]")
            print(f"  接触力峰值={max(mf):8.2f} N | 接触传感器触发={any(cf)} (第{cf.index(True) if any(cf) else -1}步)"
                  f" | 特权信息触发={any(pf)} (第{pf.index(True) if any(pf) else -1}步)")
            print(f"  最近距离={min(dd):.2f} m, 末距离={dd[-1]:.2f} m, 距离(每20步)={[round(x,2) for x in dd[::20]]}")
            print(f"  接触力(过滤后)非零步数={sum(1 for f in mf if f > 0.01)}/{len(mf)}, 序列(每20步)={[round(f,1) for f in mf[::20]]}")
            print(f"  未过滤net_forces_w 峰值={max(nf):.2f} N, 序列(每20步)={[round(f,1) for f in nf[::20]]}")

    # 对照组: 选一条远离所有障碍物的直线, 看本底
    free_y = max(p[1] for p, _ in obs_list) + 2.0
    free_y = min(free_y, 4.5)
    mf, cf, pf, dd, nf = drive_into((-3.5, free_y), 0.0, (-3.5, free_y))
    print(f"\n[对照: 远离障碍物直行 y={free_y:+.2f}] 接触力峰值={max(mf):.2f} N | 接触传感器误触发={any(cf)} | 特权误触发={any(pf)}")
    print()


if __name__ == "__main__":
    main()
    simulation_app.close()
