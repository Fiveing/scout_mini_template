"""加载一个 checkpoint, 用"确定性策略(不加探索噪声)"跑仿真, 看它到底会不会去目标点.

这是把"策略意图错了"和"探索噪声太大"两种情况分开的关键实验:
  - 确定性策略能到目标  -> 策略意图是对的, 训练不好是因为探索噪声/熵项太大
  - 确定性策略也乱跑    -> 状态(目标方向)对策略输出的影响力被图像分支淹没了

用法: python probe_policy.py --num_envs 8 --steps 900 --ckpt ckpts/model_latest.pth
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--ckpt", type=str, default="ckpts/model_latest.pth")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import sys
import numpy as np
import torch
import yaml

from scout_mini.tasks.direct.scout_mini.scout_mini_env_cfg import ScoutMiniEnvCfg
from scout_mini.tasks.direct.scout_mini.scout_mini_env import ScoutMiniEnv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Agent.agent import SACAgent  # noqa: E402


class DummyLogger:
    def add_scalar(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass

    def dump(self, *a, **k):
        pass


def fmt(x):
    return " ".join(f"{v:7.3f}" for v in x)


def main():
    cfg = ScoutMiniEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    env = ScoutMiniEnv(cfg=cfg)

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    sac_cfg = {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "buffer_size": 1000,
        "obs_dim": config["OBS_DIM"],
        "state_dim": config["STATE_SHAPE"],
        "gamma": config["DISCOUNT"],
        "tau": config["CRITIC_TAU"],
        "learning_rate": config["LEARNING_RATE"],
        "alpha_learning_rate": config["ALPHA_LR"],
        "batch_size": config["BATCH_SIZE"],
        "device": env.device,
        "use_crop_action": config["USE_CROP_ACTION"],
        "crop_size": config["CROP_SIZE"],
    }
    agent = SACAgent(sac_cfg["obs_dim"], 2, sac_cfg, DummyLogger())
    agent.load(args_cli.ckpt, config["USE_LOG_ALPHA"])
    print(f"[probe] loaded {args_cli.ckpt}, alpha={agent.alpha.item():.4f}")

    obs_dict, _ = env.reset()
    obs = agent.obs_to_input(obs_dict["policy"], downsample_size=config["OBS_DIM"][-2:])
    state = obs_dict["state"]
    crop = torch.zeros((env.num_envs, 2), device=env.device)

    n = env.num_envs
    ret_sum = torch.zeros(n, device=env.device)
    ep_returns = []
    reached_cnt = collided_cnt = timeout_cnt = 0
    probe_corr_real = []
    probe_corr_fixed = []

    for step in range(args_cli.steps):
        with torch.no_grad():
            mu, _, _, log_std = agent.actor(obs, state, last_crop_action=crop,
                                            compute_pi=False, compute_log_pi=False, use_crop_action=False)
            actions = mu

            # 真实图像下的 goal_dir 影响力
            gd = state[:, 4]
            probe_corr_real.append(torch.corrcoef(torch.stack([gd, actions[:, 1]]))[0, 1].item())
            # 固定图像下的 goal_dir 影响力
            fixed_obs = obs[0:1].expand(n, -1, -1, -1)
            mu_f, _, _, _ = agent.actor(fixed_obs, state, last_crop_action=crop,
                                        compute_pi=False, compute_log_pi=False, use_crop_action=False)
            probe_corr_fixed.append(torch.corrcoef(torch.stack([gd, mu_f[:, 1]]))[0, 1].item())

        new_obs_dict, rew, terminated, truncated, info = env.step(actions)
        obs = agent.obs_to_input(new_obs_dict["policy"], downsample_size=config["OBS_DIM"][-2:])
        state = new_obs_dict["state"]
        ret_sum += rew

        if step % 60 == 0:
            dist = torch.linalg.norm(env._robot.data.root_pos_w[:, :2] - env._desired_pos_w[:, :2], dim=1)
            print(f"step {step:4d} | dist {fmt(dist.tolist())} | goal_dir {fmt(state[:, 4].tolist())} | "
                  f"v {fmt(env._robot.data.root_lin_vel_b[:, 0].tolist())} | "
                  f"w {fmt(env._robot.data.root_ang_vel_b[:, 2].tolist())}")

        done = terminated | truncated
        if done.any():
            idx = done.nonzero(as_tuple=False).squeeze(-1)
            reached_cnt += int(env.reached[idx].sum().item())
            collided_cnt += int(env.collided[idx].sum().item())
            timeout_cnt += int(env.time_out[idx].sum().item())
            ep_returns.extend(ret_sum[idx].tolist())
            ret_sum[done] = 0.0

    print(f"[probe] episodes={len(ep_returns)} reached={reached_cnt} collided={collided_cnt} timeout={timeout_cnt}")
    if ep_returns:
        print(f"[probe] mean episode return (deterministic policy) = {np.mean(ep_returns):+.1f}")
    print(f"[probe] 确定性策略下 corr(goal_dir, w_mu): 真实图像 {np.mean(probe_corr_real):+.3f} | "
          f"固定图像 {np.mean(probe_corr_fixed):+.3f}")
    print(f"[probe] policy log_std = {log_std.mean().item():.3f} (std={log_std.exp().mean().item():.3f})")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
