# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32

"""
 
"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument("--ckpt_steps", type=int, default=0, help="Checkpoint steps to load.")
parser.add_argument("--eval", action="store_true", help="Run evaluation mode.")
parser.add_argument("--env", type=str, default="base", choices=["base", "generated", "lobby"],
                    help="训练/评估用哪个环境: base=原环境(8个动态障碍物), generated/lobby=迁移过来的评估环境")
parser.add_argument("--tier", type=int, default=4, choices=[4, 6, 8, 12], help="评估环境的动态障碍物档位")
parser.add_argument("--max_steps", type=int, default=10000000, help="本次训练的最大环境步数(课程学习每个阶段用来限时)")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg

from scout_mini.tasks.direct.scout_mini.scout_mini_env_cfg import ScoutMiniEnvCfg
from scout_mini.tasks.direct.scout_mini.scout_mini_env import ScoutMiniEnv
import sys
sys.path.append('scripts')
from Agent.agent import SACAgent, ReplayBuffer
from logger.logger import *
from debugger.traj_visualization import *
import torch.autograd as autograd
import yaml
import os
import time
import cv2
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True  # 卷积尺寸固定, 让 cudnn 自动选最快算法

def _to_uint8_rgb(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().numpy()
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        if np.nanmax(image) <= 1.0:
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255.0).astype(np.uint8)
        else:
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return image


def _to_uint8_depth(image: torch.Tensor) -> np.ndarray:
    image = image.clamp(0.0, 1.0)
    image = image.detach().cpu().numpy()
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3:
        image = image[0]
    image = np.asarray(image, dtype=np.float32)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = image.astype(np.uint8)
    return cv2.applyColorMap(image, cv2.COLORMAP_JET)


def draw_crop_anchor(rgb_image: np.ndarray, depth_image: np.ndarray, crop_midpoint, crop_size=(30, 40)):
    half_h = crop_size[0] // 2
    half_w = crop_size[1] // 2
    y = int(crop_midpoint[0])
    x = int(crop_midpoint[1])
    top_left = (max(x - half_w, 0), max(y - half_h, 0))
    bottom_right = (min(x + half_w, rgb_image.shape[1] - 1), min(y + half_h, rgb_image.shape[0] - 1))
    rgb_box = rgb_image.copy()
    depth_box = depth_image.copy()
    cv2.rectangle(rgb_box, top_left, bottom_right, color=(0, 0, 255), thickness=2)
    cv2.rectangle(depth_box, top_left, bottom_right, color=(255, 255, 255), thickness=2)
    cv2.putText(rgb_box, "RGB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(depth_box, "Depth", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return rgb_box, depth_box


def test():
    """test function."""
    # create environment configuration
    env_cfg = ScoutMiniEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.action_space = 2  # 修改动作空间维度为2
    # setup RL environment
    env = ScoutMiniEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            # if count % 300 == 0:
            #     count = 0
            #     env.reset()
            #     if count > 0:
            #         print(obs, rew, terminated, truncated, info)
            #     print("-" * 80)
            #     print("[INFO]: Resetting environment...")
            # sample random actions
            # actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # actions = 2 * torch.zeros(env.action_space.shape, device=env.unwrapped.device) - 1

            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[:, 0] = 0.5
            actions[:, 1] = -1.0
            if count % 100 == 0:
                print(f"lin vel = {env._robot.data.root_link_lin_vel_b[:, 0].mean().item()}, ang vel = {env._robot.data.root_link_ang_vel_b[:, 2].mean().item()}")
            # yaw_rate = env._robot.data.root_link_ang_vel_b[:, 2].mean().item()
            # if yaw_rate > 0.1:
            #     print(f"yaw_rate: {env._robot.data.root_link_ang_vel_b[:, 2].mean().item()}")
            # if count < 1000:
            #     actions[:, 0] = 3.0
            # actions[:, 2] = -2.0  # z velocity
            # actions[:, 0] = 0.5  # forward

            # step the environment
            obs, rew, terminated, truncated, info = env.step(actions)
            # pos = obs["pos"]
            # print(pos)
            # print current orientation of pole
            # print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            if truncated.any() or terminated.any():
            #     print(obs, rew, terminated, truncated, info)
                print(f'reach {info["log"]["Episode_Termination/reached"]}, collided {info["log"]["Episode_Termination/collided"]}')
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                env.reset()
            # update counter
            count += 1

    # close the environment
    env.close()


def eval(total_eval_episodes=100):
    """eval function."""
    # create environment configuration
    env_cfg = ScoutMiniEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    env = ScoutMiniEnv(cfg=env_cfg)

    work_path = os.path.dirname(os.path.abspath(__file__))
    yaml_path = work_path + "/config.yaml"
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    ##### Parameters #####
    save_path = work_path + "/" + config['CKPT_SAVE_PATH']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logger = Logger(
        logger_name="eval", 
        log_dir= work_path + "/" + config['LOGGER_DIR'] + "/" + "eval_" + str(config['SEED'])
    )

    
    

    # RL initialization
    action_dim = env.action_space.shape[-1]
    sac_cfg = {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "buffer_size": config['CAPACITY'],
        "obs_dim": config['OBS_DIM'],
        "state_dim": config['STATE_SHAPE'],
        "gamma": config['DISCOUNT'],
        "tau": config['CRITIC_TAU'],
        "learning_rate": config['LEARNING_RATE'],
        "alpha_learning_rate": config['ALPHA_LR'],
        "batch_size": config['BATCH_SIZE'],
        "device": env.device,
        "use_crop_action": config['USE_CROP_ACTION'],
        "crop_size": config['CROP_SIZE'],
        "init_log_alpha": config.get('INIT_LOG_ALPHA', 0.0),
        "log_alpha_max": config.get('LOG_ALPHA_MAX', None),
        "log_std_min": config.get('LOG_STD_MIN', -20),
        "log_std_max": config.get('LOG_STD_MAX', 2),
    }
    agent = SACAgent(sac_cfg["obs_dim"], action_dim, sac_cfg, logger) 
    # 自动保存配置
    checkpoint_dir = work_path + "/" + config['CKPT_SAVE_PATH'] #"logs/checkpoints" # 保存目录
    # checkpoint_dir = "logs/model_backup"
    if config['PRE_MODLE'] and parser.parse_args().ckpt_steps == 0:
        if os.path.exists(os.path.join(checkpoint_dir, "model_latest.pth")):
            agent.load(os.path.join(checkpoint_dir, "model_latest.pth"))
            logger.info("已加载最新检查点，开始验证...")
    if parser.parse_args().ckpt_steps > 0:
        ckpt_steps = parser.parse_args().ckpt_steps
        ckpt_path = os.path.join(checkpoint_dir, f"model_{ckpt_steps}.pth")
        agent.load(ckpt_path, config['USE_LOG_ALPHA'])
        logger.info(f"已加载指定检查点 {ckpt_steps}，开始验证...")
    
    current_episode = 0
    avg_episode_reward = 0.0
    H, W = env_cfg.camera.height, env_cfg.camera.width

    # 首次重置，获取初始观测值
    initial_obs_dict, _ = env.reset()
    logger.info("Reset Env...")
    goal = env._desired_pos_w[0, :2].cpu().numpy().tolist()
    obstacles = [(obstacle.data.root_link_pos_w[0, 0].cpu().numpy().tolist(), obstacle.data.root_link_pos_w[0, 1].cpu().numpy().tolist(), width) for obstacle, width in zip(env.obstacles, env.obstacles_width)]
    robot_start = env._robot.data.root_link_pos_w[0, :2].cpu().numpy().tolist()
    theta_start = env._robot.data.heading_w[0].cpu().numpy().tolist()
    # 创建可视化对象，并指定初始位置和朝向
    viz = TrajectoryVisualizer(obstacles, goal,
                                    robot_start=robot_start, theta_start=theta_start)
    # current_obs_tensor = initial_obs_dict["policy"] # tensor[env_num, obs_dim]
    obs, state = initial_obs_dict["policy"], initial_obs_dict["state"] # tensor[env_num, obs_dim]
    obs = agent.obs_to_input(obs, downsample_size=config["OBS_DIM"][-2:])  # 转换为输入格式
    last_crop_action = torch.zeros((env_cfg.scene.num_envs, 2), device=env.device)
    while simulation_app.is_running() and current_episode < total_eval_episodes:
      episode_reward = 0.0
      with autograd.set_detect_anomaly(False):  # 原来是 True, 调试用的, 会显著拖慢反传
        actions, crop_actions, _, _ = agent.act(obs, state, last_crop_action=last_crop_action, deterministic=True)
        # print(f"actions: {actions}")
        new_obs_dict, rew, terminated, truncated, info = env.step(actions)
        x, y = env._robot.data.root_link_pos_w[0, :2].cpu().numpy().tolist()
        theta = env._robot.data.heading_w[0].cpu().numpy().tolist()
        viz.update_robot(x, y, theta)
        plt.pause(0.02)

        # 迭代更新
        obs, state = new_obs_dict["policy"], new_obs_dict["state"] # tensor[env_num, obs_dim]
        last_crop_action = crop_actions if config['USE_CROP_ACTION'] else torch.zeros((env_cfg.scene.num_envs, 2), device=env.device)
        crop_midpoint = [((crop_actions[:, 0] + 1) / 2 * (H - 1)).to(torch.int), ((crop_actions[:, 1] + 1) / 2 * (W - 1)).to(torch.int)] if config['USE_CROP_ACTION'] else None

        obs = agent.obs_to_input(obs, downsample_size=config["OBS_DIM"][-2:], crop_midpoint=crop_midpoint)  # 转换为输入格式
        if (terminated.any() or truncated.any()):
            goal = env._desired_pos_w[0, :2].cpu().numpy().tolist()
            viz.update_goal(goal)
            viz.clear_trajectory()
            # 更新回合计数
            end_episode_num = torch.count_nonzero(terminated).item() + torch.count_nonzero(truncated).item()
            current_episode += end_episode_num
            for key in env._episode_sums.keys():
                episode_reward += info['log']["Episode_Reward/" + key]

            logger.add_scalar('eval/episode_reward', episode_reward, 0)
            avg_episode_reward = (current_episode * avg_episode_reward + episode_reward) / (current_episode + end_episode_num)
            logger.dump(0)
            logger.info(f"episode: {current_episode}, step: {0}, avg_episode_reward: {round(avg_episode_reward.item(), 4)}, success_rate: {info['log']['Metrics/success_rate']:.3f}, final_distance_to_goal: {info['log']['Metrics/final_distance_to_goal']:.3f}")
            logger.info(f'reach {info["log"]["Episode_Termination/reached"]}, collided {info["log"]["Episode_Termination/collided"]}. time_out {info["log"]["Episode_Termination/time_out"]}')
    # close the environment
    logger.info(f"eval end. step: {0}, avg_episode_reward: {round(avg_episode_reward.item(), 4)}, success_rate: {info['log']['Metrics/success_rate']:.3f}, final_distance_to_goal: {info['log']['Metrics/final_distance_to_goal']:.3f}")
    logger.info(f'reach {info["log"]["Episode_Termination/reached"]}, collided {info["log"]["Episode_Termination/collided"]}')
    env.close()
    viz.close()

def eval_viz(total_eval_episodes=100):
    """eval function."""
    # create environment configuration
    env_cfg = ScoutMiniEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    env = ScoutMiniEnv(cfg=env_cfg)

    work_path = os.path.dirname(os.path.abspath(__file__))
    yaml_path = work_path + "/config.yaml"
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    ##### Parameters #####
    save_path = work_path + "/" + config['CKPT_SAVE_PATH']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logger = Logger(
        logger_name="eval", 
        log_dir= work_path + "/" + config['LOGGER_DIR'] + "/" + "train_" + str(config['SEED'])
    )
    
    # RL initialization
    action_dim = env.action_space.shape[-1]
    sac_cfg = {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "buffer_size": config['CAPACITY'],
        "obs_dim": config['OBS_DIM'],
        "state_dim": config['STATE_SHAPE'],
        "gamma": config['DISCOUNT'],
        "tau": config['CRITIC_TAU'],
        "learning_rate": config['LEARNING_RATE'],
        "alpha_learning_rate": config['ALPHA_LR'],
        "batch_size": config['BATCH_SIZE'],
        "device": env.device,
        "use_crop_action": config['USE_CROP_ACTION'],
    }
    agent = SACAgent(sac_cfg["obs_dim"], action_dim, sac_cfg, logger) 
    # 自动保存配置
    checkpoint_dir = work_path + "/" + config['CKPT_SAVE_PATH'] #"logs/checkpoints" # 保存目录
    # checkpoint_dir = "logs/model_backup"
    if config['PRE_MODLE']:
        if os.path.exists(os.path.join(checkpoint_dir, "model_latest.pth")):
            agent.load(os.path.join(checkpoint_dir, "model_latest.pth"))
            logger.info("已加载最新检查点，开始验证...")

    
    current_episode = 0
    avg_episode_reward = 0.0
    H, W = env_cfg.camera.height, env_cfg.camera.width
    viz_dir = os.path.join(work_path, "eval_viz")
    rgb_dir = os.path.join(viz_dir, "rgb")
    depth_dir = os.path.join(viz_dir, "depth")
    episode_video_dir = os.path.join(viz_dir, "episode_videos")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(episode_video_dir, exist_ok=True)
    frame_index = 0
    episode_frames = []
    episode_id = 0

    # 首次重置，获取初始观测值
    initial_obs_dict, _ = env.reset()
    # current_obs_tensor = initial_obs_dict["policy"] # tensor[env_num, obs_dim]
    obs, state = initial_obs_dict["policy"], initial_obs_dict["state"]# tensor[env_num, obs_dim]
    obs = agent.obs_to_input(obs, downsample_size=config["OBS_DIM"][-2:])  # 转换为输入格式
    last_crop_action = torch.zeros((env_cfg.scene.num_envs, 2), device=env.device)
    while simulation_app.is_running() and current_episode < total_eval_episodes:
      episode_reward = 0.0
      with autograd.set_detect_anomaly(False):  # 原来是 True, 调试用的, 会显著拖慢反传
        actions, crop_actions, _, _ = agent.act(obs, state, last_crop_action=last_crop_action, deterministic=True)
        # print(f"actions: {actions}")
        new_obs_dict, rew, terminated, truncated, info = env.step(actions)

        # 迭代更新
        obs, state = new_obs_dict["policy"], new_obs_dict["state"] # tensor[env_num, obs_dim]
        last_crop_action = crop_actions
        rgb_image = obs[:, :3, :, :]  # 提取RGB图像
        depth_image = obs[:, -1, :, :]  # 提取深度图像
        
        crop_midpoint = [((crop_actions[:, 0] + 1) / 2 * (H - 1)).to(torch.int), ((crop_actions[:, 1] + 1) / 2 * (W - 1)).to(torch.int)]

        # only visualize first environment in the batch
        rgb_vis = _to_uint8_rgb(rgb_image[0])
        depth_vis = _to_uint8_depth(depth_image[0])
        crop_actions = agent._get_crop_action_by_risk_map(depth_image)
        crop_actions = torch.tensor(crop_actions, device=env.device, dtype=torch.float32)
        temp = [((crop_actions[:, 0] + 1) / 2 * (H - 1)).to(torch.int), ((crop_actions[:, 1] + 1) / 2 * (W - 1)).to(torch.int)]
        crop_point = (temp[0][0].item(), temp[1][0].item())
        rgb_vis, depth_vis = draw_crop_anchor(rgb_vis, depth_vis, crop_point, crop_size=(60, 80))
        rgb_vis_bgr = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
        depth_vis_bgr = depth_vis
        cv2.imwrite(os.path.join(rgb_dir, f"rgb_{frame_index:05d}.png"), rgb_vis_bgr)
        cv2.imwrite(os.path.join(depth_dir, f"depth_{frame_index:05d}.png"), depth_vis_bgr)
        frame = np.concatenate([rgb_vis_bgr, depth_vis_bgr], axis=1)
        episode_frames.append(frame.copy())
        frame_index += 1

        obs = agent.obs_to_input(obs, downsample_size=config["OBS_DIM"][-2:], crop_midpoint=crop_midpoint)  # 转换为输入格式
        if (terminated.any() or truncated.any()):
            # 更新回合计数
            end_episode_num = torch.count_nonzero(terminated).item() + torch.count_nonzero(truncated).item()
            current_episode += end_episode_num
            for key in env._episode_sums.keys():
                episode_reward += info['log']["Episode_Reward/" + key]

            logger.add_scalar('eval/episode_reward', episode_reward, 0)
            avg_episode_reward = (current_episode * avg_episode_reward + episode_reward) / (current_episode + end_episode_num)
            logger.dump(0)
            logger.info(f"episode: {current_episode}, step: {0}, avg_episode_reward: {round(avg_episode_reward.item(), 4)}, success_rate: {info['log']['Metrics/success_rate']:.3f}, final_distance_to_goal: {info['log']['Metrics/final_distance_to_goal']:.3f}")
            logger.info(f'reach {info["log"]["Episode_Termination/reached"]}, collided {info["log"]["Episode_Termination/collided"]}')

            if episode_reward > 0.0 and episode_frames:
                episode_video_path = os.path.join(episode_video_dir, f"episode_{episode_id:03d}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                episode_writer = cv2.VideoWriter(episode_video_path, fourcc, 10.0, (episode_frames[0].shape[1], episode_frames[0].shape[0]))
                for frame in episode_frames:
                    episode_writer.write(frame)
                episode_writer.release()
                logger.info(f"Saved positive-reward episode video to {episode_video_path}")

            episode_frames = []
            episode_id += 1
        
    # close the environment
    logger.info(f"eval end. step: {0}, avg_episode_reward: {round(avg_episode_reward.item(), 4)}, success_rate: {info['log']['Metrics/success_rate']:.3f}, final_distance_to_goal: {info['log']['Metrics/final_distance_to_goal']:.3f}")
    logger.info(f'reach {info["log"]["Episode_Termination/reached"]}, collided {info["log"]["Episode_Termination/collided"]}')
    env.close()


def eval_while_train(agent: SACAgent, env, step, height, width, total_eval_episodes=100, logger=None, config=None):
    # 让环境使用单独的队列存储结果数据以免干扰观察训练
    env.eval = True
    avg_episode_reward = 0.0
    H, W = height, width
    # 将所有正在运行的无人机统一重置用于中途验证成功率，获取初始观测值（这是考虑到SAC不需要完整回合信息，即使无人机回合中被中断也不影响训练）
    initial_obs_dict, _ = env.reset()
    # current_obs_tensor = initial_obs_dict["policy"] # tensor[env_num, obs_dim]
    obs, state = initial_obs_dict["policy"], initial_obs_dict["state"]# tensor[env_num, obs_dim]
    obs = agent.obs_to_input(obs, downsample_size=config["OBS_DIM"][-2:])  # 转换为输入格式
    current_episode = 0
    last_crop_action = torch.zeros((env.num_envs, 2), device=env.device)
    while simulation_app.is_running() and current_episode < total_eval_episodes:
      episode_reward = 0.0
      with autograd.set_detect_anomaly(False):  # 原来是 True, 调试用的, 会显著拖慢反传

        actions, crop_actions, _, _ = agent.act(obs, state, last_crop_action=last_crop_action, deterministic=True)
        new_obs_dict, rew, terminated, truncated, info = env.step(actions)

        # 迭代更新
        obs, state = new_obs_dict["policy"], new_obs_dict["state"] # tensor[env_num, obs_dim]
        last_crop_action = crop_actions

        if config.get('USE_CROP_ACTION', False):
            crop_midpoint = [((crop_actions[:, 0] + 1) / 2 * (H - 1)).to(torch.int), ((crop_actions[:, 1] + 1) / 2 * (W - 1)).to(torch.int)]
        else:
            crop_midpoint = None
        obs = agent.obs_to_input(obs, downsample_size=config["OBS_DIM"][-2:], crop_midpoint=crop_midpoint)  # 转换为输入格式
        if (terminated.any() or truncated.any()):
            # 更新回合计数
            end_episode_num = torch.count_nonzero(terminated).item() + torch.count_nonzero(truncated).item()
            current_episode += end_episode_num
            for key in env._episode_sums.keys():
                episode_reward += info['log']["Episode_Reward/" + key]

            logger.add_scalar('eval/episode_reward', episode_reward, step)
            avg_episode_reward = (current_episode * avg_episode_reward + episode_reward) / (current_episode + end_episode_num)
            logger.dump(step)
    # 打印成功率等指标
    if config.get('USE_CROP_ACTION', False):
        logger.info(f"eval end. step: {step}, avg_episode_reward: {round(avg_episode_reward.item(), 4)}, success_rate: {info['log']['Metrics/success_rate']:.3f}, final_distance_to_goal: {info['log']['Metrics/final_distance_to_goal']:.3f}, alpha: {agent.alpha[0].item():.4f}, crop_alpha: {agent.alpha[1].item():.4f}")
    else:
        logger.info(f"eval end. step: {step}, avg_episode_reward: {round(avg_episode_reward.item(), 4)}, success_rate: {info['log']['Metrics/success_rate']:.3f}, final_distance_to_goal: {info['log']['Metrics/final_distance_to_goal']:.3f}, alpha: {agent.alpha.item():.4f}")
    logger.info(f'reach {info["log"]["Episode_Termination/reached"]}, collided {info["log"]["Episode_Termination/collided"]}')
    env.episode_reuslts_eval.clear()  # 清空结果以免干扰下一次评估
    return avg_episode_reward

def _build_env_cfg_and_cls():
    """按 --env 选环境(config + 类). base = 原环境(动态障碍物 8 个)."""
    if args_cli.env == "generated":
        from scout_mini.tasks.direct.scout_mini.scout_mini_generated_env_cfg import ScoutMiniGeneratedEnvCfg
        from scout_mini.tasks.direct.scout_mini.scout_mini_generated_env import ScoutMiniGeneratedEnv
        cfg = ScoutMiniGeneratedEnvCfg()
        cfg.dynamic_obstacle_tier = args_cli.tier
        return cfg, ScoutMiniGeneratedEnv
    if args_cli.env == "lobby":
        from scout_mini.tasks.direct.scout_mini.scout_mini_lobby_env_cfg import ScoutMiniLobbyEnvCfg
        from scout_mini.tasks.direct.scout_mini.scout_mini_lobby_env import ScoutMiniLobbyEnv
        cfg = ScoutMiniLobbyEnvCfg()
        cfg.dynamic_obstacle_tier = args_cli.tier
        return cfg, ScoutMiniLobbyEnv
    return ScoutMiniEnvCfg(), ScoutMiniEnv


def main():
    """Main function."""
    # create environment configuration
    env_cfg, env_cls = _build_env_cfg_and_cls()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    env = env_cls(cfg=env_cfg)

    work_path = os.path.dirname(os.path.abspath(__file__))
    yaml_path = work_path + "/config.yaml"
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    ##### Parameters #####
    save_path = work_path + "/" + config['CKPT_SAVE_PATH']
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mode = config['MODE']
    
    ##### Logger #####
    logger = Logger(
        logger_name="train", 
        log_dir= work_path + "/" + config['LOGGER_DIR'] + "/" + "train_" + str(config['SEED'])
    )

    
    
    # RL initialization
    # obs_dim = config['OBSERVATION_SPACE'] #env.observation_space.shape[-1]
    # obs_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]
    sac_cfg = {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "buffer_size": config['CAPACITY'],
        "obs_dim": config['OBS_DIM'],
        "state_dim": config['STATE_SHAPE'],
        "gamma": config['DISCOUNT'],
        "tau": config['CRITIC_TAU'],
        "learning_rate": config['LEARNING_RATE'],
        "alpha_learning_rate": config['ALPHA_LR'],
        "batch_size": config['BATCH_SIZE'],
        "device": env.device,
        "use_crop_action": config['USE_CROP_ACTION'],
        "crop_size": config['CROP_SIZE'],
        "init_log_alpha": config.get('INIT_LOG_ALPHA', 0.0),
        "log_alpha_max": config.get('LOG_ALPHA_MAX', None),
        "log_std_min": config.get('LOG_STD_MIN', -20),
        "log_std_max": config.get('LOG_STD_MAX', 2),
    }
    agent = SACAgent(sac_cfg["obs_dim"], action_dim, sac_cfg, logger) 
    # 自动保存配置
    checkpoint_dir = work_path + "/" + config['CKPT_SAVE_PATH'] #"logs/checkpoints" # 保存目录
    # save_dir = work_path + "/" + config['SAVE_DIR']
    save_dir = checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    save_interval = 20000  # 每 20,000 步保存一次(调试期, 方便随时加载检查)
    eval_interval = 300000  # 每 300,000 步验证一次
    last_save_step = 0      # 上次保存时的步数
    last_eval_step = 0      # 上次验证时的步数

    # [可选] 加载已有模型继续训练
    if config['PRE_MODLE'] and parser.parse_args().ckpt_steps == 0:
        if os.path.exists(os.path.join(save_dir, "model_latest.pth")):
            agent.load(os.path.join(save_dir, "model_latest.pth"))
            logger.info("已加载最新检查点，继续训练...")
    
    
    # 训练循环参数
    total_timesteps = args_cli.max_steps
    current_step = 0
    last_log_time = time.time()
    episode = 0
    max_episode_reward = -300.
    # 记录最新成功率
    success_rate = 0.

    H, W = env_cfg.camera.height, env_cfg.camera.width
    # 首次重置，获取初始观测值
    initial_obs_dict, _ = env.reset()
    env.episode_reuslts_eval.clear()  # 确保评估结果队列为空
    env.episode_reuslts.clear()  # 确保训练结果队列为空
    # current_obs_tensor = initial_obs_dict["policy"] # tensor[env_num, obs_dim]
    obs, state = initial_obs_dict["policy"], initial_obs_dict["state"]# tensor[env_num, obs_dim]
    obs = agent.obs_to_input(obs, downsample_size=config["OBS_DIM"][-2:])  # 转换为输入格式
    last_crop_action = torch.zeros((env_cfg.scene.num_envs, 2), device=env.device)  # 初始化上一次的裁剪动作
    time_env, time_update = 0.0, 0.0  # 统计 env.step 和梯度更新的耗时占比
    dist_prev = torch.linalg.norm(env._robot.data.root_pos_w[:, :2] - env._desired_pos_w[:, :2], dim=1)
    while simulation_app.is_running() and current_step < total_timesteps:
      with autograd.set_detect_anomaly(False):  # 原来是 True, 调试用的, 会显著拖慢反传
        
        actions, crop_actions, _, _ = agent.act(obs, state, last_crop_action=last_crop_action, deterministic=False)
        t0 = time.perf_counter()
        new_obs_dict, rew, terminated, truncated, info = env.step(actions)
        t1 = time.perf_counter()

        next_obs, next_state = new_obs_dict["policy"], new_obs_dict["state"] # tensor[env_num, obs_dim]
        if config['USE_CROP_ACTION']:
            crop_midpoint = [((crop_actions[:, 0] + 1) / 2 * (H - 1)).to(torch.int), ((crop_actions[:, 1] + 1) / 2 * (W - 1)).to(torch.int)]
        else:
            crop_midpoint = None
        next_obs = agent.obs_to_input(next_obs, downsample_size=config["OBS_DIM"][-2:], crop_midpoint=crop_midpoint)  # 转换为输入格式
        # 数据存储
        agent.buffer.add(obs=obs, state=state, action=actions, reward=rew.unsqueeze(-1),
                         next_obs=next_obs, next_state=next_state,done=(terminated | truncated).unsqueeze(-1), crop_action=crop_actions,
                         last_crop_action=last_crop_action)
        
        # 学习更新
        if agent.buffer.size >= sac_cfg["batch_size"]:
            agent.update(step=current_step)
        t2 = time.perf_counter()
        time_env += t1 - t0
        time_update += t2 - t1

        # 迭代更新
        # current_obs_tensor = new_obs_dict["policy"] 
        obs = next_obs
        state = next_state
        if crop_actions is not None:
            last_crop_action = crop_actions
        else:
            last_crop_action = torch.zeros((env_cfg.scene.num_envs, 2), device=env.device)
        
        current_step += env_cfg.scene.num_envs
        
        # --- 自动保存逻辑 ---
        if current_step - last_save_step >= save_interval:
            # 路径 1: 带步数的备份 (如 model_100000.pth)
            save_path = os.path.join(checkpoint_dir, f"model_{current_step}.pth")
            agent.save(save_path)
            # 路径 2: 保存最新的检查点 (方便后续直接加载)
            latest_path = os.path.join(checkpoint_dir, "model_latest.pth")
            agent.save(latest_path)
            
            last_save_step = current_step
            logger.info(f"--- 模型已保存至 {checkpoint_dir} (Step: {current_step}) ---")

        if current_step - last_eval_step >= eval_interval and success_rate > 0.5:
            logger.info("----- 开始中途验证 -----")
            avg_episode_reward = eval_while_train(agent, env, current_step, H, W, total_eval_episodes=100, logger=logger, config=config)
            last_eval_step = current_step
            if avg_episode_reward > max_episode_reward:
                agent.save(work_path + "/save_model/model_best.pth")
            continue  # 跳过本次日志打印，避免干扰观察
        
        # 打印日志（可以调整频率）
        # if current_step % (200 * env_cfg.scene.num_envs) == 0:
        #     print(f"Timestep: {current_step}, Mean Reward: {rew.mean().item():.3f}, Alpha: {agent.alpha.item():.4f}")
        
        if (terminated.any() or truncated.any()) and time.time() - last_log_time >= 60.0:
            episode += torch.count_nonzero(terminated).item() + torch.count_nonzero(truncated).item()
            # 打印成功率等指标
            success_rate = info['log']['Metrics/success_rate']
            logger.add_scalar('train/success_rate', success_rate, step=current_step)
            total_reward = 0.0
            for key in env._episode_sums.keys():
                sub_reward = info['log']["Episode_Reward/" + key]
                total_reward += sub_reward
                logger.info(f"{key} Reward: {info['log']['Episode_Reward/' + key]}")
            if config['USE_CROP_ACTION']:
                logger.info(f"Episode: {episode}, Timestep: {current_step}, Success Rate: {info['log']['Metrics/success_rate']:.3f}, Final Distance To Goal: {info['log']['Metrics/final_distance_to_goal']}, Alpha: {agent.alpha[0].item():.4f}, crop_alpha: {agent.alpha[1].item():.4f}")
            else:
                logger.info(f"Episode: {episode}, Timestep: {current_step}, Success Rate: {info['log']['Metrics/success_rate']:.3f}, Final Distance To Goal: {info['log']['Metrics/final_distance_to_goal']}, Alpha: {agent.alpha.item():.4f}")
            logger.info(f"Episode Result: reach {info['log']['Episode_Termination/reached']}, collided {info['log']['Episode_Termination/collided']}")
            # debug: 观察策略退化(不动/打转)情况
            dist_now = torch.linalg.norm(env._robot.data.root_pos_w[:, :2] - env._desired_pos_w[:, :2], dim=1)
            goal_dir = state[:, 4]  # 目标点在机体系下的方位角
            ang_act = actions[:, 1]
            corr_ang = torch.corrcoef(torch.stack([goal_dir, ang_act]))[0, 1].item()
            corr_lin = torch.corrcoef(torch.stack([goal_dir.abs(), actions[:, 0]]))[0, 1].item()
            # debug: 直接把 goal_dir 扫一遍, 看策略的确定性输出(即策略意图)是否随目标方向变化
            with torch.no_grad():
                probe_state = state.clone()
                probe_goal = torch.linspace(-3.1, 3.1, state.shape[0], device=env.device)
                probe_state[:, 4] = probe_goal
                # 图像固定成同一张, 这样探测器只反映 state 对策略的影响
                probe_obs = obs[0:1].expand(state.shape[0], -1, -1, -1)
                probe_mu, _, _, probe_logstd = agent.actor(
                    probe_obs, probe_state, last_crop_action=last_crop_action,
                    compute_pi=False, compute_log_pi=False, use_crop_action=False
                )
                probe_w = probe_mu[:, 1]
                probe_v = probe_mu[:, 0]
                probe_corr_w = torch.corrcoef(torch.stack([probe_goal, probe_w]))[0, 1].item()
                probe_corr_v = torch.corrcoef(torch.stack([probe_goal.abs(), probe_v]))[0, 1].item()
            logger.info(
                f"Debug: action_mean {[round(v, 3) for v in actions.mean(dim=0).tolist()]}, "
                f"action_std {[round(v, 3) for v in actions.std(dim=0).tolist()]}, "
                f"|v| {env._robot.data.root_lin_vel_b[:, 0].abs().mean().item():.3f}, "
                f"|w| {env._robot.data.root_ang_vel_b[:, 2].abs().mean().item():.3f}, "
                f"mean_dist {dist_now.mean().item():.3f}, "
                f"d<0.5 {100.0 * (dist_now < 0.5).float().mean().item():.0f}%, "
                f"d<1.0 {100.0 * (dist_now < 1.0).float().mean().item():.0f}%, "
                f"径向速度 {(dist_prev - dist_now).mean().item() / env.step_dt:+.2f}m/s, "
                f"corr(goaldir, w_act) {corr_ang:+.3f}, corr(|goaldir|, v_act) {corr_lin:+.3f}, "
                f"PROBE corr(goaldir, w_mu) {probe_corr_w:+.3f} (策略意图是否朝目标转), "
                f"PROBE corr(|goaldir|, v_mu) {probe_corr_v:+.3f} (目标在后方时是否减速), "
                f"policy_std {probe_logstd.exp().mean().item():.3f}, "
                f"mu_amp(v,w) {probe_mu[:, 0].abs().mean().item():.2f}/{probe_mu[:, 1].abs().mean().item():.2f}, "
                f"t_env {time_env:.1f}s / t_update {time_update:.1f}s"
            )
            time_env, time_update = 0.0, 0.0
            dist_prev = dist_now
            logger.add_scalar('train/episode_reward', total_reward, current_step)
            logger.add_scalar('train/epsiode step', info["log"]["Episode_Termination/Episode_Step"], current_step)
            logger.add_scalar('train/episode', episode, current_step)
            logger.dump(step=current_step)
            last_log_time = time.time()
            
    # close the environment
    avg_episode_reward = eval_while_train(agent, env, current_step, H, W, total_eval_episodes=50, logger=logger, config=config)
    agent.save(work_path + "save_model/model_last.pth")
    logger.info("Train end...")
    env.close()
    if args_cli.max_steps < 10000000:
        # 有限步数的课程阶段: 存完就直接退出. kit 的关闭流程实测会偶发卡死(挂 10 分钟以上),
        # 卡住会阻塞课程脚本进入下一个阶段.
        logger.info(f"阶段结束(max_steps={args_cli.max_steps}), 直接退出进程")
        os._exit(0)



if __name__ == "__main__":
    # run the main function
    if not args_cli.eval:
        main()
    # test()
    # eval_viz(10)
    else:
        eval(100)
    # close sim app
    simulation_app.close()
