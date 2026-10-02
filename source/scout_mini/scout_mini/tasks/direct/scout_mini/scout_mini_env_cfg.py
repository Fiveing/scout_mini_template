# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg

##
# Pre-defined configs
##
from isaaclab.sensors import ContactSensorCfg, CameraCfg, RayCasterCfg, patterns
from gymnasium import spaces


@configclass
class ScoutMiniEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0
    # - spaces definition
    action_space = 2
    observation_space = 4
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot(s)
    usd_path = "/home/use_this_new_user/scout_mini/scout_mini/scripts/assets/robots/scout_mini.usd"
    usd_path = "/home/use_this_new_user/IsaacLab/scripts/scout_mini.usd"
    scout_cfg = ArticulationCfg(
        prim_path = "/World/envs/env_.*/ScoutMini",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,   # 接触传感器需要这个才能报接触力
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            visible=False,
        ),
        actuators={
            "wheel_acts": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
                effort_limit_sim=200.0,
                velocity_limit_sim=100.0,
            )
        },
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={}, joint_vel={}),
    )
    robot : ArticulationCfg = scout_cfg.replace(prim_path="/World/envs/env_.*/ScoutMini")

    robot_radius = 0.357
    # base_link 静止时离地高度(URDF 里 base_footprint 在 -0.178). 原来出生 z=0 等于把车
    # 埋进地面 0.178m, 物理引擎会把它弹出来并产生横向漂移, 在 Lobby 这种窄地方几步就撞墙.
    spawn_height = 0.178
    start_clearance = 0.90   # 固定起终点至少要离墙/障碍物这么远(车体之外再留的余量)
    # 0.3 太紧: Lobby 的点是给无人机选的, 小车一个急转就会蹭到墙 -> 出生几秒内就判碰撞
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # sensors
    camera = CameraCfg(
        # prim_path="/World/envs/env_.*/Robot/body/front_camera",
        prim_path="/World/envs/env_.*/ScoutMini/base_link/front_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        # offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    # ray_caster = RayCasterCfg(
    #     prim_path="/World/envs/env_.*/Robot/body",
    #     update_period=0.02,
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
    #     ray_alignment="base",
    #     pattern_cfg=patterns.BpearlPatternCfg(
    #             horizontal_res=10, # horizontal default is set to 10
    #             # vertical_ray_angles=np.linspace(-10, 20, 4),
    #         ),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )
    ray_caster = RayCasterCfg(
        prim_path="/World/envs/env_.*/ScoutMini/base_link",
        mesh_prim_paths=["/World/ground"],
        offset=RayCasterCfg.OffsetCfg(pos=(0.125, 0, 0.1),rot=(1.0, 0.0, 0.0, 0.0)),
        update_period=1 / 60,
        # pattern_cfg=patterns.LidarPatternCfg(
        #     channels=4, vertical_fov_range=(-10, 20), horizontal_fov_range=(-179, 179), horizontal_res=10.0
        # ),
        pattern_cfg=patterns.LidarPatternCfg(
                    channels=1, vertical_fov_range=(0, 0), horizontal_fov_range=(-179, 179), horizontal_res=5.0
        ),

        ray_alignment="base",
        debug_vis=False,
        # attach_yaw_only=True,
    )
    ray_caster_range = 5.   # 最大检测距离
    depth_image_range = 5.0 # 深度图最大距离

    # 碰撞判定方式 (2026-09-17 实测结论):
    #   "privileged"  特权信息: 每步用障碍物位置 + 车身 footprint 做 2D 矩形相交.
    #                 实测在刚好物理接触时触发(距离稳定值 0.63m 处), 空地误触发 0 次 -> 默认用它
    #   "contact"     接触传感器: 实测"与障碍物的接触力"在平地行驶时本底 90~130N,
    #                 撞上瞬间 2600~3200N, 阈值取 500N 有 4 倍余量. 注意带 filter 的
    #                 force_matrix_w 实测恒为 0 (PhysX 过滤 pattern 没匹配上障碍物路径)
    collision_check_mode = "privileged"
    contact_force_threshold = 500.0   # N, 未过滤接触力超过该值算撞上
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/ScoutMini/.*",     # 车体 + 四个轮子
        # filter_prim_paths_expr=["/World/Obstacles/.*"],  # 实测这个 pattern 匹配不上, 见上面的说明
        update_period=0.0,      # 每个物理步都更新
        history_length=1,
        debug_vis=False,
    )
    robot_footprint_half = (0.31, 0.29)  # 车身(含轮子)半长/半宽(m), privileged 模式用
    max_vel = 2.0
    min_vel = 0.0
    # 角速度上限决定最小转弯半径 R = v / w_max. 原来 w_max=1.0, 全速 2m/s 时 R=2m,
    # 车根本拐不进目标点周围 0.5m 的范围, 只能绕着目标转圈. 放宽到 2.0 rad/s (实车也能做到)
    max_angular_vel = 2.0
    min_angular_vel = -2.0
    reach_threshold = 0.5   # 判定"到达目标点"的距离(m), 约等于车体自身半径
    min_start_goal_distance = 2.0  # 起点与目标点的最小间距(m), 避免"开局即到达"


    # reward scales
    # 注意: 这里的奖励量级都按"每步"给, 不再乘 step_dt.
    # SAC 的熵项约为 alpha * log_pi ~ O(1), 如果任务奖励每步只有 1e-3, 策略梯度会被熵项完全淹没,
    # 机器人就会学成"原地打转/随机游走". 因此把每步任务奖励保持在 O(0.1~1).
    progress_reward_scale = 15.0        # 每靠近目标 1m 的奖励(全速 2m/s 时约 +0.6/step)
    heading_progress_reward_scale = 15.0  # 每 1rad 转向目标的奖励(角速度的即时奖励)
    vel_towards_goal_reward_scale = 1.5  # 速度朝目标方向投影 [-1,1]
    heading_align_reward_scale = 1.0     # 朝向对齐(差速小车"先转向"的引导项)
    align_vel_threshold = 0.5            # 前进速度超过该值时朝向对齐奖励饱和
    distance_penalty_scale = 0.06        # 距离惩罚(每米每步), 抑制绕远路/磨蹭
    risk_penalty_scale = -1.5            # 离障碍物过近的惩罚(<=0), 0 表示关闭
    risk_margin = 1.0                    # 安全距离(m): 小于该距离开始线性惩罚
    smooth_reward_scale = 0.0            # 动作平滑惩罚, 先关闭
    time_penalty = -0.15                 # 每步时间惩罚, 促使尽快到达
    collision_penalty = -60.0            # 撞到障碍物
    time_out_penalty = -60.0             # 超时未到达
    reached_reward = 300.0               # 到达目标点


    # scene configuration
    obstacle_map_range = 4.0 * 2 # 障碍物地图范围（正方形边长）
    difficulty = "easy" # "easy", "medium", "hard"
    alternative_num_obstacles = {"easy": 0.03, "medium": 0.06, "hard": 0.1} #0.03
    num_obstacles = round(obstacle_map_range ** 2 * alternative_num_obstacles[difficulty]) # 障碍物数量，基于地图范围和难度设定
    obstacle_height_range = [0.5, 4.5]
    obstacle_width_range = (0.4, 1.1)

    # dynamic obstacles
    N_w, N_h = 4, 2
    dynamic_obstacle_nums = 8  # 先设 8 个动态障碍物, 验证从无人机版迁移过来的生成/移动代码能否跑通
    dyn_obs_category_num = N_w * N_h # 动态障碍物数量(N_w种宽度范围，N_h种高度范围)
    dyn_obs_num_of_each_category = int(dynamic_obstacle_nums / dyn_obs_category_num)
    dynamic_obstacle_nums = dyn_obs_num_of_each_category * dyn_obs_category_num

    # 动态障碍物高度(小车在地面上, 全部贴地; 原无人机版 3D 障碍物的 z 是随机高度会浮空)
    dyn_obs_3d_height = 1.0   # 方块
    dyn_obs_2d_height = 1.5   # 圆柱(类似行人)
    local_range = [5.0, 5.0, 4.5]
    vel_range = [0.5, 1.5]