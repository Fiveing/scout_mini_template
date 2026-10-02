# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from collections import deque
from isaaclab.utils.math import sample_uniform


import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.assets.asset_base import AssetBase
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_apply_inverse, quat_apply
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip
from isaaclab.sensors import ContactSensor, RayCaster, Camera
import isaacsim.core.utils.prims as prim_utils

from .scout_mini_env_cfg import ScoutMiniEnvCfg
import numpy as np


class ScoutMiniEnv(DirectRLEnv):
    cfg: ScoutMiniEnvCfg

    def __init__(self, cfg: ScoutMiniEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.episode_reuslts = deque(maxlen=100)# 只统计100回合内的成功率
        self.episode_reuslts_eval = deque(maxlen=100) # 单独统计评估时的成功率
        self.eval = False # 是否为评估模式
        self.reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.collided = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_distance_to_goal = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.last_heading_cos = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # self.recorder = Recorder(None)
        self.last_rgb_image = None
        # print(self.num_envs, "environments")
        # print(self.max_episode_length, "max episode length")
        # print(self.max_episode_length_s, "max episode length in seconds")
        

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "progress",
                "heading_progress",
                "vel_towards_goal",
                "heading_align",
                "distance_penalty",
                "risk",
                "smooth",
                "time",
                "collision",
                "time_out",
                "reached",
            ]
        }

        self.actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.last_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        act_max = torch.tensor([self.cfg.max_vel, self.cfg.max_angular_vel], device=self.device)
        act_min = torch.tensor([self.cfg.min_vel, self.cfg.min_angular_vel], device=self.device)
        self.act_mean = (act_max + act_min) / 2
        self.act_std = (act_max - act_min) / 2

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self._lidar_sensor = RayCaster(self.cfg.ray_caster)
        self._camera = Camera(self.cfg.camera)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.scene.sensors["lidar_sensor"] = self._lidar_sensor
        self.scene.sensors["camera"] = self._camera

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        # print("Terrain env spacing: ", self.cfg.terrain.env_spacing)
        # 唤起地面(直接导入usd地形时可以去掉,但是对应的碰撞过滤的prim path也要改)
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain) # TerrainImporter(self.cfg.terrain)
        self.scene._terrain = self._terrain

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path, "/World/Obstacles"])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # self._generate_walls()
        self._generate_obstacles()
        if self.cfg.dynamic_obstacle_nums > 0:
            self.generate_dynamic_obstacles()

    def _generate_walls(self):
        wall_thickness = 0.1
        wall_cfg = sim_utils.CuboidCfg(
            size=(wall_thickness, self.cfg.obstacle_map_range + 1.0, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        )
        wall_cfg.func(f"/World/Walls/_x1", wall_cfg, translation=[0.5 + self.cfg.obstacle_map_range / 2, 0.0, 1.5])
        wall_cfg.func(f"/World/Walls/_x2", wall_cfg, translation=[-0.5 - self.cfg.obstacle_map_range / 2, 0.0, 1.5])
        wall_cfg.func(f"/World/Walls/_y1", wall_cfg, translation=[0.0, 0.5 + self.cfg.obstacle_map_range / 2 + wall_thickness, 1.5],
                        orientation=[np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
        wall_cfg.func(f"/World/Walls/_y2", wall_cfg, translation=[0.0, -0.5 - self.cfg.obstacle_map_range / 2 - wall_thickness, 1.5],
                          orientation=[np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])

    def _generate_obstacles(self):
        # 在每个被围起来的方形区域中随机生成障碍物
        self.num_obstacles = self.cfg.num_obstacles if hasattr(self.cfg, 'num_obstacles') else 0

        prev_positions = []

        def check_valid_pos(pos, prev_positions, min_dist=0.8):
            for p in prev_positions:
                if torch.linalg.norm(pos - p) < min_dist:
                    return False
            return True
        
        # add obstacles
        self.obstacles = []
        self.obstacles_width = []
        for j in range(self.num_obstacles):
            obstacle_width = sample_uniform(
                torch.tensor(self.cfg.obstacle_width_range[0]),
                torch.tensor(self.cfg.obstacle_width_range[1]),
                (1,),
                self.device,
            ).item()
            obstacle_pos = torch.zeros(3, device=self.device)
            valid_pos = False
            while not valid_pos:
                obstacle_pos[0] = sample_uniform(
                    torch.tensor(-self.cfg.obstacle_map_range / 2 + obstacle_width/2),
                    torch.tensor(self.cfg.obstacle_map_range / 2 - obstacle_width/2),
                    (1,),
                    self.device,
                ).item()
                obstacle_pos[1] = sample_uniform(
                    torch.tensor(-self.cfg.obstacle_map_range / 2 + obstacle_width/2),
                    torch.tensor(self.cfg.obstacle_map_range / 2 - obstacle_width/2),
                    (1,),
                    self.device,
                ).item()
                valid_pos = check_valid_pos(obstacle_pos, prev_positions, min_dist=0.8)
            prev_positions.append(obstacle_pos.clone())
            obstacle_pos[2] = 0.75
            # box_cfg = sim_utils.CuboidCfg(
            #     size=(obstacle_width, obstacle_width, obstacle_height),
            #     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            #     mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            #     collision_props=sim_utils.CollisionPropertiesCfg(),
            #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            # )
            box_cfg = RigidObjectCfg(
                prim_path=f"/World/Obstacles/Box{j}",
                spawn=sim_utils.CuboidCfg(
                size=(obstacle_width, obstacle_width, 1.5),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=obstacle_pos.tolist()
                ),
            )
            box_object = RigidObject(cfg=box_cfg)
            self.obstacles.append(box_object)
            self.obstacles_width.append(obstacle_width)
            # box_cfg.func(f"/World/Obstacles/Box{i}", box_cfg, translation=obstacle_pos.tolist())
            # self.scene.rigid_objects[f"Box{i}"] = box_object
    
    def generate_dynamic_obstacles(self):
        """生成动态障碍物.

        与无人机版不同: 小车在地面上, 所以 (1) 所有障碍物都贴地(原版 3D 障碍物的 z 是随机高度,
        会浮在空中), (2) 数量不再受 N_w*N_h 分类结构的限制, 支持任意个数(课程学习要 4->12 逐个上调),
        因此改成"每个障碍物一个 RigidObject", 尺寸类别轮转分配.
        """
        n_obs = self.cfg.dynamic_obstacle_nums
        self.dyn_obs_list = []
        self.dyn_obs_pos = torch.zeros((n_obs, 3), dtype=torch.float, device=self.device)
        self.dyn_obs_goal = torch.zeros((n_obs, 3), dtype=torch.float, device=self.device)
        self.dyn_obs_origin = torch.zeros((n_obs, 3), dtype=torch.float, device=self.device)
        self.dyn_obs_vel = torch.zeros((n_obs, 3), dtype=torch.float, device=self.device)
        self.dyn_obs_step_count = 0  # dynamic obstacle motion step count
        self.dyn_obs_size = torch.zeros((n_obs, 3), dtype=torch.float, device=self.device)

        max_obs_width = 1.0
        self.max_obs_3d_height = self.cfg.dyn_obs_3d_height   # 方块高度(贴地)
        self.max_obs_2d_height = self.cfg.dyn_obs_2d_height   # 圆柱高度(贴地)
        cuboid_category_num = cylinder_category_num = int(self.cfg.dyn_obs_category_num / self.cfg.N_h)
        n_categories = cuboid_category_num + cylinder_category_num

        def check_pos_validity(prev_pos_list, curr_pos, min_dist):
            for prev_pos in prev_pos_list:
                if np.linalg.norm(curr_pos - prev_pos) <= min_dist:
                    return False
            return True

        # 期望间距: 让障碍物尽量均匀铺开(否则会扎堆)
        obs_dist = 2 * np.sqrt(self.cfg.obstacle_map_range ** 2 / max(n_obs, 1)) if n_obs > 0 else 0.0
        curr_obs_dist = obs_dist
        prev_pos_list = []
        for k in range(n_obs):
            category_idx = k % max(n_categories, 1)     # 尺寸类别轮转, 保证大小混合
            if category_idx < cuboid_category_num:
                width = float(category_idx + 1) * max_obs_width / float(self.cfg.N_w)
                height, radius, is_cuboid = self.max_obs_3d_height, None, True
            else:
                radius = float(category_idx - cuboid_category_num + 1) * max_obs_width / float(self.cfg.N_w) / 2.0
                width = radius * 2
                height, is_cuboid = self.max_obs_2d_height, False
            oz = height / 2.0                             # 贴地: 中心 = 一半高度
            # 均匀分布采样
            start_time = time.time()
            while True:
                ox = np.random.uniform(low=-self.cfg.obstacle_map_range / 2, high=self.cfg.obstacle_map_range / 2)
                oy = np.random.uniform(low=-self.cfg.obstacle_map_range / 2, high=self.cfg.obstacle_map_range / 2)
                valid = check_pos_validity(prev_pos_list, np.array([ox, oy]), curr_obs_dist)
                if time.time() - start_time > 0.1:        # 放不下就放宽间距要求
                    curr_obs_dist *= 0.8
                    start_time = time.time()
                if valid:
                    prev_pos_list.append(np.array([ox, oy]))
                    break
            curr_obs_dist = obs_dist

            prim_path = f"/World/DynObs/obs_{k}"
            if is_cuboid:
                spawn_cfg = sim_utils.CuboidCfg(
                    size=[width, width, height],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                )
            else:
                spawn_cfg = sim_utils.CylinderCfg(
                    radius=radius,
                    height=height,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                )
            obj = RigidObject(
                cfg=RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=spawn_cfg,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(ox, oy, oz)),
                )
            )
            self.dyn_obs_list.append(obj)
            self.dyn_obs_size[k] = torch.tensor([width, width, height], dtype=torch.float, device=self.device)
            self.dyn_obs_origin[k] = torch.tensor([ox, oy, oz], dtype=torch.float, device=self.device)
            self.dyn_obs_pos[k] = torch.tensor([ox, oy, oz], dtype=torch.float, device=self.device)
            self.dyn_obs_goal[k] = torch.tensor([ox, oy, oz], dtype=torch.float, device=self.device)
        # 半径/半宽(圆柱用半径, 方块用半边长) 供碰撞判定用
        self.dyn_obs_radius = torch.where(
            self.dyn_obs_size[:, 0] == self.dyn_obs_size[:, 1],
            self.dyn_obs_size[:, 0] / 2.0,
            self.dyn_obs_size[:, 0] / 2.0,
        ) * float(np.sqrt(2) / 2 + 0.5) if False else self.dyn_obs_size[:, 0] / 2.0
        self.dyn_obs_is_cuboid = torch.tensor(
            [(k % max(n_categories, 1)) < cuboid_category_num for k in range(n_obs)],
            dtype=torch.bool, device=self.device,
        )

    def move_dynamic_obstacles(self):
        """动态障碍物平面随机漫游(小车版: z 固定, 只走 x/y)"""
        if self.cfg.dynamic_obstacle_nums == 0:
            return
        # Step 1: 到点后重新采样局部目标
        dyn_obs_goal_dist = (
            torch.sqrt(torch.sum((self.dyn_obs_pos - self.dyn_obs_goal) ** 2, dim=1))
            if self.dyn_obs_step_count != 0
            else torch.zeros(self.dyn_obs_pos.size(0), device=self.device)
        )
        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5
        num_new_goal = int(torch.sum(dyn_obs_new_goal_mask).item())
        if num_new_goal > 0:
            sample_x_local = -self.cfg.local_range[0] + 2.0 * self.cfg.local_range[0] * torch.rand(
                num_new_goal, 1, dtype=torch.float, device=self.device)
            sample_y_local = -self.cfg.local_range[1] + 2.0 * self.cfg.local_range[1] * torch.rand(
                num_new_goal, 1, dtype=torch.float, device=self.device)
            sample_goal_local = torch.cat([sample_x_local, sample_y_local], dim=1)
            self.dyn_obs_goal[dyn_obs_new_goal_mask, :2] = (
                self.dyn_obs_origin[dyn_obs_new_goal_mask, :2] + sample_goal_local
            )
        # 限制在场地内, z 保持不变(贴地)
        self.dyn_obs_goal[:, 0] = torch.clamp(self.dyn_obs_goal[:, 0],
                                              min=-self.cfg.obstacle_map_range / 2, max=self.cfg.obstacle_map_range / 2)
        self.dyn_obs_goal[:, 1] = torch.clamp(self.dyn_obs_goal[:, 1],
                                              min=-self.cfg.obstacle_map_range / 2, max=self.cfg.obstacle_map_range / 2)
        self.dyn_obs_goal[:, 2] = self.dyn_obs_pos[:, 2]

        # Step 2: 每 2 秒重新采样一次速度
        if self.dyn_obs_step_count % int(2.0 / self.step_dt) == 0:   # 每 2 秒重采样一次速度
            self.dyn_obs_vel_norm = self.cfg.vel_range[0] + (self.cfg.vel_range[1] - self.cfg.vel_range[0]) * torch.rand(
                self.dyn_obs_vel.size(0), 1, dtype=torch.float, device=self.device)
            direction = self.dyn_obs_goal - self.dyn_obs_pos
            direction[:, 2] = 0.0
            self.dyn_obs_vel = self.dyn_obs_vel_norm * direction / (
                torch.norm(direction, dim=1, keepdim=True) + 1e-6)

        # Step 3: 积分位置(用 step_dt: 本函数每个 env step 只调用一次, 一个 env step = decimation 个物理步)
        self.dyn_obs_pos += self.dyn_obs_vel * self.step_dt
        self.dyn_obs_pos[:, 2] = self.dyn_obs_origin[:, 2]

        # Step 4: 写回仿真
        dyn_obs_pose = torch.zeros(self.dyn_obs_pos.size(0), 7, dtype=torch.float, device=self.device)
        dyn_obs_pose[:, :3] = self.dyn_obs_pos
        dyn_obs_pose[:, 3] = 1.0
        for k, dynamic_obstacle in enumerate(self.dyn_obs_list):
            dynamic_obstacle.write_root_pose_to_sim(dyn_obs_pose[k:k + 1])
            dynamic_obstacle.write_data_to_sim()
            dynamic_obstacle.update(self.cfg.sim.dt)

        self.dyn_obs_step_count += 1

    # ---------------- 评估环境共用: 固定的 20 组起终点 ----------------
    def init_fixed_pairs(self, starts_xy, goals_xy):
        """固定起终点配对(评估环境用). starts_xy/goals_xy: [(x, y), ...]"""
        n = min(len(starts_xy), len(goals_xy))
        self._fixed_start_positions = torch.zeros((n, 3), dtype=torch.float, device=self.device)
        self._fixed_target_positions = torch.zeros((n, 3), dtype=torch.float, device=self.device)
        for i in range(n):
            self._fixed_start_positions[i, :2] = torch.tensor(starts_xy[i], device=self.device)
            self._fixed_target_positions[i, :2] = torch.tensor(goals_xy[i], device=self.device)
        self._fixed_start_positions[:, 2] = self.cfg.spawn_height
        self._fixed_target_positions[:, 2] = self.cfg.spawn_height
        self._pair_count = n
        self._eval_reset_counter = 0
        self._eval_last_pair_indices = None

    def _reset_target_fixed(self, env_ids):
        if getattr(self, "_fixed_target_positions", None) is None:
            return False
        steps = torch.arange(env_ids.size(0), device=self.device, dtype=torch.int64)
        pair_indices = (self._eval_reset_counter + steps) % self._pair_count
        self._desired_pos_w[env_ids] = self._fixed_target_positions[pair_indices]
        self._eval_last_pair_indices = pair_indices
        self._eval_reset_counter = (self._eval_reset_counter + env_ids.size(0)) % self._pair_count
        return True

    def _reset_start_pos_fixed(self, env_ids):
        if getattr(self, "_fixed_start_positions", None) is None:
            return None
        if self._eval_last_pair_indices is not None:
            pair_indices = self._eval_last_pair_indices
            self._eval_last_pair_indices = None
        else:
            steps = torch.arange(env_ids.size(0), device=self.device, dtype=torch.int64)
            pair_indices = (self._eval_reset_counter + steps) % self._pair_count
            self._eval_reset_counter = (self._eval_reset_counter + env_ids.size(0)) % self._pair_count
        return self._fixed_start_positions[pair_indices]

    # ---------------- 评估环境共用: 按 meta 规格生成"两点往返"动态障碍物 ----------------
    def spawn_dynamic_obstacles_from_specs(self, specs):
        """specs: list of dict(x1,y1,x2,y2,r,h,v) —— 圆柱在两点之间匀速往返(ping-pong),
        贴地, 朝向恒为单位四元数(复刻 ROS1 评估场景). 四档 4/6/8/12 是嵌套子集, 取前 N 项即可.
        """
        n = len(specs)
        self.dyn_obs_list = []
        self.dyn_obs_pos = torch.zeros((n, 3), dtype=torch.float, device=self.device)
        self.dyn_obs_radius = torch.zeros(n, dtype=torch.float, device=self.device)
        self.dyn_obs_size = torch.zeros((n, 3), dtype=torch.float, device=self.device)
        self._dyn_x1 = torch.tensor([s["x1"] for s in specs], dtype=torch.float, device=self.device)
        self._dyn_y1 = torch.tensor([s["y1"] for s in specs], dtype=torch.float, device=self.device)
        self._dyn_x2 = torch.tensor([s["x2"] for s in specs], dtype=torch.float, device=self.device)
        self._dyn_y2 = torch.tensor([s["y2"] for s in specs], dtype=torch.float, device=self.device)
        self._dyn_h = torch.tensor([s["h"] for s in specs], dtype=torch.float, device=self.device)
        self._dyn_v = torch.tensor([s["v"] for s in specs], dtype=torch.float, device=self.device)
        for i, s in enumerate(specs):
            obj = RigidObject(
                cfg=RigidObjectCfg(
                    prim_path=f"/World/DynObs/obs_{i}",
                    spawn=sim_utils.CylinderCfg(
                        radius=s["r"],
                        height=s["h"],
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(s["x1"], s["y1"], s["h"] / 2.0)),
                )
            )
            self.dyn_obs_list.append(obj)
            self.dyn_obs_radius[i] = s["r"]
            self.dyn_obs_size[i] = torch.tensor([2.0 * s["r"], 2.0 * s["r"], s["h"]], device=self.device)
            self.dyn_obs_pos[i] = torch.tensor([s["x1"], s["y1"], s["h"] / 2.0], device=self.device)
        self._dyn_sim_time = 0.0
        self.cfg.dynamic_obstacle_nums = n     # 让基类风格的数量判断保持一致

    def move_dynamic_obstacles_pingpong(self):
        """两点匀速往返. 注意相位推进用 step_dt: 本函数每个 env step 只调用一次,
        而一个 env step = decimation 个物理步, 用 sim.dt 会让实际速度只有标称值的一半.
        """
        t = self._dyn_sim_time
        dx = self._dyn_x2 - self._dyn_x1
        dy = self._dyn_y2 - self._dyn_y1
        length = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-6)
        vel = self._dyn_v.clamp_min(1e-6)
        half = length / vel
        period = 2.0 * half
        tt = torch.remainder(torch.tensor(t, device=self.device), period)
        s = torch.where(tt <= half, tt * vel, 2.0 * length - tt * vel)   # 三角波弧长
        k = s / length
        pos = torch.stack([self._dyn_x1 + k * dx, self._dyn_y1 + k * dy, self._dyn_h / 2.0], dim=1)
        self.dyn_obs_pos[:, :2] = pos[:, :2]
        self.dyn_obs_pos[:, 2] = self._dyn_h / 2.0
        pose = torch.zeros(pos.size(0), 7, dtype=torch.float, device=self.device)
        pose[:, :3] = pos
        pose[:, 3] = 1.0
        for i, obj in enumerate(self.dyn_obs_list):
            obj.write_root_pose_to_sim(pose[i:i + 1])
            obj.write_data_to_sim()
            obj.update(self.cfg.sim.dt)
        self._dyn_sim_time += self.step_dt

    def _reset_target(self, env_ids):
        # decide which side
        masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
        shift = self.cfg.obstacle_map_range / 2 + 0.3
        # shift -= 0.5 # to avoid being too close to the wall
        shifts = torch.tensor([[0., shift, 0.], [0., -shift, 0.], [shift, 0., 0.], [-shift, 0., 0.]], dtype=torch.float, device=self.device)
        mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
        selected_masks = masks[mask_indices]
        selected_shifts = shifts[mask_indices]


        # generate random positions
        target_pos = 2 * shift * torch.rand(env_ids.size(0), 3, dtype=torch.float, device=self.device) + (-shift)
        heights = 0.0
        target_pos[:, 2] = heights# height
        target_pos = target_pos * selected_masks + selected_shifts
        
        # apply target pos
        self._desired_pos_w[env_ids] = target_pos #+ self.scene.env_origins[env_ids, :3]
    
    def _reset_start_pos(self, env_ids):
        # decide which side
        masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
        shift = self.cfg.obstacle_map_range / 2 + 0.3
        # shift -= 0.5 # to avoid being too close to the wall
        shifts = torch.tensor([[0., shift, 0.], [0., -shift, 0.], [shift, 0., 0.], [-shift, 0., 0.]], dtype=torch.float, device=self.device)
        mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
        selected_masks = masks[mask_indices]
        selected_shifts = shifts[mask_indices]

        # generate random positions
        start_pos = 2 * shift * torch.rand(env_ids.size(0), 3, dtype=torch.float, device=self.device) + (-shift)
        heights = 0.0
        start_pos[:, 2] = heights# height
        start_pos = start_pos * selected_masks + selected_shifts

        return start_pos# + self.scene.env_origins[env_ids, :3]


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        # 动态障碍物每 env step 推进一次(原来漏了这个调用, 障碍物生成后一直没动过)
        if self.cfg.dynamic_obstacle_nums > 0:
            self.move_dynamic_obstacles()

    def _apply_action(self) -> None:
        vel = self.actions * self.act_std + self.act_mean
        vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        vel_cmd[:, 0] = vel[:, 0]
        vel_cmd = quat_apply(self._robot.data.root_quat_w, vel_cmd)
        root_velocity = torch.zeros(self.num_envs, 6, device=self.device)
        root_velocity[:, :2] = vel_cmd[:, :2]
        root_velocity[:, 5] = vel[:, 1]
        self._robot.write_root_velocity_to_sim(root_velocity)

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        log_distance = torch.log(torch.linalg.norm(desired_pos_b[:, :2], dim=-1) + 1.0)
        goal_direction = torch.atan2(desired_pos_b[:, 1], desired_pos_b[:, 0])

        depth_image = self._camera.data.output["distance_to_image_plane"]
        rgb_image = self._camera.data.output["rgb"]
        rgb_image = rgb_image.permute(0, 3, 1, 2)
        
        depth_image = torch.nan_to_num(depth_image, nan=self.cfg.depth_image_range, posinf=self.cfg.depth_image_range, neginf=0.0).clamp_max(self.cfg.depth_image_range) / self.cfg.depth_image_range
        depth_image = depth_image.permute(0, 3, 1, 2)
        if self.last_rgb_image is None:
            self.last_rgb_image = rgb_image
        obs = torch.cat([rgb_image, self.last_rgb_image, depth_image], dim=1)  # shape (N_envs, C, H, W)
        self.last_rgb_image = rgb_image
        # obs = torch.nn.AdaptiveAvgPool2d((60, 80))(obs)  # downsample to reduce observation dimension, shape (N_envs, C, 16, 16)
        state = torch.cat(
            [
                self._robot.data.root_lin_vel_b[:, :2],
                self._robot.data.root_ang_vel_b[:, 2].unsqueeze(-1),
                log_distance.unsqueeze(-1),
                goal_direction.unsqueeze(-1),
            ],
            dim=-1,
        )
        observations = {"policy": obs, "state": state}
        self.last_actions = self.actions.clone()
        
        return observations

    def _get_rewards(self) -> torch.Tensor:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        # 目标点在机体坐标系下的水平距离(m)
        distance_to_goal = torch.linalg.norm(desired_pos_b[:, :2], dim=1)
        eps = 1e-6

        # 1) 距离进度: 本步比上一步靠近了多少米(远离为负), 单位 m/step
        #    注意 last_distance_to_goal 与这里必须用同一种距离(都取原始距离), 重置时也要按新起点重新算
        progress = self.last_distance_to_goal - distance_to_goal
        self.last_distance_to_goal = distance_to_goal.clone()

        # 2) 朝向进度: 本步 cos(目标方位角) 增加了多少
        #    这是"转向"的即时奖励. 因为差速小车的角速度不直接出现在任何奖励里,
        #    光靠折扣回报去学"先转向再前进"需要很长的信用分配, 加上这一项后
        #    "朝目标转"立刻就有正反馈(势函数奖励, 总和有界, 不会骗奖励)
        heading_cos = desired_pos_b[:, 0] / (distance_to_goal + eps)
        heading_progress = heading_cos - self.last_heading_cos
        self.last_heading_cos = heading_cos.clone()

        # 3) 速度在"指向目标"方向上的投影, 归一化到 [-1, 1]
        #    车头正对目标全速前进 -> +1; 原地打转 -> 0; 背向目标前进 -> -1
        vel_towards_goal = torch.sum(self._robot.data.root_lin_vel_b[:, :2] * desired_pos_b[:, :2], dim=1)
        vel_towards_goal = vel_towards_goal / (distance_to_goal * self.cfg.max_vel + eps)
        vel_towards_goal = torch.clamp(vel_towards_goal, -1.0, 1.0)

        # 4) 朝向对齐: 用前进速度门控, 避免"原地对着目标不动"也能骗到奖励
        moving_gate = torch.clamp(
            self._robot.data.root_lin_vel_b[:, 0] / self.cfg.align_vel_threshold, 0.0, 1.0
        )
        heading_align = heading_cos * moving_gate

        # 5) 安全距离风险项: 离障碍物越近惩罚越大(特权信息), 给碰撞的硬惩罚加平滑梯度
        if self.cfg.risk_penalty_scale != 0.0:
            d_min = self._nearest_obstacle_distance()
            risk = torch.clamp(1.0 - d_min / self.cfg.risk_margin, min=0.0, max=1.0)
        else:
            risk = torch.zeros(self.num_envs, device=self.device)
        act_smooth = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        time_penalty = torch.ones(self.num_envs, device=self.device)
        rewards = {
            "progress": progress * self.cfg.progress_reward_scale,
            "heading_progress": heading_progress * self.cfg.heading_progress_reward_scale,
            "vel_towards_goal": vel_towards_goal * self.cfg.vel_towards_goal_reward_scale,
            "heading_align": heading_align * self.cfg.heading_align_reward_scale,
            "distance_penalty": -distance_to_goal * self.cfg.distance_penalty_scale,
            "risk": risk * self.cfg.risk_penalty_scale,
            "smooth": act_smooth * self.cfg.smooth_reward_scale,
            "time": self.cfg.time_penalty * time_penalty,
            "collision": self.cfg.collision_penalty * self.collided.float(),
            "time_out": self.cfg.time_out_penalty * self.time_out.float(),
            "reached": self.cfg.reached_reward * self.reached.float(),
        }
        # for key, value in rewards.items():
        #     print(f"{key}: {value.mean().item():.4f}")
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward
    def _compute_collided(self):
        '''碰撞判定. 雷达版本只扫地面(障碍物不在 mesh_prim_paths 里, 且 RayCaster 只支持单个 mesh),
        所以改成两种可用方式:
          - "contact"     接触传感器: 只看与障碍物的接触力(地面接触力常驻, 必须用 filter 滤掉)
          - "privileged"  特权信息: 每步直接用障碍物位置 + 车身 footprint 做 2D 矩形相交
        '''
        if self.cfg.collision_check_mode == "privileged":
            self.collided = self._compute_collided_privileged()
            return
        # 用未过滤的 net_forces_w (含地面反力):
        # 实测平地上正常行驶时本底约 90~130 N, 撞上障碍物瞬间到 2600~3200 N,
        # 所以阈值取 500 N 有 4 倍以上余量. (带 filter_prim_paths_expr 的 force_matrix_w
        # 实测恒为 0, 是 PhysX 过滤 pattern 没匹配上障碍物路径, 所以这里不走过滤版本)
        net_forces = self._contact_sensor.data.net_forces_w               # (N, B, 3)
        max_force = torch.norm(net_forces, dim=-1).max(dim=-1).values     # (N,)
        self.collided = max_force > self.cfg.contact_force_threshold

    # ---------------- 特权信息碰撞判定: 通用 2D 相交工具 ----------------
    def _robot_yaw_cos_sin(self):
        yaw = self._robot.data.heading_w
        return torch.cos(yaw), torch.sin(yaw)

    def _collide_with_aabbs(self, centers, half_extents):
        """车体 OBB vs 一组轴对齐矩形.

        centers: (K,2) 半宽/半长: half_extents: (K,2). 返回 (N,) bool.
        做法: 把矩形中心变到机体系, 矩形在机体系里的投影半宽用 |cos|/|sin| 加权得到.
        """
        if centers.shape[0] == 0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        cos_yaw, sin_yaw = self._robot_yaw_cos_sin()
        half_len, half_wid = self.cfg.robot_footprint_half
        robot_xy = self._robot.data.root_pos_w[:, :2]
        delta = robot_xy.unsqueeze(1) - centers.unsqueeze(0)                 # (N, K, 2)
        dx_b = cos_yaw.unsqueeze(1) * delta[..., 0] + sin_yaw.unsqueeze(1) * delta[..., 1]
        dy_b = -sin_yaw.unsqueeze(1) * delta[..., 0] + cos_yaw.unsqueeze(1) * delta[..., 1]
        proj_x = cos_yaw.abs().unsqueeze(1) * half_extents[:, 0].unsqueeze(0) + \
            sin_yaw.abs().unsqueeze(1) * half_extents[:, 1].unsqueeze(0)
        proj_y = sin_yaw.abs().unsqueeze(1) * half_extents[:, 0].unsqueeze(0) + \
            cos_yaw.abs().unsqueeze(1) * half_extents[:, 1].unsqueeze(0)
        hit = (dx_b.abs() < half_len + proj_x) & (dy_b.abs() < half_wid + proj_y)
        return hit.any(dim=1)

    def _collide_with_circles(self, centers, radii):
        """车体 OBB vs 一组圆(圆心变换到机体系后算点到矩形的距离). centers:(K,2) radii:(K,)"""
        if centers.shape[0] == 0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        cos_yaw, sin_yaw = self._robot_yaw_cos_sin()
        half_len, half_wid = self.cfg.robot_footprint_half
        robot_xy = self._robot.data.root_pos_w[:, :2]
        delta = robot_xy.unsqueeze(1) - centers.unsqueeze(0)                 # (N, K, 2)
        dx_b = (cos_yaw.unsqueeze(1) * delta[..., 0] + sin_yaw.unsqueeze(1) * delta[..., 1]).abs() - half_len
        dy_b = (-sin_yaw.unsqueeze(1) * delta[..., 0] + cos_yaw.unsqueeze(1) * delta[..., 1]).abs() - half_wid
        dist = torch.sqrt(torch.clamp(dx_b, min=0) ** 2 + torch.clamp(dy_b, min=0) ** 2)
        return (dist < radii.unsqueeze(0)).any(dim=1)

    def _nearest_obstacle_distance(self):
        """到最近障碍物表面的距离(m), 用特权信息算: 静态障碍(AABB/圆) + 动态障碍(圆).

        只用于奖励塑形(风险项), 不进入观测. 返回 (N,).
        """
        robot_xy = self._robot.data.root_pos_w[:, :2]
        half_len, half_wid = self.cfg.robot_footprint_half
        d_min = torch.full((self.num_envs,), 1e3, device=self.device)
        # 静态障碍(基础环境随机生成)
        if len(getattr(self, "obstacles", [])) > 0:
            centers = torch.stack([ob.data.root_pos_w[0, :2] for ob in self.obstacles])
            widths = torch.tensor(self.obstacles_width, dtype=torch.float, device=self.device) / 2.0
            d = torch.cdist(robot_xy.unsqueeze(1), centers.unsqueeze(0)).squeeze(1) - \
                torch.sqrt(widths.unsqueeze(0) ** 2 * 2)      # 方柱: 保守用对角线半径
            d_min = torch.minimum(d_min, d.min(dim=1).values)
        # 子类附加的静态几何
        extra = getattr(self, "_extra_static_aabb", None)
        if extra is not None and extra[0].shape[0] > 0:
            d = torch.cdist(robot_xy.unsqueeze(1), extra[0].unsqueeze(0)).squeeze(1) - \
                torch.linalg.norm(extra[1], dim=1).unsqueeze(0)
            d_min = torch.minimum(d_min, d.min(dim=1).values)
        extra_c = getattr(self, "_extra_static_circle", None)
        if extra_c is not None and extra_c[0].shape[0] > 0:
            d = torch.cdist(robot_xy.unsqueeze(1), extra_c[0].unsqueeze(0)).squeeze(1) - extra_c[1].unsqueeze(0)
            d_min = torch.minimum(d_min, d.min(dim=1).values)
        # 动态障碍
        if getattr(self, "dyn_obs_pos", None) is not None and self.dyn_obs_pos.shape[0] > 0:
            d = torch.cdist(robot_xy.unsqueeze(1), self.dyn_obs_pos[:, :2].unsqueeze(0)).squeeze(1) - \
                self.dyn_obs_radius.unsqueeze(0)
            d_min = torch.minimum(d_min, d.min(dim=1).values)
        return d_min

    def _compute_collided_privileged(self):
        '''特权信息碰撞判定: 车身 footprint(矩形) vs
        (1) 静态障碍物 (2) 子类附加的静态几何(如 lobby 的墙/生成的圆柱) (3) 动态障碍物(圆)
        '''
        collided = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # (1) 静态障碍物
        if len(getattr(self, "obstacles", [])) > 0:
            centers = torch.stack([ob.data.root_pos_w[0, :2] for ob in self.obstacles])
            widths = torch.tensor(self.obstacles_width, dtype=torch.float, device=self.device)
            collided |= self._collide_with_aabbs(centers, torch.stack([widths / 2, widths / 2], dim=1))
        # (2) 子类附加静态几何 (AABB 矩形 + 圆柱)
        extra = getattr(self, "_extra_static_aabb", None)
        if extra is not None and extra[0].shape[0] > 0:
            collided |= self._collide_with_aabbs(extra[0], extra[1])
        extra_circle = getattr(self, "_extra_static_circle", None)
        if extra_circle is not None and extra_circle[0].shape[0] > 0:
            collided |= self._collide_with_circles(extra_circle[0], extra_circle[1])
        # (3) 动态障碍物
        if getattr(self, "dyn_obs_pos", None) is not None and self.dyn_obs_pos.shape[0] > 0:
            collided |= self._collide_with_circles(self.dyn_obs_pos[:, :2], self.dyn_obs_radius)
        return collided

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.time_out = time_out

        # force = self._contact_sensor.data.net_forces_w
        # # 任意传感器合力超过阈值即认为撞了
        # self.collided = (force.norm(dim=-1) > 0.5).any(dim=-1)   # shape (N,)
        self._compute_collided()
        reached = (
            torch.linalg.norm(self._robot.data.root_pos_w[:, :2] - self._desired_pos_w[:, :2], dim=1)
            < self.cfg.reach_threshold
        )
        self.reached = reached

        died = self.collided
        died = torch.logical_or(died, reached)
        
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        self.flag = False
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        if not self.eval:
            self.episode_reuslts.extend(self.reached[env_ids].cpu().numpy().tolist())
            success_rate = sum(self.episode_reuslts) / len(self.episode_reuslts)
            # print("recent episode results", self.episode_reuslts)
        else:
            self.episode_reuslts_eval.extend(self.reached[env_ids].cpu().numpy().tolist())
            success_rate = sum(self.episode_reuslts_eval) / len(self.episode_reuslts_eval)
        # print(f"Resetting {len(env_ids)} environments.")
        # self.recorder.plot()
        # print(f"contact forces: {self._contact_sensor.data.net_forces_w[env_ids]}")
        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids][:, :2] - self._robot.data.root_pos_w[env_ids][:, :2], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Episode_Termination/Episode_Step"] = torch.mean(self.episode_length_buf[env_ids].float()).item()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        # extras["Episode_Termination/over_range"] = torch.count_nonzero(self.over_range[env_ids]).item()
        extras["Episode_Termination/reached"] = torch.count_nonzero(self.reached[env_ids]).item()
        extras["Episode_Termination/collided"] = torch.count_nonzero(self.collided[env_ids]).item()
        extras["Metrics/success_rate"] = success_rate
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        # print(f"Target pos: {self._desired_pos_w[env_ids]}")

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        
        self._reset_target(env_ids)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        start_pos = self._reset_start_pos(env_ids)
        start_pos[:, 2] = self.cfg.spawn_height   # 不要埋进地面

        # 起点/终点都是随便撒的, 有概率目标点正好落在起点旁边, 导致"开局即到达"的白送成功,
        # 既污染成功率统计也让 reached 奖励失去意义, 这里重新采样起点保证最小间距
        for _ in range(10):
            too_close = (
                torch.linalg.norm(start_pos[:, :2] - self._desired_pos_w[env_ids, :2], dim=1)
                < self.cfg.min_start_goal_distance
            )
            if not too_close.any():
                break
            start_pos[too_close] = self._reset_start_pos(env_ids[too_close])

        # orientation = self.compute_quaternion_from_to(start_pos, self._desired_pos_w[env_ids])

        default_root_state[:, :3] = start_pos
        # default_root_state[:, 3:7] = orientation

        # if self.eval_in_warehouse:
        #     self._desired_pos_w[env_ids, 0] = -13.
        #     self._desired_pos_w[env_ids, 1] = 28.
        #     default_root_state[:, 0] = 0.
        #     default_root_state[:, 1] = 1.
        # if self.eval_in_hospital:
        #     self._desired_pos_w[env_ids, 0] = 22.
        #     self._desired_pos_w[env_ids, 1] = -1.
        #     default_root_state[:, 0] = 13.
        #     default_root_state[:, 1] = 5.
        #     default_root_state[:, 2] = 0.5
        #     default_root_state[:, 3:7] = torch.tensor([0.9239, 0, 0, -0.3827], device=self.device) # 朝向目标点



        # default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # print(self._terrain.env_origins[env_ids])
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 重置后机器人位置已经被写到 start_pos, 但 self._robot.data 还是旧状态(要等下一次 scene.update 才刷新),
        # 所以这里直接用刚写入的起点和新的目标点算距离, 保证 progress 奖励在回合第一步不会出现跳变
        delta = self._desired_pos_w[env_ids, :2] - start_pos[:, :2]
        distance = torch.linalg.norm(delta, dim=1)
        self.last_distance_to_goal[env_ids] = distance
        # 重置时朝向是默认姿态(机体系 x 轴 = 世界 x 轴), 所以机体系下的目标方位角就是世界系下的
        self.last_heading_cos[env_ids] = delta[:, 0] / (distance + 1e-6)


# @torch.jit.script
# def compute_rewards(
#     rew_scale_alive: float,
#     rew_scale_terminated: float,
#     rew_scale_pole_pos: float,
#     rew_scale_cart_vel: float,
#     rew_scale_pole_vel: float,
#     pole_pos: torch.Tensor,
#     pole_vel: torch.Tensor,
#     cart_pos: torch.Tensor,
#     cart_vel: torch.Tensor,
#     reset_terminated: torch.Tensor,
# ):
#     rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
#     rew_termination = rew_scale_terminated * reset_terminated.float()
#     rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
#     rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
#     rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
#     total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
#     return total_reward