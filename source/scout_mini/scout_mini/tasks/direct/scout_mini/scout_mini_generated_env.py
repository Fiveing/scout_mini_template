# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""generated_env 评估环境 (2D 差速小车版).

对应无人机工程的 generated_env_dynamic_env.py:
  - 静态场景来自 meta json (30 圆柱 + 30 方柱, 确定性布局)
  - 动态障碍物在两点间匀速往返, 四档 4/6/8/12 取前 N 项
  - 固定 20 组起终点, 落在障碍区外侧四个边的中点上
动作/奖励/碰撞判定全部沿用 scout_mini 的 (2D 速度动作 + privileged 特权碰撞).
"""

from __future__ import annotations

import json
import os

import torch

import isaaclab.sim as sim_utils

from .scout_mini_env_cfg import ScoutMiniEnvCfg
from .scout_mini_generated_env_cfg import ScoutMiniGeneratedEnvCfg
from .scout_mini_eval_env_base import ScoutMiniEvalEnvBase


class ScoutMiniGeneratedEnv(ScoutMiniEvalEnvBase):
    cfg: ScoutMiniGeneratedEnvCfg

    # ------------------------------------------------------------------ meta
    def _load_meta(self, cfg):
        tier = int(cfg.dynamic_obstacle_tier)
        if tier not in (4, 6, 8, 12):
            raise ValueError(f"dynamic_obstacle_tier 必须是 4/6/8/12 之一, 收到 {tier}")
        meta_path = cfg.meta_path or os.path.join(cfg.asset_dir, f"generated_env_static_dynamic{tier}_meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"generated_env meta 不存在: {meta_path}")
        with open(meta_path) as f:
            meta = json.load(f)
        self._static_cylinders = meta["static"]["cylinders"]      # {x, y, r, h}
        self._static_boxes = meta["static"]["boxes"]              # {x, y, hw, hd, h}
        # 四档是嵌套子集: 取前 tier 个动态障碍物
        self._dynamic_specs = meta["dynamic"][:tier]
        cfg.dynamic_obstacle_nums = len(self._dynamic_specs)
        # 固定起终点: 20 组, 分布在 20x20 场地四条边外侧
        self._fixed_starts, self._fixed_goals = self._create_eval_start_goal_pairs(cfg)

    def _create_eval_start_goal_pairs(self, cfg):
        """20 组固定起终点: 从场地一侧外侧出发, 到另一侧外侧.

        (与无人机版一致: side_step 把每条边分成 6 份, 每条边 5 个点)
        """
        pair_count = 20
        half_range = cfg.obstacle_map_range / 2.0 + 0.3
        side_step = cfg.obstacle_map_range / (pair_count // 4 + 1)
        starts, goals = [], []
        for idx in range(pair_count):
            side = idx // 5
            offset = -half_range + (idx % 5 + 1) * side_step
            if side == 0:
                start, target = (-half_range, offset), (offset, half_range)
            elif side == 1:
                start, target = (half_range, offset), (offset, -half_range)
            elif side == 2:
                start, target = (offset, -half_range), (half_range, offset)
            else:
                start, target = (offset, half_range), (-half_range, offset)
            starts.append(start)
            goals.append(target)
        return starts, goals

    # ------------------------------------------------------------------ 场景
    def _generate_obstacles(self):
        """按 meta 生成 30 圆柱 + 30 方柱 (静态, 不需要逐帧写位姿)."""
        for i, c in enumerate(self._static_cylinders):
            cfg = sim_utils.CylinderCfg(
                radius=c["r"],
                height=c["h"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6), metallic=0.0),
            )
            cfg.func(f"/World/Obstacles/cylinder_{i}", cfg, translation=(c["x"], c["y"], c["h"] / 2.0))
        for i, b in enumerate(self._static_boxes):
            cfg = sim_utils.CuboidCfg(
                size=(b["hw"] * 2.0, b["hd"] * 2.0, b["h"]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6), metallic=0.0),
            )
            cfg.func(f"/World/Obstacles/box_{i}", cfg, translation=(b["x"], b["y"], b["h"] / 2.0))
        self._build_static_collision_geometry()
