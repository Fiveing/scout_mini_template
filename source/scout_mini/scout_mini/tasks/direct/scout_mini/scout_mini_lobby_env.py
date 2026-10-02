# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""lobby_env 评估环境 (2D 差速小车版).

对应无人机工程的 lobby_env.py:
  - 静态场景 = Gazebo Lobby 的 80 面墙 (来自 lobby_meta_{tier}.json, 由 Lobby.world 解析)
  - 动态障碍物 = 类似行人的圆柱(r≈0.35, h≈1.7), 在两点之间匀速往返
  - 固定 20 组起终点 (房间到房间), 已用占据栅格 BFS 验证过地面可达
动作/奖励/碰撞判定沿用 scout_mini (2D 速度动作 + privileged 特权碰撞, 墙也计入碰撞).
"""

from __future__ import annotations

import json
import math
import os

import isaaclab.sim as sim_utils

from .scout_mini_lobby_env_cfg import ScoutMiniLobbyEnvCfg
from .scout_mini_eval_env_base import ScoutMiniEvalEnvBase

# 20 组固定起终点(原始 Lobby.world 坐标系, 由 SDF 占据栅格 + A* 生成)
# 从无人机工程 lobby_env.py 原样迁移; 已用 BFS(墙按车半径 0.36 膨胀) 验证过 20/20 地面可达
START_GOAL_PAIRS = [
    (0.6, 4.8), (16.7, 7.7), (1.2, 14.0), (6.6, 0.6), (1.8, 17.6),
    (0.6, 6.6), (18.8, 20.0), (1.8, 13.4), (0.6, 6.6), (4.8, 17.0),
    (7.8, 3.6), (0.6, 4.8), (1.2, 3.6), (3.6, 6.0), (22.5, 10.8),
    (11.9, 7.1), (6.6, 0.6), (21.3, 13.8), (15.5, 5.9), (3.6, 9.2),
]
GOALS = [
    (21.9, 12.6), (1.8, 17.6), (21.2, 16.4), (17.5, 10.8), (21.9, 13.2),
    (15.8, 20.0), (8.7, 10.8), (20.7, 9.0), (19.4, 17.6), (21.2, 19.4),
    (22.5, 10.8), (1.8, 17.6), (16.7, 5.9), (1.8, 17.6), (21.2, 19.4),
    (22.5, 9.6), (2.4, 3.0), (18.8, 20.0), (4.8, 13.4), (4.2, 16.4),
]


class ScoutMiniLobbyEnv(ScoutMiniEvalEnvBase):
    cfg: ScoutMiniLobbyEnvCfg

    # ------------------------------------------------------------------ meta
    def _load_meta(self, cfg):
        tier = int(cfg.dynamic_obstacle_tier)
        if tier not in (4, 6, 8, 12):
            raise ValueError(f"dynamic_obstacle_tier 必须是 4/6/8/12 之一, 收到 {tier}")
        meta_path = cfg.meta_path or os.path.join(cfg.asset_dir, f"lobby_meta_{tier}.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"lobby meta 不存在: {meta_path}")
        with open(meta_path) as f:
            meta = json.load(f)
        self._static_walls = meta["walls"]              # {x,y,yaw,sx,sy,zmin,zmax,name,outer}
        self._dynamic_specs = meta["dynamic"][:tier]    # {id,x1,y1,x2,y2,r,h,v}
        cfg.dynamic_obstacle_nums = len(self._dynamic_specs)
        # 墙体是方柱, 方柱与圆柱分开缓存(碰撞判定用)
        self._static_boxes = [
            {"x": w["x"], "y": w["y"], "hw": w["sx"] / 2.0, "hd": w["sy"] / 2.0, "h": w["zmax"] - w["zmin"],
             "yaw": w.get("yaw", 0.0)}
            for w in self._static_walls
        ]
        self._static_cylinders = []
        self._fixed_starts = list(START_GOAL_PAIRS)
        self._fixed_goals = list(GOALS)

    # ------------------------------------------------------------------ 场景
    def _generate_obstacles(self):
        """按 meta 生成 lobby 的 80 面墙 (静态 kinematic 刚体, 贴地)."""
        for w in self._static_walls:
            height = w["zmax"] - w["zmin"]
            cfg = sim_utils.CuboidCfg(
                size=(w["sx"], w["sy"], height),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), roughness=0.9, metallic=0.0),
            )
            yaw = w.get("yaw", 0.0)
            yaw = float(yaw)
            quat = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
            cfg.func(
                f"/World/Obstacles/{w['name']}", cfg,
                translation=(w["x"], w["y"], (w["zmax"] + w["zmin"]) / 2.0),
                orientation=quat,
            )
        self._build_static_collision_geometry()

    def _build_static_collision_geometry(self):
        """墙的 yaw 只有 0/±90/180 度, 旋转后的外接矩形就是精确 AABB."""
        import torch

        if self._static_boxes:
            centers = torch.tensor([[b["x"], b["y"]] for b in self._static_boxes], dtype=torch.float, device=self.device)
            half = torch.tensor(
                [
                    [
                        abs(math.cos(b["yaw"])) * b["hw"] + abs(math.sin(b["yaw"])) * b["hd"],
                        abs(math.sin(b["yaw"])) * b["hw"] + abs(math.cos(b["yaw"])) * b["hd"],
                    ]
                    for b in self._static_boxes
                ],
                dtype=torch.float, device=self.device,
            )
            self._extra_static_aabb = (centers, half)
        else:
            self._extra_static_aabb = (
                torch.zeros((0, 2), device=self.device), torch.zeros((0, 2), device=self.device))
        self._extra_static_circle = (
            torch.zeros((0, 2), device=self.device), torch.zeros((0,), device=self.device))
        self.obstacles = []
        self.obstacles_width = []
