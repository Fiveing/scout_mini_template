# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""lobby_env 评估环境配置 (从小车基础环境派生).

复刻 Gazebo 的 Lobby 场景:
  - 静态: 80 面墙 (位置/尺寸来自 lobby_meta_{tier}.json, 由 Lobby.world 解析而来)
  - 动态: N 个类似行人的圆柱在两点之间匀速往返, N ∈ {4, 6, 8, 12} 四档
  - 起终点: 固定的 20 组 (室内房间到房间, 已用 BFS 验证过地面可达)
"""

from __future__ import annotations

import os

from isaaclab.utils import configclass

from .scout_mini_env_cfg import ScoutMiniEnvCfg
from .scout_mini_eval_env_base import find_project_root


@configclass
class ScoutMiniLobbyEnvCfg(ScoutMiniEnvCfg):
    # 动态障碍物档位: 4/6/8/12
    dynamic_obstacle_tier = 4

    # 用固定的 20 组起终点
    use_fixed_pairs = True

    # 场景资源
    asset_dir = os.path.join(find_project_root(), "scripts", "assets", "lobby")
    meta_path = ""        # 留空 -> lobby_meta_{tier}.json
    sdf_path = os.path.join(find_project_root(), "scripts", "assets", "world", "Lobby.world")  # 仅备查

    obstacle_map_range = 32.0     # Lobby 实际范围 x∈[-12,33.5], y∈[-0.1,24.1]
    num_obstacles = 0
    difficulty = "none"
    min_start_goal_distance = 0.0

    # 关掉"离障碍物过近"的风险项: 实测在密集场景里这项每回合累计 -700~-900,
    # 而到达奖励只有 +300, 于是最优策略变成"不敢开进去", 成功率恒为 0.
    # (稀疏的基础环境里它是有用的, 这里必须关掉)
    risk_penalty_scale = 0.0

    # 房间到房间最远约 23m(2m/s 约 12s), 加上避障留足余量
    episode_length_s = 30.0
    time_penalty = -0.10
