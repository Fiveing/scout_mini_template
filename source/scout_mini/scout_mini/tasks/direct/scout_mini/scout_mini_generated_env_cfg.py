# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""generated_env 评估环境配置 (从小车基础环境派生).

复刻 ROS1 的 20x20 generated_env 场景:
  - 静态: 30 个圆柱 + 30 个方柱, 位置/尺寸来自 meta json (无随机)
  - 动态: N 个圆柱在两点之间匀速往返, N ∈ {4, 6, 8, 12} 四档 (嵌套子集)
  - 起终点: 固定的 20 组, 落在 20x20 障碍区外侧 ±(map/2+0.3) 处
"""

from __future__ import annotations

import os

from isaaclab.utils import configclass

from .scout_mini_env_cfg import ScoutMiniEnvCfg
from .scout_mini_eval_env_base import find_project_root


@configclass
class ScoutMiniGeneratedEnvCfg(ScoutMiniEnvCfg):
    # 难度档: 动态障碍物个数, 必须是 4/6/8/12 之一
    dynamic_obstacle_tier = 4

    # 用固定的 20 组起终点(评估环境); 关掉则退回基础环境的随机起终点(可用于课程学习)
    use_fixed_pairs = True

    # 场景资源
    asset_dir = os.path.join(find_project_root(), "scripts", "assets", "generated_env")
    meta_path = ""  # 留空 -> generated_env_static_dynamic{tier}_meta.json

    # 20x20 场地: 起终点在 ±10.3 的边界外侧
    obstacle_map_range = 20.0
    num_obstacles = 0            # 静态障碍全部来自 meta, 不用基础环境的随机生成
    difficulty = "none"
    min_start_goal_distance = 0.0

    # 关掉"离障碍物过近"的风险项: 实测在密集场景里这项每回合累计 -700~-900,
    # 而到达奖励只有 +300, 于是最优策略变成"不敢开进去", 成功率恒为 0.
    # (稀疏的基础环境里它是有用的, 这里必须关掉)
    risk_penalty_scale = 0.0

    # 路径最长约 29m(对角), 2m/s 需要约 15s, 留足冗余
    episode_length_s = 30.0
    time_penalty = -0.10           # 30s=1500 步, 时间惩罚总量仍约 -150(与基础环境 20s 时一致)
