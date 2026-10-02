# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""固定评估环境基类.

从无人机的 generated_env / lobby_env 迁移过来, 按 scout_mini(2D 差速小车) 的
动作空间/奖励/碰撞判定改写. 两个评估环境共用这里的:
  - 固定的 20 组起终点配对 (cfg.use_fixed_pairs 打开时生效)
  - meta 指定的两点往返(ping-pong)动态障碍物
静态场景(生成的圆柱/方柱、lobby 的墙)由子类各自实现.
"""

from __future__ import annotations

import os

from isaaclab.utils import configclass

from .scout_mini_env import ScoutMiniEnv


def find_project_root() -> str:
    """从当前文件往上找带 scripts/assets 的目录(项目根)."""
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "scripts", "assets")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))


class ScoutMiniEvalEnvBase(ScoutMiniEnv):
    """评估环境基类：固定起终点 + 往返动态障碍物."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # 必须在 super().__init__ 之前读 meta: _setup_scene 里要按 meta 建场景
        self._load_meta(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        if self.cfg.use_fixed_pairs:
            self._sanitize_fixed_pairs()
            self.init_fixed_pairs(self._fixed_starts, self._fixed_goals)
        print(
            f"[INFO] {type(self).__name__}: 静态几何 {len(self._static_boxes)} 方柱 + "
            f"{len(self._static_cylinders)} 圆柱, 动态障碍物 {len(self._dynamic_specs)} 个 (tier {self.cfg.dynamic_obstacle_tier}), "
            f"固定起终点 {len(self._fixed_starts)} 组"
        )

    # ---- 子类实现 ----
    def _load_meta(self, cfg):
        raise NotImplementedError

    # ---- 起终点: 固定配对 ----
    def _sanitize_fixed_pairs(self):
        """把落在障碍物里的固定起终点沿场地边界方向挪开.

        无人机版按 ±(map/2+0.3) 生成这些点, 但 meta 里的静态障碍物可能越界
        (实测 generated_env 有 2/20 组起点与圆柱/方柱重叠 -> 出生即碰撞),
        小车车身 0.62m 比无人机的点更"胖", 所以这里显式检查并挪开.
        """
        import torch

        if getattr(self, "_extra_static_aabb", None) is None:
            return
        centers_a, half_a = self._extra_static_aabb
        centers_c, radii_c = self._extra_static_circle
        half_len, half_wid = self.cfg.robot_footprint_half
        clearance = self.cfg.start_clearance   # 还要额外留出的余量
        half_len = half_len + clearance
        half_wid = half_wid + clearance

        def collides(x, y, yaw=0.0):
            pt = torch.tensor([[x, y]], dtype=torch.float, device=self.device)
            if centers_a.shape[0] > 0:
                d = (centers_a - pt).abs()
                if bool((((d[:, 0] - half_a[:, 0]) < half_len) & ((d[:, 1] - half_a[:, 1]) < half_wid)).any()):
                    return True
            if centers_c.shape[0] > 0:
                dd = torch.linalg.norm(centers_c - pt, dim=1)
                if bool((dd < radii_c + max(half_len, half_wid)).any()):
                    return True
            return False

        import math as _math

        for name, pts in (("start", self._fixed_starts), ("goal", self._fixed_goals)):
            for i, (x, y) in enumerate(pts):
                if not collides(x, y):
                    continue
                # 沿 8 个方向 × 由近到远试: 只沿"从场地中心往外"一个方向会在贴墙的房间里
                # 永远躲不开(实测 Lobby 的 20 组里 8 组挪不开 -> 出生即碰撞)
                cands = []
                for ang in range(0, 360, 45):
                    cands.append((_math.cos(_math.radians(ang)), _math.sin(_math.radians(ang))))
                done = False
                for r in [0.25 * k for k in range(1, 13)]:
                    for ux, uy in cands:
                        nx, ny = x + ux * r, y + uy * r
                        if not collides(nx, ny):
                            pts[i] = (nx, ny)
                            print(f"[INFO] 固定{name}点 {i} 与障碍物重叠, 已从 ({x:.2f},{y:.2f}) 挪到 ({nx:.2f},{ny:.2f})")
                            done = True
                            break
                    if done:
                        break
                if not done:
                    print(f"[WARN] 固定{name}点 {i} ({x:.2f},{y:.2f}) 挪不开, 保持原样")

    def _reset_target(self, env_ids):
        if self.cfg.use_fixed_pairs and self._reset_target_fixed(env_ids):
            return
        super()._reset_target(env_ids)

    def _reset_start_pos(self, env_ids):
        if self.cfg.use_fixed_pairs:
            pos = self._reset_start_pos_fixed(env_ids)
            if pos is not None:
                return pos
        return super()._reset_start_pos(env_ids)

    # ---- 动态障碍物: 两点往返 ----
    def move_dynamic_obstacles(self):
        if getattr(self, "_dyn_x1", None) is not None and len(self.dyn_obs_list) > 0:
            self.move_dynamic_obstacles_pingpong()
        else:
            super().move_dynamic_obstacles()

    def generate_dynamic_obstacles(self):
        self.spawn_dynamic_obstacles_from_specs(self._dynamic_specs)

    # ---- 静态几何 -> 碰撞判定用的 AABB / 圆 ----
    def _build_static_collision_geometry(self):
        """把 meta 里的静态几何缓存成碰撞判定用的张量.

        方柱: 轴对齐矩形(半宽/半长); 圆柱: 圆(半径).
        """
        import torch

        if self._static_boxes:
            centers = torch.tensor([[b["x"], b["y"]] for b in self._static_boxes], dtype=torch.float, device=self.device)
            half = torch.tensor(
                [[b.get("hw", b.get("sx", 0.0) / 2.0), b.get("hd", b.get("sy", 0.0) / 2.0)] for b in self._static_boxes],
                dtype=torch.float, device=self.device)
            self._extra_static_aabb = (centers, half)
        else:
            self._extra_static_aabb = (
                torch.zeros((0, 2), device=self.device), torch.zeros((0, 2), device=self.device))
        if self._static_cylinders:
            centers = torch.tensor([[c["x"], c["y"]] for c in self._static_cylinders], dtype=torch.float, device=self.device)
            radii = torch.tensor([c["r"] for c in self._static_cylinders], dtype=torch.float, device=self.device)
            self._extra_static_circle = (centers, radii)
        else:
            self._extra_static_circle = (
                torch.zeros((0, 2), device=self.device), torch.zeros((0,), device=self.device))
        # 基类的 self.obstacles 机制在评估环境里不用(几何全部走 _extra_*)
        self.obstacles = []
        self.obstacles_width = []
