"""
lobby_world_isaaclab.py
将 Gazebo 的 Lobby 评估场景在 Isaac Sim 4.5.0 + Isaac Lab 2.1.1 中重建。

运行:
  ./isaaclab.sh -p scripts/lobby_world_isaaclab.py --sdf /path/to/lobby.world
  ./isaaclab.sh -p scripts/lobby_world_isaaclab.py --headless --sdf lobby.world
"""
import argparse
import math
import os
import xml.etree.ElementTree as ET

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Rebuild the Gazebo Lobby world in Isaac Sim.")
parser.add_argument("--sdf", type=str, default="/home/use_this_new_user/scout_mini/scout_mini/scripts/assets/world/Lobby.world", help="Path to the Gazebo SDF world file.")
parser.add_argument("--box_usd", type=str, default="",
                    help="Optional: cardboard box USD. If empty, boxes spawn as cuboids.")
parser.add_argument("--save_usd", action="store_true", help="Save the rebuilt world as USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import RigidObject, RigidObjectCfg
import torch

DEVICE = "cuda:0" if args_cli.device == "cuda" else args_cli.device

# -----------------------------------------------------------------------------
# 轨迹定义: 根据 lobby world 墙体坐标选出的 6 块净空区域 (见下行注释)
# -----------------------------------------------------------------------------
# 区域A 左下房间:   x 0.0~8.5,  y 0.0~8.0   (wall_2/3 围成)
# 区域B 中下走廊:   x 11.0~17.5, y 0.0~5.0 (wall_6 在 y=5)
# 区域C 右大厅:     x 17.5~31.5, y 0.0~8.0 (wall_7/8 围成)
# 区域D 顶部左区:   x 0.0~13.0, y 12.0~19.0
# 区域E 顶部中区:   x 15.0~22.0, y 16.0~21.0
# 区域F 顶部右区:   x 26.5~31.5, y 12.0~16.0
TRAJECTORIES = [
    ([(1.0, 1.0), (7.5, 1.0), (7.5, 7.5), (1.0, 7.5)], 0.5),                          # A
    ([(11.6, 5.5), (17.1, 5.5), (17.1, 7.5), (11.6, 7.5)], 0.6),                      # B
    ([(0.8, 9.0), (6.5, 9.0), (6.5, 17.5), (0.8, 17.5)], 0.5),                        # C
    ([(12.5, 8.7), (19.5, 8.7), (19.5, 10.8), (12.5, 10.8)], 0.7),                    # D
    ([(21.0, 9.0), (22.3, 9.0), (22.3, 15.0), (21.0, 15.0)], 0.55),                   # E
    ([(9.0, 16.2), (22.0, 16.2), (22.0, 20.5), (13.5, 20.5), (13.5, 16.2)], 0.6),     # F
]

OBSTACLE_RADIUS = 0.2    # 类似行人的圆柱半径
OBSTACLE_HEIGHT = 1.7    # 类似行人的高度
OBSTACLE_COLORS = [      # 每个障碍物不同颜色, 便于区分
    (0.9, 0.2, 0.2), (0.2, 0.6, 0.9), (0.2, 0.8, 0.3),
    (0.9, 0.7, 0.1), (0.6, 0.3, 0.8), (0.9, 0.5, 0.1),
    (0.2, 0.7, 0.7), (0.8, 0.3, 0.5), (0.5, 0.5, 0.9), (0.4, 0.8, 0.4),
]

class Trajectory:
    """闭合折线轨迹: 等速运动, 按弧长插值, 朝向沿运动方向。"""

    def __init__(self, waypoints: list[tuple[float, float]], speed: float):
        assert len(waypoints) >= 2
        self.pts = waypoints + [waypoints[0]]        # 闭合
        self.speed = speed
        seg = [math.dist(self.pts[i], self.pts[i + 1]) for i in range(len(self.pts) - 1)]
        self.cum = [0.0]
        for s in seg:
            self.cum.append(self.cum[-1] + s)
        self.total = self.cum[-1]

    def pose_at(self, t: float) -> tuple[float, float, float]:
        """返回 t 时刻的 (x, y, yaw)。"""
        s = (self.speed * t) % self.total
        # 找到所在线段
        i = 0
        while self.cum[i + 1] < s:
            i += 1
        (x1, y1), (x2, y2) = self.pts[i], self.pts[i + 1]
        r = (s - self.cum[i]) / max(self.cum[i + 1] - self.cum[i], 1e-9)
        x, y = x1 + r * (x2 - x1), y1 + r * (y2 - y1)
        yaw = math.atan2(y2 - y1, x2 - x1)
        return x, y, yaw


def yaw_to_quat_wxyz(yaw: float):
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def spawn_obstacles(num: int) -> list[tuple[RigidObject, Trajectory]]:
    """生成 num 个运动学障碍物并绑定轨迹。"""
    obstacles = []
    n_traj = len(TRAJECTORIES)
    for k in range(num):
        waypoints, speed = TRAJECTORIES[k % n_traj]
        traj = Trajectory(waypoints, speed)
        x0, y0, yaw0 = traj.pose_at(0.0)
        cfg = RigidObjectCfg(
            prim_path=f"/World/Obstacles/obstacle_{k}",
            spawn=sim_utils.CylinderCfg(
                radius=OBSTACLE_RADIUS,
                height=OBSTACLE_HEIGHT,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=OBSTACLE_COLORS[k % len(OBSTACLE_COLORS)],
                    roughness=0.8,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(x0, y0, OBSTACLE_HEIGHT / 2.0), rot=yaw_to_quat_wxyz(yaw0)
            ),
        )
        obstacles.append((RigidObject(cfg), traj))
        print(f"[INFO] obstacle_{k}: traj={waypoints}, v={speed} m/s, start=({x0:.2f},{y0:.2f})")
    return obstacles


def drive_obstacles(obstacles, t: float, dt: float):
    """每个 sim step 调用: 按轨迹写入所有障碍物位姿。"""
    for obj, traj in obstacles:
        x, y, yaw = traj.pose_at(t)
        pose = torch.tensor([[x, y, OBSTACLE_HEIGHT / 2.0, *yaw_to_quat_wxyz(yaw)]],
                            device=DEVICE)
        obj.write_root_pose_to_sim(pose)
        obj.write_data_to_sim()  # 仅写入 root pose, 不写入 velocity/acceleration
        obj.update(dt)

def yaw_to_quat_wxyz(yaw: float):
    """SDF 中只有绕 Z 的偏航角; Isaac Lab 四元数顺序为 (w, x, y, z)。"""
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def parse_sdf(path: str):
    """解析 SDF, 返回 (walls, boxes) 列表。
    元素: dict(name, pos=(x,y,z), yaw, size=(sx,sy,sz), mass)。
    XML 注释(被注释的 grey_wall_56~58)被 ElementTree 自动忽略。
    """
    world = ET.parse(path).getroot().find("world")
    walls, boxes = [], []
    for model in world.findall("model"):
        name = model.attrib.get("name", "model")
        if "cardboard_box" in name:
            continue
        link = model.find("link")
        pose = model.find("pose")
        pose_el = pose if pose is not None else link.find("pose")   # 纸箱在 model 级, 墙在 link 级
        x, y, z, roll, pitch, yaw = [float(v) for v in pose_el.text.split()]
        size = tuple(float(v) for v in link.find("./collision/geometry/box/size").text.split())
        mass_el = link.find("./inertial/mass")
        mass = float(mass_el.text) if mass_el is not None else 0.0
        is_static = model.find("static") is not None and model.find("static").text.strip() == "1"
        entry = dict(name=name, pos=(x, y, z), yaw=yaw, size=size, mass=mass)
        if "cardboard_box" in name:
            boxes.append(entry)
        elif is_static:
            walls.append(entry)
    return walls, boxes


def create_scene(sdf_path: str, box_usd_path: str = ""):
    walls, boxes = parse_sdf(sdf_path)
    print(f"[INFO] Parsed {len(walls)} walls and {len(boxes)} cardboard boxes.")

    # sim_utils.create_prim("/World/Lobby", "Xform")

    # Ground plane (对应 model://ground_plane)
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())

    # Sun -> 穹顶光近似
    dome_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    dome_cfg.func("/World/Light", dome_cfg)

    # ---- 墙体: 静态运动学刚体 (不受碰撞/重力影响, 但保留碰撞体) ----
    wall_spawn_cfg = sim_utils.CuboidCfg(
        size=(1.0, 1.0, 1.0),  # 占位, 每面墙替换
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.5, 0.5, 0.5), roughness=0.9, metallic=0.0),
    )
    for wall in walls:
        cfg = wall_spawn_cfg.copy()
        cfg.size = wall["size"]
        cfg.func(f"/World/Lobby/{wall['name']}", cfg,
                 translation=wall["pos"], orientation=yaw_to_quat_wxyz(wall["yaw"]))

    # ---- 纸箱: 动态刚体 (mass=2kg, 摩擦~1, 纸板色) ----
    box_entities = {}
    for box in boxes:
        if box_usd_path and os.path.exists(box_usd_path):
            spawn_cfg = sim_utils.UsdFileCfg(usd_path=box_usd_path)
        else:
            spawn_cfg = sim_utils.CuboidCfg(
                size=box["size"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=box["mass"]),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.55, 0.38, 0.24), roughness=1.0),
            )
        obj_cfg = RigidObjectCfg(
            prim_path=f"/World/Lobby/{box['name']}",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=box["pos"], rot=yaw_to_quat_wxyz(box["yaw"])),
        )
        box_entities[box["name"]] = RigidObject(obj_cfg)
    return box_entities


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))

    # 复现 Gazebo world 中的俯视相机 (13.4, 13.5, 37.9)
    sim.set_camera_view(eye=(13.42, 13.47, 37.87), target=(13.42, 13.47, 0.0))

    create_scene(args_cli.sdf, args_cli.box_usd)

    # 2. 生成动态障碍物
    obstacles = spawn_obstacles(10)

    sim.reset()
    print("[INFO] Lobby + dynamic obstacles running...")

    # 3. 仿真循环: 每个 step 驱动障碍物
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    while simulation_app.is_running():
        drive_obstacles(obstacles, sim_time, sim_dt)
        sim.step()
        sim_time += sim_dt


if __name__ == "__main__":
    main()
    simulation_app.close()