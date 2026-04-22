"""
rrt_star_dubins_3d.py
---------------------
3D RRT* path planner for a fixed-wing UAV using true 3D Dubins paths.

Requires (in the same folder):
    dubins3d.py   - pure-Python port of Dubins3D.jl
    rrt_star.py   - base RRT* class  (from PythonRobotics)

Obstacles are vertical cylinders with a bottom and top Z height,
simulating buildings in an urban environment.

Configuration: [x, y, z, heading_rad, pitch_rad]
"""

import copy
import math
import random
import time
import sys
import pathlib
import csv

import matplotlib.pyplot as plt
import numpy as np

# ── PythonRobotics paths ────────────────────────────────────────────────────
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from rrt_star import RRTStar
from .utils.plot import plot_arrow

# ── Our 3D Dubins engine ────────────────────────────────────────────────────
from continuous_path_planner import dubins3d
from continuous_path_planner.dubins3d import DubinsManeuver3D, compute_sampling

show_animation = True


# ============================================================================
# RRTStarDubins3D
# ============================================================================

class RRTStarDubins3D(RRTStar):

    # ── Node ─────────────────────────────────────────────────────────────────

    class Node(RRTStar.Node):
        def __init__(self, x, y, z, yaw, pitch=0.0):
            super().__init__(x, y)
            self.z        = z
            self.yaw      = yaw
            self.pitch    = pitch
            self.path_z   = []
            self.path_yaw = []

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(self,
                 start,          # [x, y, z, yaw, pitch]
                 goal,           # [x, y, z, yaw, pitch]
                 obstacle_list,  # list of (x, y, radius, z_bottom, z_top)
                 rand_area,      # [min_xy, max_xy]
                 rand_z,         # [min_z,  max_z]
                 goal_sample_rate=10,
                 max_iter=300,
                 connect_circle_dist=50.0,
                 robot_radius=0.0,
                 turn_radius=1.0,
                 pitch_lims=None):
        """Initialize RRT* planner for fixed-wing Dubins 3D.

        Args:
            start: [x, y, z, yaw, pitch]
            goal: [x, y, z, yaw, pitch]
            obstacle_list: list of vertical cylinder obstacles
            rand_area: [min_xy, max_xy]
            rand_z: [min_z, max_z]
            goal_sample_rate: percent chance to sample goal
            max_iter: max iterations
            connect_circle_dist: RRT* neighbor radius
            robot_radius: clearance radius for collision checking
            turn_radius: minimum fixed-wing turn radius (meters)
            pitch_lims: pitch limits [min_rad, max_rad]
        """

        self.start = self.Node(start[0], start[1], start[2], start[3], start[4])
        self.end   = self.Node(goal[0],  goal[1],  goal[2],  goal[3],  goal[4])

        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.min_z    = rand_z[0]
        self.max_z    = rand_z[1]

        self.goal_sample_rate    = goal_sample_rate
        self.max_iter            = max_iter
        self.obstacle_list       = obstacle_list
        self.connect_circle_dist = connect_circle_dist
        self.robot_radius        = robot_radius

        self.turn_radius = turn_radius
        self.rhomin      = turn_radius
        self.pitch_lims  = pitch_lims if pitch_lims else \
                          [np.deg2rad(-15.0), np.deg2rad(20.0)]

        # Goal thresholds
        self.goal_xy_th  = 1.5
        self.goal_z_th   = 1.5
        self.goal_yaw_th = np.deg2rad(45.0)

        self.cone_angle = np.pi

    # ── Planning loop ─────────────────────────────────────────────────────────

    def planning(self, animation=True, search_until_max_iter=True):
        self.node_list = [self.start]

        for i in range(self.max_iter):
            print(f"Iter: {i:4d}  nodes: {len(self.node_list)}")
            sys.stdout.flush()

            rnd         = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node    = self.steer(self.node_list[nearest_ind], rnd)

            if new_node and self.check_collision(
                    new_node, self.obstacle_list, self.robot_radius):

                near_indexes = self.find_near_nodes(new_node)
                new_node     = self.choose_parent(new_node, near_indexes)

                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_indexes)

            if animation and i % 5 == 0:
                self.draw_graph(rnd)

            if (not search_until_max_iter) and new_node:
                last_index = self.search_best_goal_node()
                if last_index is not None:
                    return self.generate_final_course(last_index)

        print("Reached max iterations")
        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index)

        print("Cannot find path")
        return None

    # ── Dubins 3D steer ───────────────────────────────────────────────────────

    def _node_to_qi(self, node):
        return [node.x, node.y, node.z, node.yaw, node.pitch]

    def steer(self, from_node, to_node):
        try:
            maneuver = DubinsManeuver3D(
                self._node_to_qi(from_node),
                self._node_to_qi(to_node),
                self.rhomin,
                self.pitch_lims)
        except Exception:
            return None

        samples = compute_sampling(maneuver, number_of_samples=50)
        if len(samples) <= 1:
            return None

        new_node          = copy.deepcopy(from_node)
        new_node.x        = samples[-1][0]
        new_node.y        = samples[-1][1]
        new_node.z        = samples[-1][2]
        new_node.yaw      = samples[-1][3]
        new_node.pitch    = samples[-1][4]
        new_node.path_x   = [s[0] for s in samples]
        new_node.path_y   = [s[1] for s in samples]
        new_node.path_z   = [s[2] for s in samples]
        new_node.path_yaw = [s[3] for s in samples]
        new_node.cost    += maneuver.length
        new_node.parent   = from_node
        return new_node

    def calc_new_cost(self, from_node, to_node):
        try:
            maneuver = DubinsManeuver3D(
                self._node_to_qi(from_node),
                self._node_to_qi(to_node),
                self.rhomin,
                self.pitch_lims)
            return from_node.cost + maneuver.length
        except Exception:
            return float("inf")

    # ── Sampling ──────────────────────────────────────────────────────────────

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            d  = random.uniform(0.0, self.max_rand - self.min_rand)
            th = self.start.yaw + random.uniform(-self.cone_angle, self.cone_angle)

            ox = self.start.x + d * math.cos(th)
            oy = self.start.y + d * math.sin(th)
            ox = min(max(self.min_rand, ox), self.max_rand)
            oy = min(max(self.min_rand, oy), self.max_rand)
            oz = random.uniform(self.min_z, self.max_z)

            pitch = random.uniform(self.pitch_lims[0], self.pitch_lims[1])
            rnd   = self.Node(ox, oy, oz,
                              random.uniform(-math.pi, math.pi),
                              pitch)
        else:
            rnd = self.Node(self.end.x, self.end.y, self.end.z,
                            self.end.yaw, self.end.pitch)
        return rnd

    # ── Collision check (3D cylinders) ────────────────────────────────────────

    def check_collision(self, node, obstacle_list, robot_radius):
        if node is None:
            return False

        path_z = node.path_z if node.path_z else [node.z]

        for ox, oy, radius, z_bottom, z_top in obstacle_list:
            for x, y, z in zip(node.path_x, node.path_y, path_z):
                if math.hypot(x - ox, y - oy) <= radius + robot_radius:
                    if z_bottom <= z <= z_top:
                        return False   # collision
        return True

    # ── Goal search ───────────────────────────────────────────────────────────

    def search_best_goal_node(self):

        close_xy = [
            i for i, node in enumerate(self.node_list)
            if self.calc_dist_to_goal(node.x, node.y) <= self.goal_xy_th
        ]

        print(f"\n  Nodes close in XY:  {len(close_xy)}")

        close_xyz = [
            i for i in close_xy
            if abs(self.node_list[i].z - self.end.z) <= self.goal_z_th
        ]

        print(f"  Nodes close in Z:   {len(close_xyz)}")

        final_goal_indexes = [
            i for i in close_xyz
            if abs(self.node_list[i].yaw - self.end.yaw) <= self.goal_yaw_th
        ]

        print(f"  Nodes close in yaw: {len(final_goal_indexes)}")

        if not final_goal_indexes:
            # Show the closest node so you can see how far off it is
            if close_xy:
                best = min(close_xy, key=lambda i: self.node_list[i].cost)
                n = self.node_list[best]
                print(f"  Closest node — z: {n.z:.2f} (need {self.end.z:.2f}), "
                    f"yaw: {np.rad2deg(n.yaw):.1f}° "
                    f"(need {np.rad2deg(self.end.yaw):.1f}°)")
            return None

        return min(final_goal_indexes, key=lambda i: self.node_list[i].cost)

    # ── Final path extraction ─────────────────────────────────────────────────

    def generate_final_course(self, goal_index):
        node  = self.node_list[goal_index]
        chain = [node]
        while node.parent:
            node = node.parent
            chain.append(node)
        chain.reverse()

        final_path = [[self.start.x, self.start.y, self.start.z]]

        for node in chain[1:]:
            path_z = node.path_z if node.path_z else [node.z] * len(node.path_x)
            for x, y, z in zip(node.path_x, node.path_y, path_z):
                if [x, y, z] != final_path[-1]:
                    final_path.append([x, y, z])

        if final_path[-1] != [self.end.x, self.end.y, self.end.z]:
            final_path.append([self.end.x, self.end.y, self.end.z])

        return final_path

    # ── 2D overhead animation (during planning) ───────────────────────────────

    def draw_graph(self, rnd=None):
        plt.clf()

        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")

        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g", alpha=0.4)

        for ox, oy, radius, z_bottom, z_top in self.obstacle_list:
            circle = plt.Circle((ox, oy), radius, color="gray", alpha=0.5)
            plt.gca().add_patch(circle)
            plt.text(ox, oy, f"z:{z_bottom:.0f}-{z_top:.0f}",
                     fontsize=7, ha="center")

        plt.plot(self.start.x, self.start.y, "xr", markersize=10)
        plt.plot(self.end.x,   self.end.y,   "xb", markersize=10)

        plot_arrow(self.start.x, self.start.y, self.start.yaw)
        plot_arrow(self.end.x,   self.end.y,   self.end.yaw)

        plt.axis([-2, 22, -2, 22])
        plt.title("RRT* Dubins 3D  –  overhead view (planning)")
        plt.grid(True)
        plt.pause(0.01)

    # ── Full 3D result graph ──────────────────────────────────────────────────

    def draw_graph_3d(self, ax):
        # Tree branches
        for node in self.node_list:
            if node.parent:
                path_z = (node.path_z if node.path_z
                          else [node.z] * len(node.path_x))
                ax.plot(node.path_x, node.path_y, path_z,
                        "-g", alpha=0.25, linewidth=0.7)

        # Cylindrical buildings
        theta = np.linspace(0, 2 * np.pi, 40)
        for ox, oy, radius, z_bottom, z_top in self.obstacle_list:
            xc = ox + radius * np.cos(theta)
            yc = oy + radius * np.sin(theta)

            # Top and bottom rings
            ax.plot(xc, yc, [z_top]    * len(theta), color="gray", alpha=0.7)
            ax.plot(xc, yc, [z_bottom] * len(theta), color="gray", alpha=0.7)

            # Vertical edges (every 8th point to keep it clean)
            for k in range(0, len(theta), 8):
                ax.plot([xc[k], xc[k]], [yc[k], yc[k]],
                        [z_bottom, z_top], color="gray", alpha=0.4)

        # Start and goal markers
        ax.scatter(self.start.x, self.start.y, self.start.z,
                   c="red",  marker="*", s=200, zorder=5, label="Start")
        ax.scatter(self.end.x,   self.end.y,   self.end.z,
                   c="blue", marker="*", s=200, zorder=5, label="Goal")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z / Altitude (m)")
        ax.set_title("RRT* Dubins 3D  –  Fixed-Wing UAV")
        ax.grid(True)


# ============================================================================
# Obstacle generator
# ============================================================================

def create_random_obstacles_3d(n,
                                min_x, max_x, min_y, max_y,
                                min_radius, max_radius,
                                min_z_bottom, max_z_bottom,
                                min_height,   max_height,
                                avoid=None, avoid_dist=2.0,
                                min_separation=1.0):
    obstacles = []
    avoid     = avoid or []

    attempts = 0
    while len(obstacles) < n:
        attempts += 1
        if attempts > 10000:
            raise RuntimeError("Could not place all obstacles — "
                               "try relaxing constraints.")

        ox       = random.uniform(min_x, max_x)
        oy       = random.uniform(min_y, max_y)
        radius   = random.uniform(min_radius, max_radius)
        z_bottom = random.uniform(min_z_bottom, max_z_bottom)
        z_top    = z_bottom + random.uniform(min_height, max_height)

        valid = True
        for ax, ay in avoid:
            if math.hypot(ox - ax, oy - ay) <= avoid_dist + radius:
                valid = False
                break

        if valid:
            for ex, ey, er, _, _ in obstacles:
                if math.hypot(ox - ex, oy - ey) <= radius + er + min_separation:
                    valid = False
                    break

        if valid:
            obstacles.append((ox, oy, radius, z_bottom, z_top))

    return obstacles


# ============================================================================
# Path smoother  (X, Y, Z, yaw  –  polynomial fit)
# ============================================================================

def smooth_path_3d(path, degree=6, n_points=300):
    if path is None or len(path) < 3:
        return None, None, None, None

    xs = np.array([p[0] for p in path])
    ys = np.array([p[1] for p in path])
    zs = np.array([p[2] for p in path])

    # Estimate yaw from consecutive waypoints
    raw_yaw = [
        math.atan2(path[i + 1][1] - path[i][1],
                   path[i + 1][0] - path[i][0])
        for i in range(len(path) - 1)
    ]
    raw_yaw.append(raw_yaw[-1])   # repeat last
    yaws = np.unwrap(np.array(raw_yaw))

    t   = np.linspace(0.0, 1.0, len(xs))
    deg = min(degree, len(xs) - 1)
    ts  = np.linspace(0.0, 1.0, n_points)

    smooth_x   = np.polyval(np.polyfit(t, xs,   deg), ts)
    smooth_y   = np.polyval(np.polyfit(t, ys,   deg), ts)
    smooth_z   = np.polyval(np.polyfit(t, zs,   deg), ts)
    smooth_yaw = np.polyval(np.polyfit(t, yaws, deg), ts)
    smooth_yaw = (smooth_yaw + np.pi) % (2 * np.pi) - np.pi

    return smooth_x, smooth_y, smooth_z, smooth_yaw


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 55)
    print("  RRT* Dubins 3D  –  Fixed-Wing UAV Path Planner")
    print("=" * 55)

    # ── Start / goal:  [x, y, z, heading_rad, pitch_rad] ────────────────────
    start = [0.0,  0.0,  2.0,  np.deg2rad(0.0),  np.deg2rad(0.0)]
    goal  = [15.0, 15.0, 10.0,  np.deg2rad(0.0),  np.deg2rad(0.0)]

    # ── Buildings ────────────────────────────────────────────────────────────
    obstacle_list = create_random_obstacles_3d(
        n=4,
        min_x=1.5,    max_x=8.5,
        min_y=1.5,    max_y=8.5,
        min_radius=0.4, max_radius=1.2,
        min_z_bottom=0.0, max_z_bottom=0.5,
        min_height=4.0,   max_height=9.0,
        avoid=[(start[0], start[1]), (goal[0], goal[1])],
        avoid_dist=2.0,
        min_separation=1.0
    )

    print("\nBuildings (x, y, radius, z_bottom, z_top):")
    for b in obstacle_list:
        print(f"  {b}")

    # ── Planner ──────────────────────────────────────────────────────────────
    rrtstar = RRTStarDubins3D(
        start, goal,
        obstacle_list,
        rand_area=[0, 20],
        rand_z=[0.0, 12.0],
        turn_radius=4.5,
        pitch_lims=[np.deg2rad(-15.0), np.deg2rad(20.0)],
        max_iter=300,
        goal_sample_rate=10
    )

    print("\nPlanning...\n")
    t0   = time.perf_counter()
    path = rrtstar.planning(animation=False)
    elapsed = time.perf_counter() - t0

    print(f"\nPlanning time : {elapsed:.2f} s")
    print(f"Path found    : {path is not None}")

    if path is not None:
        # Calculate distances and time
        start_pos = [start[0], start[1], start[2]]
        goal_pos = [goal[0], goal[1], goal[2]]
        straight_distance = math.sqrt(sum((a - b)**2 for a, b in zip(start_pos, goal_pos)))
        
        path_length = 0.0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            path_length += math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))
        
        # Assume a speed for UAV, say 10 m/s
        speed = 10.0  # m/s
        traversal_time = path_length / speed
        
        print(f"Straight-line distance between start and goal: {straight_distance:.2f} m")
        print(f"Total path length: {path_length:.2f} m")
        print(f"Estimated traversal time at {speed} m/s: {traversal_time:.2f} s")
        
        # Create waypoints with time
        waypoints = [(0.0, start[0], start[1], start[2])]
        cumulative_time = 0.0
        for i in range(1, len(path)):
            p1 = path[i-1]
            p2 = path[i]
            dist = math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))
            dt = dist / speed
            cumulative_time += dt
            waypoints.append((cumulative_time, p2[0], p2[1], p2[2]))
        
        print("\nWaypoints (time, x, y, z):")
        for wp in waypoints:
            print(f"  {wp[0]:.2f} s, {wp[1]:.2f} m, {wp[2]:.2f} m, {wp[3]:.2f} m")
        
        raw_x = [p[0] for p in path]
        raw_y = [p[1] for p in path]
        raw_z = [p[2] for p in path]

        smooth_x, smooth_y, smooth_z, smooth_yaw = smooth_path_3d(path, degree=4)
        
        # Smoothed waypoints
        if smooth_x is not None:
            waypoints_smooth = [(0.0, smooth_x[0], smooth_y[0], smooth_z[0])]
            cumulative_time_smooth = 0.0
            for i in range(1, len(smooth_x)):
                dist = math.sqrt((smooth_x[i] - smooth_x[i-1])**2 + (smooth_y[i] - smooth_y[i-1])**2 + (smooth_z[i] - smooth_z[i-1])**2)
                dt = dist / speed
                cumulative_time_smooth += dt
                waypoints_smooth.append((cumulative_time_smooth, smooth_x[i], smooth_y[i], smooth_z[i]))
            
            print("\nSmoothed Waypoints (time, x, y, z) - first 10:")
            for wp in waypoints_smooth[:10]:
                print(f"  {wp[0]:.2f} s, {wp[1]:.2f} m, {wp[2]:.2f} m, {wp[3]:.2f} m")
            if len(waypoints_smooth) > 10:
                print("  ...")
            
            # Export to CSV
            with open('dubins_waypoints.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time', 'x', 'y', 'z'])
                for wp in waypoints:
                    writer.writerow(wp)
            
            with open('smoothed_waypoints.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time', 'x', 'y', 'z'])
                for wp in waypoints_smooth:
                    writer.writerow(wp)
            
            print("\nExported dubins_waypoints.csv and smoothed_waypoints.csv")

    if path is None:
        print("No path found — try increasing max_iter or relaxing constraints.")
        plt.show()
        return

    # ── Figure 1: overhead 2-D planning view ─────────────────────────────────
    plt.figure(figsize=(8, 7))
    rrtstar.draw_graph()
    plt.plot(raw_x, raw_y, "-r",  linewidth=2, label="RRT* path")
    if smooth_x is not None:
        plt.plot(smooth_x, smooth_y, "-b", linewidth=2, label="Smoothed path")
    plt.title("Overhead View")
    plt.legend()
    plt.grid(True)

    # ── Figure 2: full 3-D view ───────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")

    rrtstar.draw_graph_3d(ax)

    ax.plot(raw_x, raw_y, raw_z,
            "-r", linewidth=2, label="RRT* Dubins 3D path")
    if smooth_x is not None:
        ax.plot(smooth_x, smooth_y, smooth_z,
                "-b", linewidth=2.5, label="Smoothed path")

    ax.legend()
    plt.tight_layout()

    # ── Figure 3: altitude profile ────────────────────────────────────────────
    plt.figure(figsize=(9, 4))
    ts_raw = np.linspace(0, 1, len(raw_z))
    plt.plot(ts_raw, raw_z, "-r", label="Raw altitude")
    if smooth_z is not None:
        plt.plot(np.linspace(0, 1, len(smooth_z)), smooth_z,
                 "-b", linewidth=2, label="Smoothed altitude")
    plt.axhline(start[2], color="green",  linestyle="--",
                label=f"Start Z = {start[2]:.1f} m")
    plt.axhline(goal[2],  color="purple", linestyle="--",
                label=f"Goal  Z = {goal[2]:.1f} m")
    plt.xlabel("Normalised path parameter")
    plt.ylabel("Altitude (m)")
    plt.title("Altitude Profile")
    plt.legend()
    plt.grid(True)

    # ── Figure 4: yaw profile ─────────────────────────────────────────────────
    if smooth_yaw is not None:
        plt.figure(figsize=(9, 4))
        plt.plot(np.linspace(0, 1, len(smooth_yaw)),
                 np.rad2deg(smooth_yaw), "-b", linewidth=2)
        plt.xlabel("Normalised path parameter")
        plt.ylabel("Heading (degrees)")
        plt.title("Heading Profile")
        plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
