import copy
import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import pathlib

# Import from parent directory (workspace root)
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from . import dubins_path_planner
from rrt_star import RRTStar
from .utils.plot import plot_arrow

show_animation = True


class RRTStarDubins(RRTStar):

    class Node(RRTStar.Node):

        def __init__(self, x, y, yaw):
            super().__init__(x, y)
            self.yaw = yaw
            self.path_yaw = []

    def __init__(self, start, goal, obstacle_list, rand_area,
                 goal_sample_rate=10,
                 max_iter=300,
                 connect_circle_dist=50.0,
                 robot_radius=0.0):

        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])

        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]

        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list

        self.connect_circle_dist = connect_circle_dist
        self.robot_radius = robot_radius

        self.curvature = 1.0
        self.goal_yaw_th = np.deg2rad(1.0)
        self.goal_xy_th = 0.5

        self.cone_angle = np.pi

    def planning(self, animation=True, search_until_max_iter=True):

        self.node_list = [self.start]

        for i in range(self.max_iter):

            print("Iter:", i, ", number of nodes:", len(self.node_list))
            sys.stdout.flush()

            rnd = self.get_random_node()

            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)

            new_node = self.steer(self.node_list[nearest_ind], rnd)

            if self.check_collision(new_node, self.obstacle_list, self.robot_radius):

                near_indexes = self.find_near_nodes(new_node)

                new_node = self.choose_parent(new_node, near_indexes)

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

    def steer(self, from_node, to_node):

        px, py, pyaw, mode, course_lengths = \
            dubins_path_planner.plan_dubins_path(
                from_node.x, from_node.y, from_node.yaw,
                to_node.x, to_node.y, to_node.yaw,
                self.curvature)

        if len(px) <= 1:
            return None

        new_node = copy.deepcopy(from_node)

        new_node.x = px[-1]
        new_node.y = py[-1]
        new_node.yaw = pyaw[-1]

        new_node.path_x = px
        new_node.path_y = py
        new_node.path_yaw = pyaw

        new_node.cost += sum([abs(c) for c in course_lengths])

        new_node.parent = from_node

        return new_node

    def calc_new_cost(self, from_node, to_node):

        _, _, _, _, course_lengths = dubins_path_planner.plan_dubins_path(
            from_node.x, from_node.y, from_node.yaw,
            to_node.x, to_node.y, to_node.yaw,
            self.curvature)

        cost = sum([abs(c) for c in course_lengths])

        return from_node.cost + cost

    def get_random_node(self):

        if random.randint(0, 100) > self.goal_sample_rate:

            d = random.uniform(0.0, self.max_rand - self.min_rand)
            th = self.start.yaw + random.uniform(-self.cone_angle, self.cone_angle)

            ox = self.start.x + d * math.cos(th)
            oy = self.start.y + d * math.sin(th)

            ox = min(max(self.min_rand, ox), self.max_rand)
            oy = min(max(self.min_rand, oy), self.max_rand)

            rnd = self.Node(ox, oy, random.uniform(-math.pi, math.pi))

        else:
            rnd = self.Node(self.end.x, self.end.y, self.end.yaw)

        return rnd

    def search_best_goal_node(self):

        goal_indexes = []

        for i, node in enumerate(self.node_list):

            if self.calc_dist_to_goal(node.x, node.y) <= self.goal_xy_th:
                goal_indexes.append(i)

        final_goal_indexes = []

        for i in goal_indexes:

            if abs(self.node_list[i].yaw - self.end.yaw) <= self.goal_yaw_th:
                final_goal_indexes.append(i)

        if not final_goal_indexes:
            return None

        best_index = min(final_goal_indexes, key=lambda i: self.node_list[i].cost)

        return best_index

    def generate_final_course(self, goal_index):

        node = self.node_list[goal_index]

        chain = [node]

        while node.parent:
            node = node.parent
            chain.append(node)

        chain.reverse()

        final_path = [[self.start.x, self.start.y]]

        for node in chain[1:]:

            for x, y in zip(node.path_x, node.path_y):

                if [x, y] != final_path[-1]:
                    final_path.append([x, y])

        if final_path[-1] != [self.end.x, self.end.y]:
            final_path.append([self.end.x, self.end.y])

        return final_path

    def draw_graph(self, rnd=None):

        plt.clf()

        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")

        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for ox, oy, size in self.obstacle_list:
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")

        plot_arrow(self.start.x, self.start.y, self.start.yaw)
        plot_arrow(self.end.x, self.end.y, self.end.yaw)

        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)


def create_random_obstacles(n, min_x, max_x, min_y, max_y,
                            min_size, max_size,
                            avoid=None,
                            avoid_dist=1.5,
                            min_separation=0.5):

    obstacles = []
    avoid = avoid or []

    while len(obstacles) < n:

        ox = random.uniform(min_x, max_x)
        oy = random.uniform(min_y, max_y)

        size = random.uniform(min_size, max_size)

        valid = True

        for ax, ay in avoid:
            if math.hypot(ox - ax, oy - ay) <= avoid_dist + size:
                valid = False

        for ex, ey, esize in obstacles:
            if math.hypot(ox - ex, oy - ey) <= size + esize + min_separation:
                valid = False

        if valid:
            obstacles.append((ox, oy, size))

    return obstacles


def smooth_path_2d(path, degree=5, n_points=300):

    if path is None or len(path) < 3:
        return None, None

    xs = np.array([p[0] for p in path])
    ys = np.array([p[1] for p in path])

    t = np.linspace(0.0, 1.0, len(xs))

    deg = min(degree, len(xs) - 1)

    px = np.polyfit(t, xs, deg)
    py = np.polyfit(t, ys, deg)

    ts = np.linspace(0.0, 1.0, n_points)

    smooth_x = np.polyval(px, ts)
    smooth_y = np.polyval(py, ts)

    return smooth_x, smooth_y


def main():

    print("Start RRT* Dubins planning")

    start = [0.0, 0.0, np.deg2rad(0.0)]
    goal = [10.0, 10.0, np.deg2rad(0.0)]

    obstacleList = create_random_obstacles(
        n=3,
        min_x=1,
        max_x=12,
        min_y=1,
        max_y=10,
        min_size=2,
        max_size=3.2,
        avoid=[(start[0], start[1]), (goal[0], goal[1])],
        avoid_dist=2,
        min_separation=1
    )

    print("Obstacles:", obstacleList)

    rrtstar_dubins = RRTStarDubins(
        start,
        goal,
        obstacleList,
        rand_area=[0, 15]
    )

    start_time = time.perf_counter()

    path = rrtstar_dubins.planning(animation=show_animation)

    planning_time = time.perf_counter() - start_time

    print("Planning complete:", path is not None)
    print("Planning time:", planning_time)

    if path:

        raw_x = [p[0] for p in path]
        raw_y = [p[1] for p in path]

        smooth_x, smooth_y = smooth_path_2d(path, degree=6)

        # Figure 1
        plt.figure(figsize=(8,6))

        rrtstar_dubins.draw_graph()

        plt.plot(raw_x, raw_y, "-r", label="RRT* Dubins path")

        if smooth_x is not None:
            plt.plot(smooth_x, smooth_y, "-b", linewidth=2,
                     label="Polynomial smoothed path")

        plt.title("RRT* Dubins Path with Smoothing")
        plt.legend()
        plt.grid(True)

        # Figure 2
        plt.figure(figsize=(8,6))

        plt.plot(raw_x, raw_y, "-r", label="Original")

        if smooth_x is not None:
            plt.plot(smooth_x, smooth_y, "-b", linewidth=2, label="Smoothed")

        plt.title("Original vs Smoothed Path")
        plt.legend()
        plt.grid(True)

        plt.show()


if __name__ == '__main__':
    main()