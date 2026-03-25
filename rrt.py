import math
import random
import numpy as np


class Node:
    """RRT node representing 3D position and tree link."""

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None


def distance(n1: Node, n2: Node) -> float:
    dx = n1.x - n2.x
    dy = n1.y - n2.y
    dz = n1.z - n2.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def steer(from_node: Node, to_node: Node, step_size: float) -> Node:
    dx = to_node.x - from_node.x
    dy = to_node.y - from_node.y
    dz = to_node.z - from_node.z
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    if dist < 1e-8:
        return Node(from_node.x, from_node.y, from_node.z)

    if dist <= step_size:
        return Node(to_node.x, to_node.y, to_node.z)

    ratio = step_size / dist
    new_x = from_node.x + dx * ratio
    new_y = from_node.y + dy * ratio
    new_z = from_node.z + dz * ratio
    return Node(new_x, new_y, new_z)


def is_point_in_sphere(point: Node, center: tuple, radius: float) -> bool:
    dx = point.x - center[0]
    dy = point.y - center[1]
    dz = point.z - center[2]
    return dx * dx + dy * dy + dz * dz <= radius**2


def line_sphere_collision(p1: Node, p2: Node, center: tuple, radius: float) -> bool:
    # segment p1-p2 intersects sphere at center/radius
    x1, y1, z1 = p1.x, p1.y, p1.z
    x2, y2, z2 = p2.x, p2.y, p2.z
    cx, cy, cz = center

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    if abs(dx) < 1e-9 and abs(dy) < 1e-9 and abs(dz) < 1e-9:
        return is_point_in_sphere(p1, center, radius)

    t = ((cx - x1) * dx + (cy - y1) * dy + (cz - z1) * dz) / (dx * dx + dy * dy + dz * dz)
    t = max(0.0, min(1.0, t))

    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    proj_z = z1 + t * dz
    ddx = proj_x - cx
    ddy = proj_y - cy
    ddz = proj_z - cz
    return ddx * ddx + ddy * ddy + ddz * ddz <= radius**2


def is_collision(p1: Node, p2: Node, obstacles: list) -> bool:
    for (ox, oy, oz, r) in obstacles:
        if line_sphere_collision(p1, p2, (ox, oy, oz), r):
            return True
    return False


def get_random_node(rand_area: tuple, goal_node: Node, goal_sample_rate: float) -> Node:
    if random.random() * 100.0 <= goal_sample_rate:
        return Node(goal_node.x, goal_node.y, goal_node.z)

    x = random.uniform(rand_area[0], rand_area[1])
    y = random.uniform(rand_area[2], rand_area[3])
    z = random.uniform(rand_area[4], rand_area[5])
    return Node(x, y, z)


def get_nearest_node_index(node_list: list, random_node: Node) -> int:
    distances = [
        (node.x - random_node.x) ** 2 +
        (node.y - random_node.y) ** 2 +
        (node.z - random_node.z) ** 2
        for node in node_list
    ]
    return distances.index(min(distances))


def generate_final_course(last_node: Node) -> list:
    path = []
    node = last_node
    while node is not None:
        path.append((node.x, node.y, node.z))
        node = node.parent
    return path[::-1]


def rrt_planner(start: tuple,
                goal: tuple,
                obstacles: list,
                rand_area: tuple,
                max_iter: int = 500,
                step_size: float = 1.0,
                goal_sample_rate: float = 5.0,
                goal_tolerance: float = 1.0,
                ) -> list:
    """Perform basic 3D RRT and return path as list of (x,y,z) points or [] if failed."""

    start_node = Node(start[0], start[1], start[2])
    goal_node = Node(goal[0], goal[1], goal[2])
    node_list = [start_node]

    if is_point_in_sphere(start_node, (goal_node.x, goal_node.y, goal_node.z), 0.0):
        return [(start_node.x, start_node.y, start_node.z),
                (goal_node.x, goal_node.y, goal_node.z)]

    for _ in range(max_iter):
        rand_node = get_random_node(rand_area, goal_node, goal_sample_rate)
        nearest_index = get_nearest_node_index(node_list, rand_node)
        nearest_node = node_list[nearest_index]

        new_node = steer(nearest_node, rand_node, step_size)

        if is_collision(nearest_node, new_node, obstacles):
            continue

        new_node.parent = nearest_node
        node_list.append(new_node)

        if distance(new_node, goal_node) <= goal_tolerance:
            if not is_collision(new_node, goal_node, obstacles):
                goal_node.parent = new_node
                return generate_final_course(goal_node)

    return []


def plot_rrt_3d(path: list, obstacles: list, start: tuple, goal: tuple):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from matplotlib.patches import Circle
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as e:
        raise ImportError("matplotlib is required for 3D visualization") from e

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot path
    if path:
        xs, ys, zs = zip(*path)
        ax.plot(xs, ys, zs, '-o', color='blue', label='RRT path', linewidth=2, markersize=3)

    # Plot start/goal
    ax.scatter([start[0]], [start[1]], [start[2]], color='green', s=80, label='start')
    ax.scatter([goal[0]], [goal[1]], [goal[2]], color='red', s=80, label='goal')

    # Plot obstacles as translucent spheres
    for (ox, oy, oz, r) in obstacles:
        # sphere param
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 12)
        x = ox + r * np.outer(np.cos(u), np.sin(v))
        y = oy + r * np.outer(np.sin(u), np.sin(v))
        z = oz + r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color='gray', alpha=0.35, rstride=1, cstride=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D RRT Planning')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example use and smoke test for 3D RRT
    start = (0.0, 0.0, 0.0)
    goal = (15.0, 15.0, 6.0)
    obstacles = [
        (5.0, 5.0, 2.0, 2.5),
        (10.0, 10.0, 4.0, 2.5),
        (8.0, 5.0, 3.5, 1.5),
    ]
    rand_area = (-2.0, 20.0, -2.0, 20.0, -2.0, 10.0)

    path = rrt_planner(start, goal, obstacles, rand_area,
                       max_iter=5000,
                       step_size=1.0,
                       goal_sample_rate=10.0,
                       goal_tolerance=1.2)

    if path:
        print("3D path found with {} points".format(len(path)))
        for p in path:
            print(f"{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}")
    else:
        print("No 3D path found")

    plot_rrt_3d(path, obstacles, start, goal)
