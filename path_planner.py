import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cyecca.dubins import derive_dubins
import casadi as ca

# Cache derive_dubins once
_dubins_plan, _dubins_eval = derive_dubins()


# ─────────────────────────────────────────────
# Wall definition
# ─────────────────────────────────────────────

class RectWall:
    def __init__(self, x_min, x_max, y_min, y_max, height=15.0):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.height = height

    def collides_point(self, x, y, z=None, margin=2.5):
        if z is not None and z > self.height:
            return False
        return (self.x_min - margin <= x <= self.x_max + margin and
                self.y_min - margin <= y <= self.y_max + margin)


# ─────────────────────────────────────────────
# Dubins helpers
# ─────────────────────────────────────────────

def dubins_collides(p0, psi0, p1, psi1, R, walls, n_samples=15, z0=None, z1=None, uav_radius=1.0):
    try:
        cost, type_, a1, d, a2, tp0, tp1, c0, c1 = _dubins_plan(
            ca.DM(p0), psi0, ca.DM(p1), psi1, R)
    except Exception:
        return True, None

    pts = []
    for i, s in enumerate(np.linspace(0, 1, n_samples)):
        x, y, _ = _dubins_eval(s, ca.DM(p0), psi0, a1, d, a2, tp0, tp1, c0, c1, R)
        x, y = float(x), float(y)
        z = None
        if z0 is not None and z1 is not None:
            z = z0 + (z1 - z0) * s
        for wall in walls:
            if wall.collides_point(x, y, z=z, margin=2.5 + uav_radius):
                return True, None
        pts.append((x, y))
    return False, pts


def sample_dubins_path(p0, psi0, p1, psi1, R, n_samples=40):
    cost, type_, a1, d, a2, tp0, tp1, c0, c1 = _dubins_plan(
        ca.DM(p0), psi0, ca.DM(p1), psi1, R)
    points = []
    for s in np.linspace(0, 1, n_samples):
        x, y, psi = _dubins_eval(s, ca.DM(p0), psi0, a1, d, a2, tp0, tp1, c0, c1, R)
        points.append((float(x), float(y), float(psi)))
    return points


# ─────────────────────────────────────────────
# Shortcut post-processing (disabled by default)
# ─────────────────────────────────────────────

def shortcut_path(path, walls, R):
    if len(path) <= 2:
        return path
    print("  Shortcutting path...")
    optimized = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            x0, y0, psi0 = optimized[-1]
            x1, y1, psi1 = path[j]
            blocked, _ = dubins_collides([x0, y0], psi0, [x1, y1], psi1, R, walls, n_samples=20)
            if not blocked:
                break
            j -= 1
        optimized.append(path[j])
        i = j
    print(f"  Shortcut: {len(path)} -> {len(optimized)} waypoints")
    return optimized


# ─────────────────────────────────────────────
# A* with Dubins edges + turn penalty
# ─────────────────────────────────────────────

def astar_dubins(start, goal, walls, R,
                 grid_res=3.0, heading_bins=8,
                 x_bounds=(0, 50), y_bounds=(0, 50),
                 w_turn=2.0, uav_radius=1.0):
    headings = np.linspace(0, 2 * np.pi, heading_bins, endpoint=False)

    def quantize(x, y, psi):
        xi   = round((x - x_bounds[0]) / grid_res)
        yi   = round((y - y_bounds[0]) / grid_res)
        psii = int(round(psi % (2*np.pi) / (2*np.pi) * heading_bins)) % heading_bins
        return (xi, yi, psii)

    def to_cont(xi, yi, psii):
        return (xi * grid_res + x_bounds[0],
                yi * grid_res + y_bounds[0],
                headings[psii])

    def heuristic(node):
        x0, y0, _ = to_cont(*node)
        return np.hypot(gx - x0, gy - y0)

    start_node = quantize(*start)
    goal_node  = quantize(*goal)
    gx, gy, _  = to_cont(*goal_node)

    edge_cache = {}
    open_heap  = [(heuristic(start_node), 0.0, start_node, None)]
    came_from  = {}
    g_score    = {start_node: 0.0}
    visited    = set()
    expanded   = 0
    best_path  = None
    best_cost  = float("inf")

    while open_heap:
        f, g, current, parent = heapq.heappop(open_heap)

        if current in visited:
            continue
        visited.add(current)
        came_from[current] = parent
        expanded += 1

        cx, cy, cpsi = to_cont(*current)

        if expanded % 50 == 0:
            print(f"  A*: {expanded} nodes expanded, queue: {len(open_heap)}")

        if np.hypot(cx - gx, cy - gy) < grid_res * 1.5:
            candidate = []
            node = current
            while node is not None:
                candidate.append(node)
                node = came_from[node]
            candidate.reverse()
            candidate_cost = g
            if best_path is None or candidate_cost < best_cost:
                best_path = candidate
                best_cost = candidate_cost
                print(f"  A*: better path found at {expanded} nodes, cost={best_cost:.1f}")
            if open_heap and open_heap[0][0] >= best_cost:
                print(f"  A*: optimal path confirmed after {expanded} nodes")
                return best_path, to_cont, edge_cache, headings, visited

        for next_psii, next_psi in enumerate(headings):
            x1 = cx + grid_res * np.cos(next_psi)
            y1 = cy + grid_res * np.sin(next_psi)

            if not (x_bounds[0] <= x1 <= x_bounds[1] and
                    y_bounds[0] <= y1 <= y_bounds[1]):
                continue

            neighbor = quantize(x1, y1, next_psi)
            edge_key = (current, neighbor)

            if edge_key not in edge_cache:
                blocked, pts = dubins_collides(
                    [cx, cy], cpsi, [x1, y1], next_psi, R, walls, uav_radius=uav_radius)
                edge_cache[edge_key] = None if blocked else pts

            if edge_cache[edge_key] is None:
                continue

            heading_change = abs(next_psi - cpsi)
            heading_change = min(heading_change, 2*np.pi - heading_change)
            turn_penalty = w_turn * heading_change

            new_g = g + grid_res + turn_penalty
            if new_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = new_g
                heapq.heappush(open_heap,
                    (new_g + heuristic(neighbor), new_g, neighbor, current))

    if best_path is not None:
        print(f"  A*: returning best path found, cost={best_cost:.1f}")
        return best_path, to_cont, edge_cache, headings, visited
    print("A* failed to find a path!")
    return None, None, None, None, None


# ─────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────

def plan_fixed_wing(start, goal, walls, R=4.0,
                    grid_res=4.0, heading_bins=8,
                    x_bounds=(0, 50), y_bounds=(0, 50),
                    w_turn=2.0, uav_radius=1.0):
    """
    start: (x, y, z, heading_degrees)
    goal:  (x, y, z, heading_degrees)
    heading: 0=east, 90=north, 180=west, 270=south
    uav_radius: physical size of UAV for collision margin (meters)
    """
    start_z = start[2]
    goal_z  = goal[2]
    start = (start[0], start[1], np.deg2rad(start[3]))
    # goal heading ignored — UAV arrives from whatever direction is most natural
    goal  = (goal[0], goal[1], 0.0)

    print("Running A* with Dubins edges...")
    nx = int((x_bounds[1]-x_bounds[0])/grid_res)
    ny = int((y_bounds[1]-y_bounds[0])/grid_res)
    print(f"  grid_res={grid_res}, heading_bins={heading_bins}, max nodes: {nx*ny*heading_bins}")

    bm = 1.0  # boundary margin for search bounds (independent of uav_radius)
    x_bounds_inner = (x_bounds[0] + bm, x_bounds[1] - bm)
    y_bounds_inner = (y_bounds[0] + bm, y_bounds[1] - bm)
    # uav_radius is used only in collision checking, not for bounding the search

    path_nodes, to_cont, edge_cache, headings, visited = astar_dubins(
        start, goal, walls, R,
        grid_res=grid_res, heading_bins=heading_bins,
        x_bounds=x_bounds_inner, y_bounds=y_bounds_inner,
        w_turn=w_turn, uav_radius=uav_radius)

    if path_nodes is None:
        raise RuntimeError("No path found!")

    path = [to_cont(*node) for node in path_nodes]
    # compute natural arrival heading from last node direction toward goal
    last_x, last_y, last_psi = path[-1]
    dx = goal[0] - last_x
    dy = goal[1] - last_y
    natural_heading = np.arctan2(dy, dx)
    path.append((goal[0], goal[1], natural_heading))

    # path = shortcut_path(path, walls, R)  # disabled: can miss narrow walls

    print(f"Final path: {len(path)} waypoints")

    xs   = [p[0] for p in path]
    ys   = [p[1] for p in path]
    psis = [p[2] for p in path]
    zs   = [start_z + (goal_z - start_z) * i / (len(path) - 1) for i in range(len(path))]
    speed = 3.0

    return {
        'x1':  xs,
        'y1':  ys,
        'z1':  zs,
        'v_x': [speed * np.cos(psi) for psi in psis],
        'v_y': [speed * np.sin(psi) for psi in psis],
        'v_z': [0.0] * len(path),
        'ax':  [0.0] * len(path),
        'ay':  [0.0] * len(path),
        'az':  [0.0] * len(path),
        'path': path,
        'R': R,
        'visited': [(to_cont(*n)[0], to_cont(*n)[1]) for n in visited]
    }


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────


def fix_wall_clips(result, walls, uav_radius=1.0, margin=2.5):
    """
    Check if waypoints are too close to walls and insert repulsion waypoints.
    Pushes any waypoint that is inside the margin zone away from the wall.
    """
    total_margin = margin + uav_radius
    xs = result['x1']
    ys = result['y1']
    zs = result['z1']
    path = result['path']

    new_xs, new_ys, new_zs, new_path = [], [], [], []

    for i in range(len(xs)):
        x, y, z = xs[i], ys[i], zs[i]
        psi = path[i][2]
        pushed = False

        for w in walls:
            # check if this waypoint is inside the margin zone
            if w.collides_point(x, y, margin=total_margin):
                # push away from wall center
                wx_center = (w.x_min + w.x_max) / 2
                wy_center = (w.y_min + w.y_max) / 2
                dx = x - wx_center
                dy = y - wy_center
                dist = np.hypot(dx, dy)
                if dist > 0:
                    # push to just outside the margin
                    scale = (total_margin + 0.5) / dist
                    x = wx_center + dx * scale
                    y = wy_center + dy * scale
                pushed = True

        new_xs.append(x)
        new_ys.append(y)
        new_zs.append(z)
        new_path.append((x, y, psi))

    result['x1']   = new_xs
    result['y1']   = new_ys
    result['z1']   = new_zs
    result['path'] = new_path
    return result

def plot_plan(path, walls, start, goal, x_bounds, y_bounds, R=4.0, visited=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    for w in walls:
        rect = patches.Rectangle(
            (w.x_min, w.y_min), w.x_max - w.x_min, w.y_max - w.y_min,
            linewidth=1, edgecolor='black', facecolor='gray', alpha=0.5)
        ax.add_patch(rect)

    if visited is not None:
        vxs = [v[0] for v in visited]
        vys = [v[1] for v in visited]
        ax.plot(vxs, vys, 'b.', markersize=4, alpha=0.4, label='Searched nodes')

    for i in range(len(path) - 1):
        x0, y0, psi0 = path[i]
        x1, y1, psi1 = path[i+1]
        try:
            pts = sample_dubins_path([x0, y0], psi0, [x1, y1], psi1, R, n_samples=40)
            dxs = [p[0] for p in pts]
            dys = [p[1] for p in pts]
            ax.plot(dxs, dys, 'r-', linewidth=2,
                    label='Dubins edge' if i == 0 else '')
        except Exception as e:
            print(f"  Edge {i}: FAILED: {e}")
            ax.plot([x0, x1], [y0, y1], 'r--', linewidth=1.5,
                    label='fallback' if i == 0 else '')

    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, 'bo', markersize=6, label='A* nodes')

    for x, y, psi in path:
        ax.annotate('', xy=(x + 1.8*np.cos(psi), y + 1.8*np.sin(psi)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    ax.plot(start[0], start[1], 'go', markersize=12, label='Start')
    ax.plot(goal[0],  goal[1],  'rs', markersize=12, label='Goal')
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_title('A* with Dubins Edges - Fixed Wing UAV')
    plt.tight_layout()
    plt.show()


def plot_trajectory_2d(path_leg, walls, start, goal):
    """2D x-y trajectory plot with walls, start, and goal."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for w in walls:
        rect = patches.Rectangle(
            (w.x_min, w.y_min), w.x_max - w.x_min, w.y_max - w.y_min,
            linewidth=1, edgecolor='black', facecolor='gray', alpha=0.5)
        ax.add_patch(rect)
    ax.plot(path_leg['x'], path_leg['y'], 'b-', linewidth=1.5, label='Trajectory')
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0],  goal[1],  'rs', markersize=10, label='Goal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_title('2D Trajectory with Walls')
    plt.tight_layout()
    plt.show()


def plot_3d_with_walls(path_leg, walls):
    """3D trajectory plot with walls."""
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.colors as colors

    traj_x = path_leg['x']
    traj_y = path_leg['y']
    traj_z = path_leg['z']
    t      = path_leg['t_x']

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    for w in walls:
        h = w.height
        verts = [
            [(w.x_min,w.y_min,0),(w.x_max,w.y_min,0),(w.x_max,w.y_max,0),(w.x_min,w.y_max,0)],
            [(w.x_min,w.y_min,h),(w.x_max,w.y_min,h),(w.x_max,w.y_max,h),(w.x_min,w.y_max,h)],
            [(w.x_min,w.y_min,0),(w.x_max,w.y_min,0),(w.x_max,w.y_min,h),(w.x_min,w.y_min,h)],
            [(w.x_min,w.y_max,0),(w.x_max,w.y_max,0),(w.x_max,w.y_max,h),(w.x_min,w.y_max,h)],
            [(w.x_min,w.y_min,0),(w.x_min,w.y_max,0),(w.x_min,w.y_max,h),(w.x_min,w.y_min,h)],
            [(w.x_max,w.y_min,0),(w.x_max,w.y_max,0),(w.x_max,w.y_max,h),(w.x_max,w.y_min,h)],
        ]
        poly = Poly3DCollection(verts, alpha=0.3, facecolor='gray', edgecolor='black', linewidth=0.5)
        ax.add_collection3d(poly)

    normalize = colors.Normalize(vmin=min(t), vmax=max(t))
    scatter = ax.scatter(traj_x, traj_y, traj_z, c=t, s=8, cmap='viridis', norm=normalize)
    fig.colorbar(scatter, ax=ax, label='Time')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Trajectory with Walls')
    plt.tight_layout()
    plt.show()



def plot_position(path_leg):
    """Combined x, y, z vs time on one graph."""
    t = path_leg['t_x']
    plt.figure(figsize=(10, 6))
    plt.plot(t, path_leg['x'], label='x')
    plt.plot(t, path_leg['y'], label='y')
    plt.plot(t, path_leg['z'], label='z')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Position vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_acceleration(path_leg):
    """Calculates acceleration analytically from polynomial coefficients."""
    T_legs = path_leg['T_legs']
    # You'll need to pass cx_f, cy_f, cz_f into this function or store them in path_leg
    # Assuming they are stored or accessible:
    cx, cy, cz = path_leg['coeffs_x'], path_leg['coeffs_y'], path_leg['coeffs_z']
    
    all_t = []
    all_ax, all_ay, all_az = [], [], []
    elapsed = 0
    
    for i, T in enumerate(T_legs):
        beta = np.linspace(0, 1, 100)
        c = cx[i*6 : (i+1)*6]
        # Second derivative of: c0*b^5 + c1*b^4 + c2*b^3 + c3*b^2 + c4*b + c5
        # is: (20*c0*b^3 + 12*c1*b^2 + 6*c2*b + 2*c3) / T^2
        ax = (20*c[0]*beta**3 + 12*c[1]*beta**2 + 6*c[2]*beta + 2*c[3]) / (T**2)
        
        c = cy[i*6 : (i+1)*6]
        ay = (20*c[0]*beta**3 + 12*c[1]*beta**2 + 6*c[2]*beta + 2*c[3]) / (T**2)
        
        c = cz[i*6 : (i+1)*6]
        az = (20*c[0]*beta**3 + 12*c[1]*beta**2 + 6*c[2]*beta + 2*c[3]) / (T**2)
        
        all_t.append(beta * T + elapsed)
        all_ax.append(ax); all_ay.append(ay); all_az.append(az)
        elapsed += T

    plt.figure(figsize=(10, 5))
    plt.plot(np.concatenate(all_t), np.concatenate(all_ax), label='a_x (Analytical)')
    plt.plot(np.concatenate(all_t), np.concatenate(all_ay), label='a_y (Analytical)')
    plt.grid(True)
    plt.legend()
    plt.title("Smooth Analytical Acceleration")
    plt.show()

def plot_velocities(path_leg, v_min=None, v_max=None):
    t = np.array(path_leg['t_x'])
    x = np.array(path_leg['x'])
    y = np.array(path_leg['y'])
    z = np.array(path_leg['z'])

    dt = np.diff(t)
    valid = dt > 1e-5 
    
    vx = np.diff(x)[valid] / dt[valid]
    vy = np.diff(y)[valid] / dt[valid]
    vz = np.diff(z)[valid] / dt[valid]
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    t_mid = ((t[:-1] + t[1:]) / 2)[valid]

    plt.figure(figsize=(10, 6))
    plt.plot(t_mid, v_mag, 'k-', linewidth=2, label='Actual Airspeed')
    
    if v_min is not None:
        plt.axhline(y=v_min, color='r', linestyle='--', alpha=0.7, label=f'Target Min Speed ({v_min} m/s)')
    if v_max is not None:
        plt.axhline(y=v_max, color='b', linestyle='--', alpha=0.7, label=f'Target Max Speed ({v_max} m/s)')

    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Velocity Profile with Continuous Constraints')
    plt.legend()
    plt.grid(True)
    plt.show()