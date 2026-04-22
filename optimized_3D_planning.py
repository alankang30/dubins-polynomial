import sympy
import numpy as np
import scipy.optimize
import scipy.linalg


# ─────────────────────────────────────────────
# Cost matrix Q — minimizes jerk (3rd derivative)
# ─────────────────────────────────────────────

def find_Q(deriv, poly_deg, n_legs):
    k, l, m, n, n_c, n_l = sympy.symbols('k, l, m, n, n_c, n_l', integer=True)
    beta = sympy.symbols('beta')
    c = sympy.MatrixSymbol('c', n_c, 1)
    T = sympy.symbols('T')
    P = sympy.summation(c[k,0]*sympy.factorial(k)/sympy.factorial(k-m)*beta**(k-m)/T**m, (k,m,n))
    P = P.subs({m: deriv, n: poly_deg}).doit()
    J = sympy.integrate(P**2, (beta, 0, 1)).doit()
    p = sympy.Matrix([c[i,0] for i in range(poly_deg+1)])
    Q = sympy.Matrix([J]).jacobian(p).jacobian(p)/2
    assert (p.T@Q@p)[0,0].expand() == J
    Ti = sympy.MatrixSymbol('T', n_l, 1)
    return sympy.diag(*[Q.subs(T, Ti[i]) for i in range(n_legs)])


def find_A(deriv, poly_deg, beta, n_legs, leg, value):
    k, m, n, n_c, n_l = sympy.symbols('k, m, n, n_c, n_l', integer=True)
    c = sympy.MatrixSymbol('c', n_c, n_l)
    T = sympy.MatrixSymbol('T', n_l, 1)
    p = sympy.Matrix([c[i,l] for l in range(n_legs) for i in range(poly_deg+1)])
    P = sympy.summation(c[k,leg]*sympy.factorial(k)/sympy.factorial(k-m)*beta**(k-m)/T[leg]**m, (k,m,n))
    P = P.subs({m: deriv, n: poly_deg}).doit()
    return sympy.Matrix([P]).jacobian(p), sympy.Matrix([value])


def compute_trajectory(p, T_opt, n_pts=200):
    p = np.asarray(p, dtype=float).flatten()
    T_opt = np.asarray(T_opt, dtype=float).flatten()
    S = np.hstack([0, np.cumsum(T_opt)])
    t_all, x_all = [], []
    n_coeffs = 6
    for i in range(len(T_opt)):
        beta = np.linspace(0, 1, n_pts, dtype=float)
        ti   = T_opt[i]*beta + S[i]
        coeff = np.asarray(np.flip(p[i*n_coeffs:(i+1)*n_coeffs]), dtype=float)
        xi   = np.polyval(coeff, beta)
        t_all.append(ti); x_all.append(xi)
    return {'t': np.hstack(t_all), 'x': np.hstack(x_all)}


def find_cost_function(poly_deg=5, min_deriv=3, n_legs=2):
    """
    Builds the closed-form cost function using the MIT polynomial planning approach.
    All boundary conditions fixed: position, velocity, acceleration at each waypoint.
    Returns callable cost and coefficient functions.
    """
    n_coeffs = poly_deg + 1
    rowsf = list(range(n_coeffs * n_legs))  # all rows fixed

    A_rows_x, b_rows_x = [], []
    A_rows_y, b_rows_y = [], []
    A_rows_z, b_rows_z = [], []

    Q = find_Q(deriv=min_deriv, poly_deg=poly_deg, n_legs=n_legs)

    n_l, n_d = sympy.symbols('n_l, n_d', integer=True)
    x = sympy.MatrixSymbol('x', n_d, n_l)
    y = sympy.MatrixSymbol('y', n_d, n_l)
    z = sympy.MatrixSymbol('z', n_d, n_l)
    T = sympy.MatrixSymbol('T', n_l, 1)

    for i in range(n_legs):
        for m in range(3):  # pos, vel, acc
            for dim, rows, brows in [('x', A_rows_x, b_rows_x),
                                      ('y', A_rows_y, b_rows_y),
                                      ('z', A_rows_z, b_rows_z)]:
                sym = x if dim=='x' else (y if dim=='y' else z)
                Ar, br = find_A(deriv=m, poly_deg=poly_deg, beta=0,
                                n_legs=n_legs, leg=i, value=sym[m, i])
                rows.append(Ar); brows.append(br)
                Ar, br = find_A(deriv=m, poly_deg=poly_deg, beta=1,
                                n_legs=n_legs, leg=i, value=sym[m, i+1])
                rows.append(Ar); brows.append(br)

    def build(A_rows, b_rows):
        A = sympy.Matrix.vstack(*A_rows)
        b = sympy.Matrix.vstack(*b_rows)
        if A.shape[0] != A.shape[1]:
            raise ValueError(f'A not square: {A.shape}')
        I  = sympy.Matrix.eye(A.shape[0])
        rowsp = []  # no free rows — all fixed
        C  = sympy.Matrix.vstack(*[I[i,:] for i in rowsf + rowsp])
        Ai = A.inv()
        R  = (C @ Ai.T @ Q @ Ai @ C.T); R.simplify()
        n_f = len(rowsf)
        df  = (C @ b)[:n_f, 0]
        p_c = Ai @ df
        return p_c, df

    p_x, _ = build(A_rows_x, b_rows_x)
    p_y, _ = build(A_rows_y, b_rows_y)
    p_z, _ = build(A_rows_z, b_rows_z)

    Ti = sympy.symbols('T_0:{:d}'.format(n_legs))
    k  = sympy.symbols('k')

    for sym_list in [p_x, p_y, p_z]:
        sym_list = sym_list.subs(T, sympy.Matrix(Ti))

    p_x = p_x.subs(T, sympy.Matrix(Ti))
    p_y = p_y.subs(T, sympy.Matrix(Ti))
    p_z = p_z.subs(T, sympy.Matrix(Ti))
    Q2  = Q.subs(T, sympy.Matrix(Ti))

    J = ((p_x.T@Q2@p_x)[0,0]).simplify() + \
        ((p_y.T@Q2@p_y)[0,0]).simplify() + \
        ((p_z.T@Q2@p_z)[0,0]).simplify() + k*sum(Ti)

    return {
        'f_J':   sympy.lambdify([Ti, x, y, z, k], J),
        'f_p_x': sympy.lambdify([Ti, x, k], list(p_x)),
        'f_p_y': sympy.lambdify([Ti, y, k], list(p_y)),
        'f_p_z': sympy.lambdify([Ti, z, k], list(p_z)),
    }


# ─────────────────────────────────────────────
# Main trajectory solver
# ─────────────────────────────────────────────

def run_traj(x1, y1, z1, headings, k_time=15.0, v_min=1.0, v_max=15.0):
    """
    Minimum-jerk polynomial trajectory.

    @param x1, y1, z1  : waypoint positions (lists)
    @param headings     : heading at each waypoint (radians)
    @param k_time       : time penalty weight — lower = smoother/slower
    @param v_min, v_max : desired speed range — used to set waypoint velocities
    @return             : dict with x, y, z, t_x arrays
    """
    n_waypoints = len(x1)
    n_legs      = n_waypoints - 1

    x1       = np.array(x1,       dtype=float)
    y1       = np.array(y1,       dtype=float)
    z1       = np.array(z1,       dtype=float)
    headings = np.array(headings, dtype=float)

    # set waypoint speed as midpoint of desired range
    v_wp = (v_min + v_max) / 2.0
    vx   = v_wp * np.cos(headings)
    vy   = v_wp * np.sin(headings)
    vz   = np.zeros(n_waypoints)
    ax   = np.zeros(n_waypoints)
    ay   = np.zeros(n_waypoints)
    az   = np.zeros(n_waypoints)

    cost = find_cost_function(poly_deg=5, min_deriv=3, n_legs=n_legs)

    xm = sympy.Matrix([list(x1), list(vx), list(ax)])
    ym = sympy.Matrix([list(y1), list(vy), list(ay)])
    zm = sympy.Matrix([list(z1), list(vz), list(az)])

    # initial leg time guess from distance / v_wp
    dists = np.sqrt(np.diff(x1)**2 + np.diff(y1)**2 + np.diff(z1)**2)
    T0    = np.maximum(dists / v_wp, 0.5)

    sol = scipy.optimize.minimize(
        lambda T: cost['f_J'](T, xm, ym, zm, k_time),
        T0,
        method='L-BFGS-B',
        bounds=[(0.1, 100.0)] * n_legs,
        options={'maxiter': 500, 'ftol': 1e-9}
    )

    T_opt  = sol.x
    p_optx = cost['f_p_x'](T_opt, xm, k_time)
    p_opty = cost['f_p_y'](T_opt, ym, k_time)
    p_optz = cost['f_p_z'](T_opt, zm, k_time)

    traj_x = compute_trajectory(p_optx, T_opt)
    traj_y = compute_trajectory(p_opty, T_opt)
    traj_z = compute_trajectory(p_optz, T_opt)

    # speed report
    t  = traj_x['t']
    dt = np.diff(t)
    vx_t = np.diff(traj_x['x']) / dt
    vy_t = np.diff(traj_y['x']) / dt
    spd  = np.sqrt(vx_t**2 + vy_t**2)
    print("-" * 45)
    print(f"REPORT")
    print(f"Observed Speed Range: {spd.min():.2f} to {spd.max():.2f} m/s")
    print(f"Target Window:        {v_min} to {v_max} m/s")
    print(f"Leg Times: {np.round(T_opt, 2)}")
    print(f"Total Time: {sum(T_opt):.2f} s")
    print("-" * 45)

    return {
        'x':   traj_x['x'],
        'y':   traj_y['x'],
        'z':   traj_z['x'],
        't_x': traj_x['t'],
        't_y': traj_y['t'],
        't_z': traj_z['t'],
        'T_legs': T_opt
    }